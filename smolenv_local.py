"""SmolDataEnvs as a LOCAL environment: sandboxed executor, rollouts, and the reward.

This replaces FineEnvs' `SandboxRunner` (Hugging Face Sandboxes: a remote VM with
network and your HF token) with a subprocess on the machine you own. The eval
harness and the GRPO reward function both import from here, so evaluation and
training grade with the same code path.

Two rollout shapes:

  program  One turn, exactly as FineEnvs scripts/rollout.py: the model writes one
           Python program; it runs in the task's input dir; the LAST printed line is
           the answer; command-shaped answers score 0; the dataset grader scores it.
  agent    Multi-turn, as in SmolDataEnvs-sft and the Harbor tasks: the model calls a
           `bash` tool (same schema and system prompt as the SFT trajectories) up to
           `max_turns` times and must write its answer to /workdir/answer.txt. The
           answer file is graded exactly as Harbor's tests/test.sh does.

Isolation backends (auto-detected, strongest first):

  sandbox-exec  macOS. Denies all network, denies writes outside the sandbox dir,
                denies reads of $HOME except the interpreter, the guard dir and this
                task's own input files (so the answer key is unreadable).
  bwrap         Linux (DGX Spark). New user/net/pid namespaces; only /usr, /etc, the
                interpreter, the task's inputs (read-only, at /home/user/input) and a
                writable /workdir are visible. Untested on this Mac.
  docker        Linux, when bwrap is blocked (Ubuntu 24.04 AppArmor denies unprivileged
                user namespaces, as on the DGX Sparks). One throwaway container per call:
                --network none, non-root host uid, no capabilities, read-only rootfs,
                memory/pid/cpu caps. The image is an empty rootfs (built locally with
                `python smolenv_local.py build-docker-image`, no download); the host's
                /usr and this venv are bind-mounted read-only, so programs run on exactly
                the same Python and packages as the harness. Only the task's inputs
                (read-only, at /home/user/input) and a writable /workdir are visible.
  unshare       Linux fallback: network namespace only. Filesystem NOT confined.
  none          rlimits + a Python audit hook that refuses sockets. Mocks only.

Every backend also gets: a wall-clock timeout that kills the whole process group,
CPU-time / file-size / open-file rlimits, a scrubbed environment (no tokens, no API
keys), and output capped at `max_output_chars`.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from checker_v2 import check as check_v2  # noqa: E402
from vendor.fineenvs_rollout import (  # noqa: E402
    build_prompt,
    extract_code,
    grade_with_method,
    last_line,
    looks_like_a_command,
)

GUARD_DIR = ROOT / "smolenv_guard"
DOCKER_IMAGE = os.environ.get("SMOLENV_DOCKER_IMAGE", "smolenv-sandbox:local")
DOCKER_LABEL = "smolenv.sandbox=1"   # cleanup: docker ps -q --filter label=smolenv.sandbox=1 | xargs -r docker kill

# Verbatim from FineEnvs/SmolDataEnvs-sft (identical across all 4,677 rows).
SFT_SYSTEM = (
    "You are an autonomous data-analysis agent operating in a sandboxed Linux container. Your only tool is "
    "`bash`. The dataset files are in /home/user/input/. Python 3 + pandas + numpy + scikit-learn + scipy are "
    "pre-installed.\n\nWork the problem step-by-step: first inspect the data (ls, head, shape, dtypes), then "
    "plan, then compute, then submit.\n\nTo submit your final answer you MUST call the `bash` tool to write it "
    "to /workdir/answer.txt, e.g. `echo -n \"<value>\" > /workdir/answer.txt`. Keep the answer short and "
    "concise. Do NOT end your turn without submitting."
)
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command in the sandbox and return its combined stdout+stderr. The shell is non-stateful "
            "between calls (each call is a fresh `bash -c`). Use it to explore files (ls, head, cat), run Python "
            "(`python3 -c ...`), and write the final answer (`echo -n \"<value>\" > /workdir/answer.txt`)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to execute."}},
            "required": ["command"],
        },
    },
}
SFT_MAX_TOOL_OUTPUT = 8000  # longest tool message in the SFT trajectories is 8,016 chars


# ── sandbox ──────────────────────────────────────────────────────────────────

@dataclass
class SandboxConfig:
    backend: str = "auto"
    timeout_s: float = 90.0          # rollout.py: `timeout 90 python3 /tmp/solve.py`
    max_output_chars: int = 20_000
    cpu_s: int = 0                   # 0 = 4 x timeout + 10
    fsize_mb: int = 512
    mem_mb: int = 16_384             # Linux only (RLIMIT_AS); HF cpu-basic sandboxes have 16 GB
    cpus: float = 4.0                # docker backend only (--cpus)
    pids: int = 512                  # docker backend only (--pids-limit)
    # interpreter that has pandas & co. SMOLENV_SANDBOX_PYTHON points at a HOST venv when this
    # code runs inside a container that starts sandboxes through the host's docker socket
    python: str = os.environ.get("SMOLENV_SANDBOX_PYTHON", sys.executable)
    keep: bool = True                # keep sandbox dirs (they hold program, stdout, answer.txt)


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    seconds: float
    backend: str


def _which(x: str) -> str | None:
    return shutil.which(x)


_BACKEND_CACHE: dict[str, str] = {}


def detect_backend(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if "auto" in _BACKEND_CACHE:
        return _BACKEND_CACHE["auto"]
    chosen = "none"
    if platform.system() == "Darwin" and _which("sandbox-exec"):
        ok = subprocess.run(["sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
                            capture_output=True).returncode == 0
        chosen = "sandbox-exec" if ok else "none"
    elif platform.system() == "Linux":
        if _which("bwrap") and subprocess.run(
            ["bwrap", "--unshare-all", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "true"],
            capture_output=True).returncode == 0:
            chosen = "bwrap"
        elif docker_ready():
            chosen = "docker"
        elif _which("unshare") and subprocess.run(["unshare", "-rn", "true"], capture_output=True).returncode == 0:
            chosen = "unshare"
    _BACKEND_CACHE["auto"] = chosen
    return chosen


def _docker_ro(src: str, dst: str | None = None) -> list[str]:
    src, dst = str(src), str(dst or src)
    if "," in src or "," in dst:
        raise ValueError(f"path with a comma cannot be bind-mounted safely: {src}")
    return ["--mount", f"type=bind,src={src},dst={dst},readonly"]


def _docker_system_mounts() -> list[str]:
    """Host /usr (merged-usr Ubuntu: /bin, /lib, /sbin are symlinks into it) plus the few
    /etc entries that programs in /usr need. Never $HOME, never /etc as a whole."""
    m = _docker_ro("/usr")
    for p in ("/etc/ld.so.cache", "/etc/alternatives", "/etc/fonts", "/etc/localtime"):
        if os.path.exists(p):
            m += _docker_ro(os.path.realpath(p), p)
    for p in ("/bin", "/lib", "/lib64", "/sbin"):     # non-merged-usr hosts only
        if os.path.isdir(p) and not os.path.islink(p):
            m += _docker_ro(p)
    return m


def docker_ready() -> bool:
    """True when the docker CLI works, the sandbox image exists, and a no-network
    container can start with the host /usr mounted."""
    if not _which("docker"):
        return False
    try:
        if subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE], capture_output=True, timeout=30).returncode:
            return False
        return subprocess.run(["docker", "run", "--rm", "--network", "none", "--log-driver", "none",
                               *_docker_system_mounts(), DOCKER_IMAGE, "/usr/bin/true"],
                              capture_output=True, timeout=60).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def build_docker_image(image: str = DOCKER_IMAGE) -> None:
    """Build the sandbox image from an (almost) empty rootfs with `docker import`: mount
    points, merged-usr symlinks, and a passwd/group naming the host uid. No network, no
    base image: every binary comes from the host /usr mounted read-only at run time."""
    import io
    import tarfile

    uid, gid = os.getuid(), os.getgid()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        def add_dir(name, mode=0o755):
            ti = tarfile.TarInfo(name); ti.type = tarfile.DIRTYPE; ti.mode = mode; tar.addfile(ti)

        def add_file(name, text, mode=0o644):
            data = text.encode(); ti = tarfile.TarInfo(name); ti.size = len(data); ti.mode = mode
            tar.addfile(ti, io.BytesIO(data))

        for d in ("usr", "etc", "proc", "dev", "sys", "run", "workdir", "home", "home/user", "home/user/input"):
            add_dir(d)
        add_dir("tmp", 0o1777)
        for link in ("bin", "lib", "lib64", "sbin"):
            host = Path("/") / link
            if host.is_symlink():
                ti = tarfile.TarInfo(link); ti.type = tarfile.SYMTYPE; ti.linkname = os.readlink(host); tar.addfile(ti)
            elif not host.exists():
                continue
            else:
                add_dir(link)
        add_file("etc/passwd", f"root:x:0:0:root:/root:/usr/sbin/nologin\nsandbox:x:{uid}:{gid}:sandbox:/workdir:/bin/bash\n")
        add_file("etc/group", f"root:x:0:\nsandbox:x:{gid}:\n")
        add_file("etc/hosts", "127.0.0.1 localhost\n::1 localhost\n")
        add_file("etc/nsswitch.conf", "passwd: files\ngroup: files\nhosts: files\n")
    r = subprocess.run(["docker", "import", "--change", "WORKDIR /workdir", "-", image], input=buf.getvalue(),
                       capture_output=True)
    if r.returncode:
        raise SystemExit(f"docker import failed: {r.stderr.decode()[-400:]}")
    _BACKEND_CACHE.clear()
    print(f"built {image}; docker_ready() = {docker_ready()}")


def _real(p) -> str:
    return os.path.realpath(str(p))


def _interpreter_hops(python: str) -> set[str]:
    """Every install root the interpreter's symlink chain passes through. sandbox-exec
    checks the path as written at each hop (uv's `cpython-3.11-...` is itself a symlink
    to `cpython-3.11.15-...`), so each hop's root must be readable, not just the target."""
    hops, p = set(), Path(python)
    for _ in range(10):
        hops.add(str(p.parent.parent))
        if not p.is_symlink():
            break
        t = Path(os.readlink(p))
        p = t if t.is_absolute() else (p.parent / t)
    hops.add(str(Path(_real(python)).parent.parent))
    return hops


MUTABLE_SUFFIXES = {".sqlite", ".sqlite3", ".db", ".h5", ".hdf5"}  # libraries open these read-write


def _stage_file(src: Path, dst: Path, confined_writes: bool) -> None:
    """Put one input file in the sandbox without ever exposing the shared copy to writes.

    macOS: APFS clone (`cp -c`), instant and copy-on-write. Linux: symlink to the shared
    copy when the backend denies writes to it (bwrap binds it read-only), except for
    SQLite/HDF5 files, which libraries open read-write, so those are copied."""
    if platform.system() == "Darwin":
        if subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True).returncode == 0:
            return
        shutil.copy2(src, dst)
        return
    if confined_writes and src.suffix.lower() not in MUTABLE_SUFFIXES:
        dst.symlink_to(_real(src))
        return
    if subprocess.run(["cp", "--reflink=auto", str(src), str(dst)], capture_output=True).returncode != 0:
        shutil.copy2(src, dst)


class LocalSandbox:
    """One task's scratch area: input/ (the task's files), workdir/ (answer.txt), tmp/."""

    def __init__(self, row: dict, data_root: Path | None, work_parent: Path, cfg: SandboxConfig):
        self.row, self.cfg = row, cfg
        self.backend = detect_backend(cfg.backend)
        self.root = Path(_real(work_parent)) / f"{row['task_id']}-{uuid.uuid4().hex[:8]}"
        self.input, self.workdir, self.tmp = self.root / "input", self.root / "workdir", self.root / "tmp"
        for d in (self.input, self.workdir, self.tmp):
            d.mkdir(parents=True, exist_ok=True)
        self.prefix_dir = (Path(_real(data_root)) / row["bucket_prefix"]) if data_root else None
        self.missing_files: list[str] = []
        confined_writes = self.backend in ("sandbox-exec", "bwrap", "docker")
        names = list(row.get("files", []))
        if not names and self.prefix_dir and self.prefix_dir.is_dir():
            # empty `files` column (424 tasks): Harbor pulls the whole prefix, so expose all of it
            names = sorted(p.name for p in self.prefix_dir.iterdir() if p.is_file())
        elif not names:
            self.missing_files.append("(empty files list; whole prefix not staged)")
        for f in names:
            src = self.prefix_dir / f if self.prefix_dir else None
            if src is None or not src.exists():
                self.missing_files.append(f)
                continue
            _stage_file(src, self.input / Path(f).name, confined_writes)
        self._n = 0
        self._stdin = False
        self._container = ""

    # path mapping: the model is told /home/user/input and /workdir, as in training
    def _to_host(self, s: str) -> str:
        if self.backend in ("bwrap", "docker"):
            return s
        return s.replace("/home/user/input", str(self.input)).replace("/workdir", str(self.workdir))

    def _to_model(self, s: str) -> str:
        if self.backend in ("bwrap", "docker"):
            return s
        return (s.replace(str(self.input), "/home/user/input").replace(_real(self.input), "/home/user/input")
                 .replace(str(self.workdir), "/workdir").replace(_real(self.workdir), "/workdir")
                 .replace(_real(self.root), "/sandbox").replace(str(self.root), "/sandbox"))

    def _profile(self) -> Path:
        home = _real(Path.home())
        reads = {_real(self.root), _real(GUARD_DIR), _real(sys.prefix), _real(sys.base_prefix),
                 _real(Path(self.cfg.python).parent.parent), *_interpreter_hops(self.cfg.python)}
        if self.prefix_dir and self.prefix_dir.exists():
            reads.add(_real(self.prefix_dir))
        allow_reads = " ".join(f'(subpath "{p}")' for p in sorted(reads))
        prof = f"""(version 1)
(allow default)
(deny network*)
(allow network* (remote unix-socket))
(allow network* (local unix-socket))
(deny file-write*)
(allow file-write* (subpath "{_real(self.root)}") (literal "/dev/null") (regex #"^/dev/tty") (literal "/dev/dtracehelper"))
(deny file-read-data (subpath "{home}"))
(deny file-read-xattr (subpath "{home}"))
(allow file-read-data (literal "{home}") {allow_reads})
(allow file-read-xattr {allow_reads})
"""
        # file-read-metadata (lstat) stays allowed everywhere: SQLite resolves its path by
        # lstat-ing every ancestor directory, and metadata reveals names/sizes, not contents.
        p = self.root / "sandbox.sb"
        p.write_text(prof)
        return p

    @property
    def _native_paths(self) -> bool:
        """The program sees /home/user/input and /workdir as real paths (no rewriting)."""
        return self.backend in ("bwrap", "docker")

    def _env(self) -> dict:
        env = {
            "PATH": f"{Path(self.cfg.python).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(self.root) if not self._native_paths else "/workdir",
            "TMPDIR": str(self.tmp) if not self._native_paths else "/tmp",
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(self.tmp) if not self._native_paths else "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(GUARD_DIR),          # sitecustomize.py: socket audit hook
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "SMOLENV_SANDBOX": "1",
        }
        return env

    def _wrap(self, argv: list[str], cwd_model: str) -> tuple[list[str], str]:
        cpu = self.cfg.cpu_s or int(4 * self.cfg.timeout_s + 10)
        mem = self.cfg.mem_mb if platform.system() == "Linux" else 0
        launch = [self.cfg.python, str(GUARD_DIR / "launch.py"), str(cpu), str(self.cfg.fsize_mb), str(mem), "--"]
        if self.backend == "sandbox-exec":
            return ["sandbox-exec", "-f", str(self._profile()), *launch, *argv], self._to_host(cwd_model)
        if self.backend == "bwrap":
            ro = []
            for p in ("/usr", "/bin", "/lib", "/lib64", "/etc", "/sbin"):
                if os.path.exists(p):
                    ro += ["--ro-bind", p, p]
            for p in {_real(sys.prefix), _real(sys.base_prefix), _real(GUARD_DIR), _real(Path(self.cfg.python).parent.parent)}:
                ro += ["--ro-bind", p, p]
            return ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", *ro,
                    "--ro-bind", _real(self.input), "/home/user/input",
                    *( ["--ro-bind", _real(self.prefix_dir), _real(self.prefix_dir)] if self.prefix_dir and self.prefix_dir.exists() else []),
                    "--bind", _real(self.workdir), "/workdir", "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev",
                    "--chdir", cwd_model, *launch, *argv], str(self.root)
        if self.backend == "docker":
            self._container = f"smolsb-{uuid.uuid4().hex[:12]}"
            mounts = [*_docker_system_mounts(), *_docker_ro(_real(self.input), "/home/user/input"),
                      "--mount", f"type=bind,src={_real(self.workdir)},dst=/workdir"]
            own = {_real(sys.prefix)} if self.cfg.python == sys.executable else set()  # else: container paths
            for p in sorted(own | {_real(GUARD_DIR), _real(Path(self.cfg.python).parent.parent)}):
                if not p.startswith("/usr"):
                    mounts += _docker_ro(p)
            if self.prefix_dir and self.prefix_dir.exists():   # staged inputs are symlinks into it
                mounts += _docker_ro(_real(self.prefix_dir))
            envs = [x for k, v in self._env().items() for x in ("-e", f"{k}={v}")]
            # PID 1 is `timeout`: when it exits, the kernel kills everything else in the
            # container's pid namespace, so the program dies even if this harness dies
            backstop = ["/usr/bin/timeout", "-s", "KILL", str(int(self.cfg.timeout_s) + 10)]
            return ["docker", "run", "--rm", "--name", self._container, "--label", DOCKER_LABEL,
                    "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges", "--read-only",
                    "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g", "--hostname", "sandbox",
                    "--memory", f"{self.cfg.mem_mb}m", "--memory-swap", f"{self.cfg.mem_mb}m",
                    "--pids-limit", str(self.cfg.pids), "--cpus", str(self.cfg.cpus), "--log-driver", "none",
                    *( ["-i"] if self._stdin else []), "-w", cwd_model, *mounts, *envs, DOCKER_IMAGE,
                    *backstop, *launch, *argv], str(self.root)
        if self.backend == "unshare":
            return ["unshare", "-rn", *launch, *argv], self._to_host(cwd_model)
        return [*launch, *argv], self._to_host(cwd_model)

    def _exec(self, argv: list[str], cwd_model: str, stdin_text: str | None = None) -> ExecResult:
        self._n += 1
        out_p, err_p = self.root / f"call{self._n:02d}.stdout", self.root / f"call{self._n:02d}.stderr"
        self._stdin = stdin_text is not None
        full, cwd = self._wrap(argv, cwd_model)
        # the docker CLI itself is not sandboxed; the container gets the scrubbed env via -e
        popen_env = self._env() if self.backend != "docker" else {
            "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": os.environ.get("HOME", "/"),
            **{k: os.environ[k] for k in ("DOCKER_HOST", "DOCKER_CONTEXT") if k in os.environ}}
        t0 = time.time()
        timed_out = False
        with open(out_p, "wb") as fo, open(err_p, "wb") as fe:
            proc = subprocess.Popen(full, cwd=cwd, env=popen_env, stdout=fo, stderr=fe,
                                    stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                                    start_new_session=True)
            try:
                if stdin_text is not None:
                    proc.stdin.write(stdin_text.encode())
                    proc.stdin.close()
                proc.wait(timeout=self.cfg.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                if self.backend == "docker":   # killing the CLI would leave the container running
                    subprocess.run(["docker", "kill", self._container], capture_output=True, timeout=60)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        dt = time.time() - t0

        def tail(p: Path) -> str:
            b = p.read_bytes()
            s = b.decode("utf-8", errors="replace")
            if len(s) > self.cfg.max_output_chars:
                s = s[: self.cfg.max_output_chars // 2] + "\n...[truncated]...\n" + s[-self.cfg.max_output_chars // 2:]
            return self._to_model(s)

        return ExecResult(tail(out_p), tail(err_p), proc.returncode, timed_out, dt, self.backend)

    def run_python(self, code: str, cwd_model: str = "/home/user/input") -> ExecResult:
        name = f".prog{self._n + 1:02d}.py"
        if self._native_paths:   # must be visible inside the namespace/container: put it in /workdir
            (self.workdir / name).write_text(code)
            return self._exec([self.cfg.python, f"/workdir/{name}"], cwd_model)
        prog = self.root / name
        prog.write_text(self._to_host(code))
        return self._exec([self.cfg.python, str(prog)], cwd_model)

    def run_bash(self, command: str, cwd_model: str = "/workdir") -> ExecResult:
        return self._exec(["/bin/bash", "-c", self._to_host(command)], cwd_model)

    def answer_txt(self) -> str | None:
        p = self.workdir / "answer.txt"
        if p.is_symlink() or not p.is_file():
            return None
        return p.read_text(errors="replace")[:4096]


# ── grading (the ONE place rewards come from) ────────────────────────────────

def score(row: dict, prediction: str | None) -> dict:
    """Grade a prediction with the official grader (v1) and the strict checker (v2)."""
    pred = (prediction or "").strip()
    r1, m1 = grade_with_method(row, pred)
    r2, m2 = check_v2(row["answer"], pred, row["reward_mode"], row["atol"], row["rtol"])
    return {"reward_v1": r1, "method_v1": m1, "reward_v2": r2, "reason_v2": m2}


# ── program mode: FineEnvs rollout.py, locally ───────────────────────────────

def program_messages(row: dict) -> list[dict]:
    return build_prompt(row)


def program_rollout(sb: LocalSandbox, row: dict, completion: str) -> dict:
    code = extract_code(completion)
    try:
        compile(code, "<rollout>", "exec")
    except SyntaxError as exc:
        return {"prediction": "", "ran": 0.0, "stderr": f"syntax: {exc}", "timed_out": False,
                "exec_seconds": 0.0, "filtered": None, **score(row, "")}
    res = sb.run_python(code)
    pred = last_line(res.stdout)
    if looks_like_a_command(pred):
        return {"prediction": pred, "ran": 1.0, "stderr": "answer is a command, not a value", "timed_out": res.timed_out,
                "exec_seconds": res.seconds, "filtered": "command_shaped",
                "reward_v1": 0.0, "method_v1": "command_filter", **{k: v for k, v in score(row, pred).items() if k.endswith("v2")}}
    return {"prediction": pred, "ran": 1.0 if pred else 0.0, "stderr": res.stderr[-400:], "timed_out": res.timed_out,
            "exec_seconds": res.seconds, "filtered": None, **score(row, pred)}


# ── agent mode: Harbor / SFT-shaped multi-turn loop ──────────────────────────

def agent_messages(row: dict) -> list[dict]:
    return [{"role": "system", "content": SFT_SYSTEM}, {"role": "user", "content": row["instruction"]}]


def tool_output(res: ExecResult, limit: int = SFT_MAX_TOOL_OUTPUT) -> str:
    text = (res.stdout or "") + (res.stderr or "")
    if res.timed_out:
        text += f"\n[killed: exceeded {res.seconds:.0f}s]"
    if not text.strip():
        return f"(empty output, rc={res.returncode})"
    return text[:limit]


def run_tool_call(sb: LocalSandbox, name: str, arguments) -> tuple[str, ExecResult | None]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return f"error: tool arguments are not valid JSON: {arguments[:200]}", None
    if name == "bash":
        cmd = (arguments or {}).get("command")
        if not isinstance(cmd, str):
            return "error: bash requires a string `command`", None
        res = sb.run_bash(cmd)
        return tool_output(res), res
    if name == "python":
        code = (arguments or {}).get("code")
        if not isinstance(code, str):
            return "error: python requires a string `code`", None
        res = sb.run_python(code, cwd_model="/workdir")
        return tool_output(res), res
    return f"error: unknown tool {name!r}; the only tool is `bash`", None


def agent_final(sb: LocalSandbox, row: dict) -> dict:
    """Harbor semantics: no /workdir/answer.txt (or empty) -> correctness 0, submission 0."""
    ans = sb.answer_txt()
    submitted = bool(ans and ans.strip())
    return {"prediction": (ans or "").strip(), "submitted": submitted,
            **(score(row, ans.strip()) if submitted else {"reward_v1": 0.0, "method_v1": "no_answer_file",
                                                          "reward_v2": 0.0, "reason_v2": "no_answer_file"})}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="SmolDataEnvs local sandbox utilities")
    ap.add_argument("cmd", choices=("build-docker-image", "detect"))
    a = ap.parse_args()
    if a.cmd == "build-docker-image":
        build_docker_image()
    else:
        print(detect_backend("auto"))
