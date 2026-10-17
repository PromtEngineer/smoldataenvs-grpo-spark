"""Local eval harness for SmolDataEnvs: any OpenAI-compatible endpoint, a sandboxed local
executor, and two graders (the dataset's own grader = v1, checker_v2 = strict).

Run a model (vLLM / SGLang / llama.cpp server / a frontier API with an OpenAI-compatible route):

    .venv/bin/python eval_harness.py run --endpoint http://SPARK_HOST:8000/v1 \
        --model Qwen/Qwen3.5-2B --split eval --mode program --no-think --out runs/qwen35-2b-eval-program

Prove the plumbing with no model at all (mock "models", no network, no GPU):

    .venv/bin/python eval_harness.py selftest

Outputs per run directory:
    results.jsonl      one line per (task, sample): prediction, both rewards, tokens, timings
    summary.json       accuracy by tier / level / reward_mode / seen-vs-unseen dataset, both checkers
    summary.txt        the same as a table
    sandboxes/         each task's program(s), stdout/stderr, answer.txt  (kept, never deleted)
    transcripts/       agent-mode conversations (one JSON per task/sample)

Task data (the CSV/SQLite/... files) must be staged under --data-root/<bucket_prefix>/
by scripts/stage_task_files.py first; without it, only mocks that do not read data work.
"""
from __future__ import annotations

import argparse
import json
import shutil
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import smolenv_local as env  # noqa: E402

SPLITS = ("train", "test", "eval")


# ── tasks ────────────────────────────────────────────────────────────────────

def load_tasks(split: str | None = None, slice_name: str | None = None, limit: int = 0,
               tiers: str = "", seed: int = 0, tasks_file: Path | None = None) -> list[dict]:
    import pandas as pd

    if tasks_file:
        rows = [json.loads(l) for l in Path(tasks_file).read_text().splitlines() if l.strip()]
    else:
        frames = []
        for s in SPLITS:
            d = pd.read_parquet(ROOT / f"data/SmolDataEnvs/data/{s}-00000-of-00001.parquet")
            d["split"] = s
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
        train_ds = set(df[df.split == "train"].kaggle_dataset)
        df["seen_dataset_in_train"] = df.kaggle_dataset.isin(train_ds)
        if slice_name:
            ids = json.loads((ROOT / "data/slices.json").read_text())[slice_name]
            df = df[df.task_id.isin(set(ids))]
        elif split:
            df = df[df.split == split]
        rows = df.to_dict(orient="records")
        for r in rows:
            r["files"] = list(r["files"])
    if tiers:
        keep = {t.strip() for t in tiers.split(",") if t.strip()}
        rows = [r for r in rows if r["difficulty_tier"] in keep]
    if limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


# ── model clients ────────────────────────────────────────────────────────────

class OpenAICompatClient:
    """Minimal /chat/completions client (stdlib only). Retries 429/5xx; never raises on
    a bad completion, only on transport failure after retries."""

    def __init__(self, endpoint: str, model: str, api_key: str = "EMPTY", timeout: float = 600,
                 extra_body: dict | None = None, retries: int = 4):
        self.url = endpoint.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.extra_body, self.retries = extra_body or {}, retries
        self._opener = urllib.request.urlopen  # swappable in tests

    def chat(self, messages, tools=None, max_tokens=1024, temperature=0.0, top_p=1.0, seed=None, task=None) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": top_p, **self.extra_body}
        if seed is not None:
            body["seed"] = seed
        if tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        data = json.dumps(body).encode()
        last_err = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.url, data=data, method="POST", headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
            t0 = time.time()
            try:
                with self._opener(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode())
                return parse_chat_response(payload, time.time() - t0)
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read()[:300]!r}"
                if e.code not in (408, 409, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(60, 2 ** attempt + random.random()))
        raise RuntimeError(f"model call failed: {last_err}")


def parse_chat_response(payload: dict, seconds: float) -> dict:
    ch = (payload.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        calls.append({"id": tc.get("id") or f"call_{len(calls)}", "type": "function",
                      "function": {"name": fn.get("name"), "arguments": fn.get("arguments") or "{}"}})
    usage = payload.get("usage") or {}
    return {
        "content": msg.get("content") or "",
        # vLLM on the Spark puts thinking under `reasoning`, not `reasoning_content`
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        "tool_calls": calls,
        "finish_reason": ch.get("finish_reason"),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        # thinking tokens: vLLM/OpenAI report them inside completion_tokens (details.reasoning_tokens);
        # Gemini's OpenAI endpoint leaves them out of completion_tokens but bills them as output,
        # so fall back to total - prompt - completion
        "reasoning_tokens": _reasoning_tokens(usage),
        "seconds": seconds,
    }


# a file that really exists in $HOME, so "home_blocked" means the sandbox blocked it
HOME_PROBE = next((p for p in (Path.home() / n for n in (".zshrc", ".bashrc", ".profile")) if p.is_file()),
                  ROOT / "eval_harness.py")


class MockClient:
    """A stand-in 'model' that needs no GPU and no network. It reads the task row
    (including the gold answer) and emits a canned behaviour, so the plumbing and the
    graders can be proven end to end."""

    PROGRAM = {
        "gold": lambda r: f"Here you go.\n```python\nprint({str(r['answer'])!r})\n```",
        "wrong": lambda r: "```python\nprint('definitely not the answer')\n```",
        "crash": lambda r: "```python\nimport pandas as pd\n1/0\n```",
        "prose": lambda r: "I think the answer is probably in the data somewhere.",
        "command": lambda r: f"```python\nprint('echo -n \"{r['answer']}\" > /workdir/answer.txt')\n```",
        "timeout": lambda r: "```python\nwhile True:\n    pass\n```",
        "network": lambda r: (
            "```python\nimport socket, urllib.request\nres = []\n"
            "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=3); res.append('tcp_open')\n"
            "except Exception as e:\n    res.append('tcp_blocked')\n"
            "try:\n    urllib.request.urlopen('https://huggingface.co', timeout=5); res.append('http_open')\n"
            "except Exception as e:\n    res.append('http_blocked')\n"
            "print('NETWORK_OPEN' if any(x.endswith('open') for x in res) else 'NETWORK_BLOCKED')\n```"),
        "answer_key": lambda r: (
            "```python\nimport pandas as pd\n"
            f"df = pd.read_parquet({str(ROOT / 'data/SmolDataEnvs/data')!r} + '/' + {r['split']!r} + '-00000-of-00001.parquet')\n"
            f"print(df.set_index('task_id').loc[{r['task_id']!r}, 'answer'])\n```"),
        "escape": lambda r: (
            "```python\nimport os\nres = []\n"
            f"try:\n    open({str(ROOT / 'runs' / 'ESCAPE_PROBE_should_not_exist.txt')!r}, 'w').write('x'); res.append('write_escaped')\n"
            "except Exception:\n    res.append('write_blocked')\n"
            f"try:\n    open({str(HOME_PROBE)!r}).read(); res.append('home_read')\n"
            "except Exception:\n    res.append('home_blocked')\n"
            "print('ESCAPED:' + ','.join(res) if ('write_escaped' in res or 'home_read' in res) else 'CONTAINED')\n```"),
        "analyst": lambda r: f"```python\n{r['mock_solution']}\n```",
    }

    def __init__(self, kind: str):
        self.kind = kind
        self.model = f"mock:{kind}"

    def chat(self, messages, tools=None, task=None, **kw) -> dict:
        r = task
        base = {"reasoning": "", "finish_reason": "stop", "prompt_tokens": sum(len(str(m.get("content") or "")) for m in messages) // 4,
                "seconds": 0.0}
        if not tools:  # program mode
            text = self.PROGRAM[self.kind](r)
            return {**base, "content": text, "tool_calls": [], "completion_tokens": len(text) // 4}
        # agent mode: scripted tool calls, then a final message
        turn = sum(1 for m in messages if m["role"] == "assistant")
        q = lambda s: "'" + str(s).replace("'", "'\\''") + "'"  # noqa: E731
        script = {
            "gold": [f"ls /home/user/input", f"printf '%s' {q(r['answer'])} > /workdir/answer.txt"],
            "wrong": ["ls /home/user/input", "echo -n 'definitely not the answer' > /workdir/answer.txt"],
            "noanswer": ["ls /home/user/input"],
            "loop": ["ls /home/user/input"] * 1000,
            "network": ["(curl -s -m 5 -o /dev/null https://huggingface.co && echo NET_OPEN || echo NET_BLOCKED) > /workdir/answer.txt; "
                        "python3 -c \"import socket;socket.create_connection(('1.1.1.1',53),timeout=3);print('PY_NET_OPEN')\" "
                        ">> /workdir/answer.txt 2>/dev/null || true; cat /workdir/answer.txt"],
            "escape": [f"(cat {str(ROOT / 'data/SmolDataEnvs/grader.py')} >/dev/null 2>&1 && echo READ_PROJECT || echo CONTAINED) > /workdir/answer.txt; "
                       f"(touch {str(ROOT / 'runs' / 'ESCAPE_PROBE_should_not_exist.txt')} 2>/dev/null && echo ' WRITE_ESCAPED' >> /workdir/answer.txt) || true"],
            "analyst": [f"python3 << 'EOF'\n{r.get('mock_solution_agent', '')}\nEOF"],
        }[self.kind]
        if turn < len(script):
            cmd = script[turn]
            call = {"id": f"call_{turn}", "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}
            return {**base, "content": "", "tool_calls": [call], "completion_tokens": len(cmd) // 4,
                    "finish_reason": "tool_calls"}
        return {**base, "content": "Submitted.", "tool_calls": [], "completion_tokens": 3}


# ── one task ─────────────────────────────────────────────────────────────────

def _reasoning_tokens(usage: dict) -> int:
    det = usage.get("completion_tokens_details") or {}
    if det.get("reasoning_tokens"):
        return int(det["reasoning_tokens"])
    extra = int(usage.get("total_tokens") or 0) - int(usage.get("prompt_tokens") or 0) - int(usage.get("completion_tokens") or 0)
    return max(extra, 0)


def run_one(client, row: dict, sample: int, args, run_dir: Path) -> dict:
    cfg = env.SandboxConfig(backend=args.sandbox, timeout_s=args.exec_timeout)
    sb = env.LocalSandbox(row, Path(args.data_root) if args.data_root else None, run_dir / "sandboxes", cfg)
    rec = {"task_id": row["task_id"], "split": row.get("split"), "sample": sample, "tier": row["difficulty_tier"],
           "level": int(row["difficulty_level"]), "reward_mode": row["reward_mode"], "atol": row["atol"], "rtol": row["rtol"],
           "kaggle_dataset": row["kaggle_dataset"], "seen_dataset_in_train": bool(row.get("seen_dataset_in_train", False)),
           "gold": row["answer"], "mode": args.mode, "model": client.model, "sandbox_backend": sb.backend,
           "sandbox_dir": str(sb.root.relative_to(run_dir)), "missing_files": sb.missing_files,
           "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
           "gen_seconds": 0.0, "n_model_calls": 0}
    if sb.missing_files and args.require_data:
        return {**rec, "reward_v1": None, "reward_v2": None, "error": f"missing data files: {sb.missing_files}"}
    seed = args.seed * 1000 + sample if args.temperature > 0 else None
    t0 = time.time()
    try:
        if args.mode == "program":
            resp = client.chat(env.program_messages(row), max_tokens=args.max_tokens, temperature=args.temperature,
                               top_p=args.top_p, seed=seed, task=row)
            rec.update(prompt_tokens=resp["prompt_tokens"], completion_tokens=resp["completion_tokens"],
                       reasoning_tokens=resp.get("reasoning_tokens", 0),
                       gen_seconds=resp["seconds"], n_model_calls=1, finish_reason=resp["finish_reason"],
                       completion=resp["content"][-4000:])
            rec.update(env.program_rollout(sb, row, resp["content"]))
            rec["truncated"] = resp["finish_reason"] == "length"
        else:
            messages = env.agent_messages(row)
            tools = [env.BASH_TOOL]
            n_tool_calls, hit_cap, ctx_out = 0, True, False
            for turn in range(args.max_turns):
                try:
                    resp = client.chat(messages, tools=tools, max_tokens=args.max_tokens, temperature=args.temperature,
                                       top_p=args.top_p, seed=seed, task=row)
                except RuntimeError as e:
                    if turn == 0 or "maximum context length" not in str(e):
                        raise
                    # the agent filled its own context window with tool output: that ends the
                    # episode (as in Harbor), and whatever it wrote to answer.txt is graded
                    ctx_out, hit_cap = True, False
                    break
                rec["prompt_tokens"] += resp["prompt_tokens"]
                rec["completion_tokens"] += resp["completion_tokens"]
                rec["reasoning_tokens"] += resp.get("reasoning_tokens", 0)
                rec["gen_seconds"] += resp["seconds"]
                rec["n_model_calls"] += 1
                asst = {"role": "assistant", "content": resp["content"]}
                if resp["tool_calls"]:
                    asst["tool_calls"] = resp["tool_calls"]
                messages.append(asst)
                if not resp["tool_calls"]:
                    hit_cap = False
                    break
                for tc in resp["tool_calls"]:
                    n_tool_calls += 1
                    out, _res = env.run_tool_call(sb, tc["function"]["name"], tc["function"]["arguments"])
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"], "content": out})
            rec.update(n_turns=rec["n_model_calls"], n_tool_calls=n_tool_calls, hit_max_turns=hit_cap,
                       context_exhausted=ctx_out)
            rec.update(env.agent_final(sb, row))
            tdir = run_dir / "transcripts"
            tdir.mkdir(exist_ok=True)
            (tdir / f"{row['task_id']}-s{sample}.json").write_text(json.dumps(messages, indent=1))
    except RuntimeError as e:  # transport failure: ungraded, never a silent zero
        rec.update(reward_v1=None, reward_v2=None, error=str(e)[:500])
    rec["wall_seconds"] = round(time.time() - t0, 3)
    rec["gen_tok_per_s"] = round(rec["completion_tokens"] / rec["gen_seconds"], 2) if rec["gen_seconds"] else None
    return rec


# ── summary ──────────────────────────────────────────────────────────────────

def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def summarize(records: list[dict], wall_seconds: float | None = None) -> dict:
    graded = [r for r in records if r.get("reward_v1") is not None]
    by_task = defaultdict(list)
    for r in graded:
        by_task[r["task_id"]].append(r)

    def block(rs: list[dict]) -> dict:
        tasks = defaultdict(list)
        for r in rs:
            tasks[r["task_id"]].append(r)
        n = len(tasks)
        # pass@1 = mean over tasks of the mean reward across samples
        p1_v1 = sum(sum(x["reward_v1"] for x in v) / len(v) for v in tasks.values())
        p1_v2 = sum(sum(x["reward_v2"] for x in v) / len(v) for v in tasks.values())
        out = {"n_tasks": n, "pass@1_v1": round(p1_v1 / n, 4) if n else None,
               "pass@1_v2": round(p1_v2 / n, 4) if n else None,
               "ci95_v1": wilson(p1_v1, n), "ci95_v2": wilson(p1_v2, n),
               "v1_minus_v2": round((p1_v1 - p1_v2) / n, 4) if n else None}
        ns = {len(v) for v in tasks.values()}
        if n and min(ns) > 1:
            k = min(ns)
            out[f"pass@{k}_v1"] = round(sum(pass_at_k(len(v), int(sum(x["reward_v1"] for x in v)), k) for v in tasks.values()) / n, 4)
        return out

    def group(key) -> dict:
        g = defaultdict(list)
        for r in graded:
            g[str(key(r))].append(r)
        return {k: block(v) for k, v in sorted(g.items())}

    tok = sum(r.get("completion_tokens", 0) for r in records)
    gen_s = sum(r.get("gen_seconds", 0.0) for r in records)
    s = {
        "model": records[0]["model"] if records else None,
        "mode": records[0]["mode"] if records else None,
        "n_records": len(records),
        "n_ungraded": len(records) - len(graded),
        "ungraded_reasons": dict(sorted(defaultdict(int, {}).items())),
        "overall": block(graded),
        "by_tier": group(lambda r: r["tier"]),
        "by_level": group(lambda r: r["level"]),
        "by_reward_mode": group(lambda r: r["reward_mode"]),
        "by_dataset_seen_in_train": group(lambda r: r["seen_dataset_in_train"]),
        "by_split": group(lambda r: r["split"]),
        "diagnostics": {
            "command_filtered": sum(1 for r in graded if r.get("filtered") == "command_shaped"),
            "timed_out": sum(1 for r in graded if r.get("timed_out")),
            "empty_prediction": sum(1 for r in graded if not r.get("prediction")),
            "truncated_completions": sum(1 for r in graded if r.get("truncated")),
            "agent_no_answer_file": sum(1 for r in graded if r.get("method_v1") == "no_answer_file"),
            "agent_hit_max_turns": sum(1 for r in graded if r.get("hit_max_turns")),
            "agent_context_exhausted": sum(1 for r in graded if r.get("context_exhausted")),
            "v1_rungs": dict(sorted(_count(r.get("method_v1") for r in graded if r["reward_v1"] == 1.0).items())),
            "v1_pass_v2_fail": sum(1 for r in graded if r["reward_v1"] == 1.0 and r["reward_v2"] == 0.0),
            "v2_pass_v1_fail": sum(1 for r in graded if r["reward_v2"] == 1.0 and r["reward_v1"] == 0.0),
            # WHY v1 accepted what v2 rejected. Tolerance misses ("numeric_miss") are expected
            # even for honest models; growth in not_a_bare_number / too_long / list_length /
            # v1 rung math_verify during training is the grader-gaming signal.
            "v1_only_by_v2_reason": dict(sorted(_count(r.get("reason_v2") for r in graded
                                                       if r["reward_v1"] == 1.0 and r["reward_v2"] == 0.0).items())),
            "v1_only_by_v1_rung": dict(sorted(_count(r.get("method_v1") for r in graded
                                                     if r["reward_v1"] == 1.0 and r["reward_v2"] == 0.0).items())),
        },
        "throughput": {
            "completion_tokens": tok, "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in records),
            "reasoning_tokens": sum(r.get("reasoning_tokens", 0) for r in records),
            "gen_seconds_sum": round(gen_s, 2),
            "per_request_tok_per_s": round(tok / gen_s, 2) if gen_s else None,
            "wall_seconds": round(wall_seconds, 2) if wall_seconds else None,
            "aggregate_tok_per_s": round(tok / wall_seconds, 2) if wall_seconds else None,
            "mean_completion_tokens": round(tok / len(records), 1) if records else None,
        },
    }
    reasons = defaultdict(int)
    for r in records:
        if r.get("reward_v1") is None:
            reasons[(r.get("error") or "unknown").split(":")[0][:60]] += 1
    s["ungraded_reasons"] = dict(reasons)
    return s


def _count(it) -> dict:
    d = defaultdict(int)
    for x in it:
        d[x] += 1
    return d


def summary_table(s: dict) -> str:
    lines = [f"model={s['model']}  mode={s['mode']}  records={s['n_records']}  ungraded={s['n_ungraded']}",
             f"{'slice':34} {'n':>5} {'v1 pass@1':>10} {'95% CI':>16} {'v2 pass@1':>10} {'v1-v2':>7}"]

    def row(name, b):
        lines.append(f"{name:34} {b['n_tasks']:5} {b['pass@1_v1'] if b['pass@1_v1'] is not None else '-':>10} "
                     f"{str(b['ci95_v1']):>16} {b['pass@1_v2'] if b['pass@1_v2'] is not None else '-':>10} "
                     f"{b['v1_minus_v2'] if b['v1_minus_v2'] is not None else '-':>7}")

    row("overall", s["overall"])
    for sect in ("by_tier", "by_dataset_seen_in_train", "by_reward_mode", "by_split"):
        for k, b in s[sect].items():
            row(f"{sect[3:]}={k}", b)
    lines.append(f"diagnostics: {json.dumps(s['diagnostics'])}")
    lines.append(f"throughput:  {json.dumps(s['throughput'])}")
    if s["ungraded_reasons"]:
        lines.append(f"ungraded:    {json.dumps(s['ungraded_reasons'])}")
    return "\n".join(lines)


# ── run ──────────────────────────────────────────────────────────────────────

def run(args, client=None, rows=None) -> dict:
    run_dir = Path(args.out).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if client is None:
        if args.mock:
            client = MockClient(args.mock)
        else:
            extra = json.loads(args.extra_body) if args.extra_body else {}
            if args.no_think:
                extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
            client = OpenAICompatClient(args.endpoint, args.model, os.environ.get(args.api_key_env, "EMPTY"),
                                        timeout=args.request_timeout, extra_body=extra)
    backend = env.detect_backend(args.sandbox)
    if backend == "none" and not args.mock and not args.unsafe_no_sandbox:
        raise SystemExit("no OS sandbox available (sandbox-exec/bwrap/unshare); refusing to run model code. "
                         "Install bubblewrap, or pass --unsafe-no-sandbox if you accept the risk.")
    if backend == "unshare" and not args.mock:
        print("WARNING: backend 'unshare' isolates the network only; the filesystem (including gold answers) "
              "is readable by model code. Prefer bwrap.", file=sys.stderr)
    rows = rows if rows is not None else load_tasks(args.split, args.slice, args.limit, args.tiers, args.seed)
    if getattr(args, "task_ids", None):  # a JSON list of task ids (e.g. one node's shard of a pre-screen)
        keep = json.loads(Path(args.task_ids).read_text())
        by_id = {r["task_id"]: r for r in rows}
        rows = [by_id[t] for t in keep if t in by_id]
    res_path = run_dir / "results.jsonl"
    done = set()
    if res_path.exists() and not args.resume:  # keep the old run, start a fresh file
        res_path.rename(run_dir / f"results.{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    if res_path.exists() and args.resume:
        keep, retry = [], 0
        for l in res_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("reward_v1") is None and not str(r.get("error", "")).startswith("missing data files"):
                    retry += 1   # transport or context failure: run it again
                    continue
                keep.append(l)
                done.add((r["task_id"], r["sample"]))
        if retry:
            shutil.copy2(res_path, run_dir / f"results.before-retry-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
            res_path.write_text("".join(k + "\n" for k in keep))
            print(f"retrying {retry} ungraded records (original kept as results.before-retry-*.jsonl)", flush=True)
    jobs = [(r, s) for r in rows for s in range(args.n_samples) if (r["task_id"], s) not in done]
    (run_dir / "config.json").write_text(json.dumps({k: v for k, v in vars(args).items() if k != "func"} |
                                                     {"sandbox_backend": backend, "n_tasks": len(rows)}, indent=1, default=str))
    print(f"{client.model}: {len(rows)} tasks x {args.n_samples} samples, mode={args.mode}, sandbox={backend}, "
          f"{len(jobs)} to run ({len(done)} resumed)", flush=True)
    lock = threading.Lock()
    t0 = time.time()
    with open(res_path, "a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(run_one, client, r, s, args, run_dir): (r["task_id"], s) for r, s in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:  # harness bug: record it, keep going
                tid, s = futs[fut]
                rec = {"task_id": tid, "sample": s, "reward_v1": None, "reward_v2": None, "model": client.model,
                       "mode": args.mode, "error": f"harness: {type(e).__name__}: {e}"[:500]}
            with lock:
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
            if args.verbose or i % 25 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {rec['task_id'][:26]:26} v1={rec.get('reward_v1')} v2={rec.get('reward_v2')} "
                      f"pred={str(rec.get('prediction'))[:30]!r}", flush=True)
    wall = time.time() - t0
    records = [json.loads(l) for l in res_path.read_text().splitlines() if l.strip()]
    s = summarize(records, wall if not done else None)
    (run_dir / "summary.json").write_text(json.dumps(s, indent=1))
    (run_dir / "summary.txt").write_text(summary_table(s) + "\n")
    print(summary_table(s))
    return s


# ── selftest: everything except a real model ─────────────────────────────────

def make_fixture(dir_: Path) -> list[dict]:
    """Three synthetic tasks with real files (CSV, SQLite, CSV list answer)."""
    import sqlite3

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    p1 = dir_ / "fixture__sales"
    p1.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"region": rng.choice(["West", "East", "North"], 500), "revenue": rng.gamma(2, 50, 500).round(2),
                       "category": rng.choice(["toys", "books", "garden", "tools"], 500), "price": rng.normal(20, 5, 500).round(2)})
    df.to_csv(p1 / "sales.csv", index=False)
    west = round(df[df.region == "West"].revenue.sum(), 2)
    top2 = df.groupby("category").price.mean().sort_values(ascending=False).index[:2].tolist()
    p2 = dir_ / "fixture__shop"
    p2.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p2 / "shop.sqlite")
    orders = pd.DataFrame({"customer_id": rng.integers(1, 60, 400), "amount": rng.gamma(2, 10, 400).round(2)})
    orders.to_sql("orders", con, index=False, if_exists="replace")
    con.close()
    n_gt3 = int((orders.groupby("customer_id").size() > 3).sum())
    base = {"split": "fixture", "difficulty_level": 2, "difficulty_tier": "easy", "atol": 0.0, "rtol": 0.0,
            "hf_bucket": "local", "seen_dataset_in_train": False, "package_tier": 1}
    rows.append(base | {"task_id": "fixture_sales_west", "bucket_prefix": "fixture__sales", "files": ["sales.csv"],
                        "kaggle_dataset": "fixture/sales", "reward_mode": "numeric", "answer": f"{west}", "atol": 0.01, "rtol": 0.001,
                        "question": "What is the total revenue for the West region?",
                        "mock_solution": "import pandas as pd\ndf = pd.read_csv('sales.csv')\nprint(round(df[df.region=='West'].revenue.sum(), 2))",
                        "mock_solution_agent": "import pandas as pd\ndf = pd.read_csv('/home/user/input/sales.csv')\n"
                                               "open('/workdir/answer.txt','w').write(str(round(df[df.region=='West'].revenue.sum(), 2)))"})
    rows.append(base | {"task_id": "fixture_shop_gt3", "bucket_prefix": "fixture__shop", "files": ["shop.sqlite"],
                        "kaggle_dataset": "fixture/shop", "reward_mode": "numeric", "answer": str(n_gt3),
                        "question": "How many customers placed more than 3 orders?",
                        "mock_solution": "import sqlite3, pandas as pd\ncon = sqlite3.connect('shop.sqlite')\n"
                                         "o = pd.read_sql('select * from orders', con)\nprint(int((o.groupby('customer_id').size() > 3).sum()))",
                        "mock_solution_agent": "import sqlite3, pandas as pd\ncon = sqlite3.connect('/home/user/input/shop.sqlite')\n"
                                               "o = pd.read_sql('select * from orders', con)\n"
                                               "open('/workdir/answer.txt','w').write(str(int((o.groupby('customer_id').size() > 3).sum())))"})
    rows.append(base | {"task_id": "fixture_sales_top2", "bucket_prefix": "fixture__sales", "files": ["sales.csv"],
                        "kaggle_dataset": "fixture/sales", "reward_mode": "list_csv", "answer": ", ".join(top2),
                        "question": "Which two categories have the highest mean price? Comma-separated, highest first.",
                        "mock_solution": "import pandas as pd\ndf = pd.read_csv('sales.csv')\n"
                                         "print(', '.join(df.groupby('category').price.mean().sort_values(ascending=False).index[:2]))",
                        "mock_solution_agent": "import pandas as pd\ndf = pd.read_csv('/home/user/input/sales.csv')\n"
                                               "open('/workdir/answer.txt','w').write(', '.join(df.groupby('category').price.mean().sort_values(ascending=False).index[:2]))"})
    for r in rows:
        r["instruction"] = (f"You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files "
                            f"and compute the answer.\n\nFiles (in /home/user/input, no subfolders):\n- {r['files'][0]}\n\n"
                            f"Question:\n{r['question']}\n\nWrite only that value to /workdir/answer.txt, then stop.")
    return rows


def http_client_selftest() -> dict:
    """Exercise OpenAICompatClient against canned responses (no socket is opened)."""
    import io

    canned = [
        urllib.error.HTTPError("http://x", 503, "busy", {}, io.BytesIO(b"overloaded")),
        {"choices": [{"message": {"content": "", "reasoning": "thinking...", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]},
            "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 120, "completion_tokens": 17}},
    ]
    sent = []

    class _Resp:
        def __init__(self, obj):
            self.obj = obj

        def read(self):
            return json.dumps(self.obj).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=None):
        sent.append(json.loads(req.data.decode()))
        item = canned.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Resp(item)

    c = OpenAICompatClient("http://unused.invalid/v1", "m", extra_body={"chat_template_kwargs": {"enable_thinking": False}}, retries=2)
    c._opener = fake_open
    orig_sleep = time.sleep
    time.sleep = lambda s: None
    try:
        out = c.chat([{"role": "user", "content": "hi"}], tools=[env.BASH_TOOL], max_tokens=64)
    finally:
        time.sleep = orig_sleep
    ok = (out["tool_calls"][0]["function"]["name"] == "bash" and out["reasoning"] == "thinking..."
          and out["completion_tokens"] == 17 and len(sent) == 2
          and sent[-1]["chat_template_kwargs"] == {"enable_thinking": False} and sent[-1]["tool_choice"] == "auto")
    return {"ok": ok, "retried_after_503": len(sent) == 2, "parsed": {k: out[k] for k in ("tool_calls", "reasoning", "completion_tokens")}}


def selftest(args) -> None:
    base_out = Path(args.out or ROOT / "runs/selftest")
    base_out.mkdir(parents=True, exist_ok=True)
    sandbox = getattr(args, "sandbox", "auto")
    report: dict = {"sandbox_backend": env.detect_backend(sandbox), "checks": {}}

    def ns(**kw):
        d = dict(mode="program", max_turns=8, max_tokens=1024, temperature=0.0, top_p=1.0, n_samples=1, concurrency=8,
                 exec_timeout=90, data_root=None, sandbox=sandbox, seed=0, require_data=False, resume=False, verbose=False,
                 unsafe_no_sandbox=False, mock=None, split="eval", slice=None, limit=0, tiers="")
        d.update(kw)
        return argparse.Namespace(**d)

    def check(name, cond, detail):
        report["checks"][name] = {"pass": bool(cond), **detail}
        print(f"{'PASS' if cond else 'FAIL'}  {name}: {json.dumps(detail)[:300]}", flush=True)

    eval_rows = load_tasks("eval")
    few = eval_rows[:12]

    # 1. program mode, gold stub over the WHOLE eval split -> every task must score 1.0 under v1
    s = run(ns(out=base_out / "program-gold-eval", mock="gold"), MockClient("gold"), eval_rows)
    check("program_gold_full_eval", s["overall"]["pass@1_v1"] == 1.0,
          {"n": s["overall"]["n_tasks"], "v1": s["overall"]["pass@1_v1"], "v2": s["overall"]["pass@1_v2"]})
    test_rows = load_tasks("test")
    s = run(ns(out=base_out / "program-gold-test", mock="gold", split="test"), MockClient("gold"), test_rows)
    check("program_gold_full_test", s["overall"]["pass@1_v1"] == 1.0,
          {"n": s["overall"]["n_tasks"], "v1": s["overall"]["pass@1_v1"], "v2": s["overall"]["pass@1_v2"]})
    # 2. wrong / crash / prose / command-shaped -> 0.0
    for kind in ("wrong", "crash", "prose", "command"):
        s = run(ns(out=base_out / f"program-{kind}", mock=kind), MockClient(kind), eval_rows if kind == "wrong" else few)
        check(f"program_{kind}_scores_zero", s["overall"]["pass@1_v1"] == 0.0 and s["overall"]["pass@1_v2"] == 0.0,
              {"n": s["overall"]["n_tasks"], "v1": s["overall"]["pass@1_v1"], "diag": s["diagnostics"]})
    # 3. timeout is enforced and the process group is killed
    s = run(ns(out=base_out / "program-timeout", mock="timeout", exec_timeout=3), MockClient("timeout"), few[:2])
    recs = [json.loads(l) for l in (base_out / "program-timeout/results.jsonl").read_text().splitlines()]
    check("program_timeout_killed", all(r["timed_out"] and r["exec_seconds"] < 8 for r in recs),
          {"exec_seconds": [r["exec_seconds"] for r in recs]})
    # 4. isolation: network, filesystem escape, answer-key read
    s = run(ns(out=base_out / "program-network", mock="network"), MockClient("network"), few[:2])
    recs = [json.loads(l) for l in (base_out / "program-network/results.jsonl").read_text().splitlines()]
    check("program_network_blocked", all(r["prediction"] == "NETWORK_BLOCKED" for r in recs), {"pred": [r["prediction"] for r in recs]})
    s = run(ns(out=base_out / "program-escape", mock="escape"), MockClient("escape"), few[:2])
    recs = [json.loads(l) for l in (base_out / "program-escape/results.jsonl").read_text().splitlines()]
    probe = ROOT / "runs" / "ESCAPE_PROBE_should_not_exist.txt"
    check("program_fs_contained", all(r["prediction"] == "CONTAINED" for r in recs) and not probe.exists(),
          {"pred": [r["prediction"] for r in recs], "probe_file_exists": probe.exists()})
    s = run(ns(out=base_out / "program-answer-key", mock="answer_key"), MockClient("answer_key"), few)
    check("program_answer_key_unreadable", s["overall"]["pass@1_v1"] == 0.0,
          {"v1": s["overall"]["pass@1_v1"], "sample_stderr": json.loads((base_out / "program-answer-key/results.jsonl").read_text().splitlines()[0])["stderr"][-160:]})
    # control: the same network probe OUTSIDE the sandbox, to show the block is the sandbox's doing
    import subprocess as sp

    ctl = sp.run([sys.executable, "-c", "import socket\ntry:\n socket.create_connection(('1.1.1.1',53),timeout=3);print('open')\nexcept Exception as e:\n print('blocked', type(e).__name__)"],
                 capture_output=True, text=True).stdout.strip()
    report["checks"]["network_control_outside_sandbox"] = {"pass": True, "result": ctl,
                                                           "note": "if 'open', the in-sandbox block above is the sandbox's doing"}
    print(f"INFO  network control outside the sandbox: {ctl}")
    # 5. real files: fixture tasks solved by a mock analyst that actually reads CSV / SQLite
    fx_dir = base_out / "fixture_data"
    fx = make_fixture(fx_dir)
    s = run(ns(out=base_out / "program-analyst-fixture", mock="analyst", data_root=str(fx_dir), require_data=True), MockClient("analyst"), fx)
    check("program_analyst_reads_files", s["overall"]["pass@1_v1"] == 1.0 and s["overall"]["pass@1_v2"] == 1.0,
          {"v1": s["overall"]["pass@1_v1"], "v2": s["overall"]["pass@1_v2"]})
    s = run(ns(out=base_out / "program-missing-data", mock="analyst", data_root=str(fx_dir / "nope"), require_data=True), MockClient("analyst"), fx)
    check("program_missing_data_is_ungraded_not_zero", s["n_ungraded"] == len(fx), {"ungraded": s["n_ungraded"]})
    # 6. agent mode (bash tool, OpenAI tool_calls, answer.txt)
    s = run(ns(out=base_out / "agent-gold-eval", mode="agent", mock="gold"), MockClient("gold"), eval_rows)
    check("agent_gold_full_eval", s["overall"]["pass@1_v1"] == 1.0, {"n": s["overall"]["n_tasks"], "v1": s["overall"]["pass@1_v1"], "v2": s["overall"]["pass@1_v2"]})
    for kind, expect in (("wrong", 0.0), ("noanswer", 0.0), ("loop", 0.0)):
        s = run(ns(out=base_out / f"agent-{kind}", mode="agent", mock=kind, max_turns=4), MockClient(kind), few)
        check(f"agent_{kind}_scores_zero", s["overall"]["pass@1_v1"] == expect,
              {"v1": s["overall"]["pass@1_v1"], "diag": {k: s["diagnostics"][k] for k in ("agent_no_answer_file", "agent_hit_max_turns")}})
    s = run(ns(out=base_out / "agent-network", mode="agent", mock="network"), MockClient("network"), few[:2])
    recs = [json.loads(l) for l in (base_out / "agent-network/results.jsonl").read_text().splitlines()]
    check("agent_network_blocked", all(r["prediction"] == "NET_BLOCKED" for r in recs), {"pred": [r["prediction"] for r in recs]})
    s = run(ns(out=base_out / "agent-escape", mode="agent", mock="escape"), MockClient("escape"), few[:2])
    recs = [json.loads(l) for l in (base_out / "agent-escape/results.jsonl").read_text().splitlines()]
    check("agent_fs_contained", all(r["prediction"] == "CONTAINED" for r in recs) and not probe.exists(), {"pred": [r["prediction"] for r in recs]})
    s = run(ns(out=base_out / "agent-analyst-fixture", mode="agent", mock="analyst", data_root=str(fx_dir), require_data=True), MockClient("analyst"), fx)
    check("agent_analyst_reads_files", s["overall"]["pass@1_v1"] == 1.0, {"v1": s["overall"]["pass@1_v1"], "v2": s["overall"]["pass@1_v2"]})
    # 7. HTTP client against canned responses
    h = http_client_selftest()
    check("http_client_parse_and_retry", h["ok"], h)
    # 8. pass@k plumbing: 3 samples of the gold stub
    s = run(ns(out=base_out / "program-gold-k3", mock="gold", n_samples=3), MockClient("gold"), few)
    check("pass_at_k_plumbing", s["overall"].get("pass@3_v1") == 1.0, {"overall": s["overall"]})

    report["all_pass"] = all(v["pass"] for v in report["checks"].values())
    (base_out / "selftest_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nselftest: {sum(v['pass'] for v in report['checks'].values())}/{len(report['checks'])} checks pass "
          f"-> {base_out / 'selftest_report.json'}")
    if not report["all_pass"]:
        raise SystemExit(1)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate a model (or a mock) on a split/slice")
    r.add_argument("--endpoint", help="OpenAI-compatible base URL, e.g. http://host:8000/v1")
    r.add_argument("--model", help="model name the server expects")
    r.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the key (default OPENAI_API_KEY; 'EMPTY' if unset)")
    r.add_argument("--extra-body", default="", help='JSON merged into every request, e.g. \'{"reasoning_effort":"low"}\'')
    r.add_argument("--no-think", action="store_true", help="send chat_template_kwargs.enable_thinking=false (vLLM/SGLang Qwen)")
    r.add_argument("--split", choices=SPLITS, default="eval")
    r.add_argument("--slice", help="named slice from data/slices.json (overrides --split), e.g. transfer_unseen_dataset")
    r.add_argument("--tiers", default="", help="comma list: easy,medium,hard")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--task-ids", help="JSON list of task ids to run, in that order (filters --split/--slice)")
    r.add_argument("--mode", choices=("program", "agent"), default="program")
    r.add_argument("--max-turns", type=int, default=12, help="agent mode: model calls per task (SFT p99 = 10 turns)")
    r.add_argument("--max-tokens", type=int, default=1024, help="per model call (FineEnvs uses 1024)")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--top-p", type=float, default=1.0)
    r.add_argument("--n-samples", type=int, default=1, help=">1 gives pass@k (use temperature > 0)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--exec-timeout", type=float, default=90.0, help="seconds per program / tool call (FineEnvs: 90)")
    r.add_argument("--request-timeout", type=float, default=600.0)
    r.add_argument("--data-root", default=str(ROOT / "data/task_files"), help="dir holding <bucket_prefix>/<file>")
    r.add_argument("--require-data", action=argparse.BooleanOptionalAction, default=None,
                   help="mark tasks whose files are not staged as ungraded (default: on for real models, off for mocks)")
    r.add_argument("--sandbox", default="auto", choices=("auto", "sandbox-exec", "bwrap", "docker", "unshare", "none"))
    r.add_argument("--unsafe-no-sandbox", action="store_true")
    r.add_argument("--mock", choices=sorted(set(MockClient.PROGRAM) | {"noanswer", "loop"}), help="use a mock model instead of an endpoint")
    r.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--out", required=True)
    t = sub.add_parser("selftest", help="prove harness + graders + sandbox with mock models (no network, no GPU)")
    t.add_argument("--out", default=str(ROOT / "runs/selftest"))
    t.add_argument("--sandbox", default="auto", choices=("auto", "sandbox-exec", "bwrap", "docker", "unshare", "none"))
    a = ap.parse_args()
    if a.cmd == "run":
        if not a.mock and not (a.endpoint and a.model):
            ap.error("--endpoint and --model are required unless --mock is given")
        if a.require_data is None:
            a.require_data = not a.mock or a.mock == "analyst"
        run(a)
    else:
        selftest(a)


if __name__ == "__main__":
    main()
