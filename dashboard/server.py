"""Live training dashboard for the SmolDataEnvs GRPO runs (read-only).

Serves dashboard/index.html and a small JSON API over the files the runs already write:
runs/<run>/train_log.jsonl, meta.json, heldout_step*.json, completions/*.parquet, and
runs/*/summary.json for the exam leaderboard. Also reports node health (memory, GPU
processes, containers) because GB10 freezes are the main operational risk.

    .venv/bin/python dashboard/server.py --port 8950        # then open http://SPARK_HOST:8950
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
HERE = Path(__file__).resolve().parent

METRIC_KEYS = (
    "loss", "grad_norm", "learning_rate", "entropy", "reward", "reward_std", "frac_reward_zero_std",
    "completions/mean_length", "completions/max_length", "completions/clipped_ratio",
    "perf/step_seconds", "perf/tokens_per_s_end_to_end", "step_time",
    "clip_ratio/region_mean", "sampling/importance_sampling_ratio/mean", "sampling/sampling_logp_difference/mean",
    "tools/call_frequency",
)
REWARD_PREFIX = "rewards/"
_cache: dict = {}
_lock = threading.Lock()


def cached(key: str, ttl: float, fn):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (time.time(), val)
    return val


def sh(cmd: list[str], timeout: float = 5) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


# ── runs ────────────────────────────────────────────────────────────────────

def training_runs() -> list[dict]:
    out = []
    for d in RUNS.glob("grpo-*"):
        log = d / "train_log.jsonl"
        if not log.exists():
            continue
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        out.append({"name": d.name, "updated": log.stat().st_mtime, "meta": meta,
                    "finished": (d / "final").exists(), "has_heldout": bool(list(d.glob("heldout_step*.json")))})
    return sorted(out, key=lambda r: -r["updated"])


def run_detail(name: str) -> dict:
    d = RUNS / name
    rows, heldout = [], []
    for line in (d / "train_log.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if any(k.startswith("heldout/") for k in r):  # TRL can merge the eval into the step's own log line
            heldout.append({"step": r["step"], **{k[8:]: v for k, v in r.items() if k.startswith("heldout/")}})
        if "train_runtime" in r or not any(k.startswith(REWARD_PREFIX) for k in r):
            continue
        keep = {"step": r["step"], "t": r.get("t")}
        for k, v in r.items():
            if k in METRIC_KEYS or (k.startswith(REWARD_PREFIX) and k.endswith("/mean")):
                try:
                    keep[k] = float(v)
                except (TypeError, ValueError):
                    pass
        rows.append(keep)
    for f in sorted(d.glob("heldout_step*.json")):  # full held-out summaries (tiers, seen/unseen)
        step = int(re.search(r"(\d+)", f.stem).group(1))
        s = json.loads(f.read_text())
        if not any(h["step"] == step for h in heldout):
            heldout.append({"step": step, "pass@1_v1": s["overall"]["pass@1_v1"], "pass@1_v2": s["overall"]["pass@1_v2"]})
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    if not meta.get("max_steps") and "pilot" in name:
        meta["max_steps"] = 40
    return {"name": name, "meta": meta, "rows": rows, "heldout": sorted(heldout, key=lambda h: h["step"]),
            "finished": (d / "final").exists(), "samples": latest_samples(d)}


def _question(prompt: str) -> str:
    m = re.search(r"Question:\s*(.+?)(?:\n\n|$)", prompt, re.S) or re.search(r"\nuser\n(.+?)(?:\n\n|$)", prompt, re.S)
    return (m.group(1) if m else prompt[-300:]).strip()[:400]


def latest_samples(d: Path) -> dict:
    files = sorted(d.glob("completions/completions_*.parquet"))
    if not files:
        return {}
    try:
        import pandas as pd

        df = pd.read_parquet(files[-1])
    except Exception as e:
        return {"error": str(e)}
    groups = []
    for prompt, g in df.groupby("prompt", sort=False):
        recs = []
        for _, r in g.head(8).iterrows():
            recs.append({"completion": str(r["completion"])[-1800:],
                         "strict": _f(r.get("reward_correct_strict", r.get("SmolDataEnv"))),
                         "lenient": _f(r.get("reward_correct", r.get("reward_lenient"))),
                         "advantage": _f(r.get("advantage"))})
        groups.append({"question": _question(str(prompt)), "samples": recs})
    return {"step": int(df["step"].iloc[0]) if "step" in df else None, "groups": groups[:4]}


def _f(v):
    try:
        return None if v is None else round(float(v), 3)
    except (TypeError, ValueError):
        return None


# ── exams (baselines and trained checkpoints) ───────────────────────────────

LABELS = {
    "A0-qwen35-2b": "Qwen3.5-2B (base)", "A0-grpo2b-v0": "FineEnvs v0 (trained 2B)", "A0-qwen35-4b": "Qwen3.5-4B (base)",
    "A0-qwen35-27b-fp8": "Qwen3.5-27B-FP8", "REF-gemini38flash-low": "Gemini 3.8 Flash (low thinking)",
    "P40-4b-grpo-pilot": "4B + GRPO pilot (40 steps)", "T200-4b-grpo": "4B + GRPO (200 steps)",
}


def exams() -> list[dict]:
    out = []
    for f in RUNS.glob("*/summary.json"):
        name = f.parent.name
        m = re.match(r"(A0|REF|P\d+|T\d+|S\d+)-(.+)-(program|agent)$", name)  # S<epochs> = SFT
        if not m:
            continue
        s = json.loads(f.read_text())
        base = f"{m.group(1)}-{m.group(2)}"
        unseen = (s.get("by_dataset_seen_in_train") or {}).get("False", {})
        n, ung = s["overall"]["n_tasks"], s.get("n_ungraded", 0)
        out.append({"run": name, "label": LABELS.get(base, base), "mode": m.group(3), "n": n,
                    "ungraded": ung, "partial": ung > 0.05 * (n + ung), "strict": s["overall"]["pass@1_v2"],
                    "lenient": s["overall"]["pass@1_v1"], "unseen_strict": unseen.get("pass@1_v2"),
                    "trained": bool(re.match(r"[PTS]\d+", m.group(1)))})
    return sorted(out, key=lambda r: (r["mode"], r["partial"], -r["strict"]))


def base_eval_split() -> dict:
    """Untrained 4B on the 144 eval-split tasks: the reference line for in-loop held-out evals."""
    res = {}
    for mode in ("program", "agent"):
        p = RUNS / f"A0-qwen35-4b-{mode}/results.jsonl"
        if p.exists():
            rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            ev = [r for r in rs if r.get("split") == "eval" and r.get("reward_v2") is not None]
            if ev:
                res[mode] = {"strict": sum(r["reward_v2"] for r in ev) / len(ev),
                             "lenient": sum(r["reward_v1"] or 0 for r in ev) / len(ev), "n": len(ev)}
    return res


# ── node health ─────────────────────────────────────────────────────────────

def system() -> dict:
    mi = {l.split(":")[0]: int(l.split()[1]) / 1048576 for l in open("/proc/meminfo") if l.split()[1].isdigit()}
    gpu = []
    for l in sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
        pid, mem = [x.strip() for x in l.split(",")[:2]]
        gpu.append({"pid": pid, "mib": int(mem) if mem.isdigit() else None})
    containers = []
    for l in sh(["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Label \"smolenv.sandbox\"}}"]).splitlines():
        n, st, sbx = (l.split("\t") + ["", ""])[:3]
        if not sbx:
            containers.append({"name": n, "status": st})
    n_sandboxes = len(sh(["docker", "ps", "-q", "--filter", "label=smolenv.sandbox=1"]).split())
    tmux = [l.split(":")[0] for l in sh(["tmux", "ls"]).splitlines() if ":" in l]
    peer = os.environ.get("PEER_NODE")   # optional: the second Spark's address, shown as reachable/down
    node1 = subprocess.run(["ping", "-c", "1", "-W", "1", peer], capture_output=True).returncode == 0 if peer else None
    stage = []
    ov = ROOT / "logs/overnight.log"
    if ov.exists():
        stage = [l.strip() for l in ov.read_text().splitlines() if "STAGE" in l or "FAILED" in l or "DONE" in l][-8:]
    return {"mem": {"total": mi.get("MemTotal"), "free": mi.get("MemFree"), "available": mi.get("MemAvailable"),
                    "cached": mi.get("Cached")},
            "gpu": gpu, "containers": containers, "sandboxes": n_sandboxes, "tmux": tmux, "node1_up": node1,
            "load": os.getloadavg()[0], "pipeline": stage, "now": time.time()}


# ── http ────────────────────────────────────────────────────────────────────

class H(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                body = (HERE / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/runs":
                self._json(cached("runs", 5, training_runs))
            elif u.path == "/api/run":
                name = q.get("name", [""])[0]
                if not re.fullmatch(r"grpo-[\w.-]+", name) or not (RUNS / name / "train_log.jsonl").exists():
                    return self._json({"error": "unknown run"}, 404)
                self._json(cached(f"run:{name}", 5, lambda: run_detail(name)))
            elif u.path == "/api/exams":
                self._json(cached("exams", 30, lambda: {"exams": exams(), "base_eval": base_eval_split()}))
            elif u.path == "/api/system":
                self._json(cached("system", 5, system))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # keep the dashboard up while files are mid-write
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8950)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    print(f"dashboard on http://{a.host}:{a.port}  (runs: {RUNS})", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
