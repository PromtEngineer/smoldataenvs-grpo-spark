"""Turn SmolDataEnvs-sft agent trajectories into single-program SFT examples.

Why: FineEnvs' SFT script trains on multi-turn `bash` trajectories that end with
`echo -n ... > /workdir/answer.txt`, but its GRPO and eval scripts use a different,
single-turn protocol: one ```python block, the last printed line is the answer.
Warm-starting on one protocol and then doing RL/eval on the other wastes the warm start.

This script keeps a trajectory only when one of its Python calls is, by itself, a
complete program whose recorded output ends with a line the dataset grader accepts as
the gold answer. That program becomes the target completion for the program-mode
prompt (vendor/fineenvs_rollout.build_prompt), i.e. exactly what GRPO will sample.

Offline: it uses the tool outputs recorded in the trajectories and never executes code.
`--verify` re-executes the kept programs in the local sandbox and re-grades them; that
needs the train task files staged under data/task_files (scripts/stage_task_files.py).

    .venv/bin/python scripts/convert_sft_to_program.py            # writes data/sft_program/
    .venv/bin/python scripts/convert_sft_to_program.py --verify   # after staging train files
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checker_v2 import check as check_v2  # noqa: E402
from vendor.fineenvs_rollout import build_prompt, grade_with_method, last_line  # noqa: E402

HEREDOC = re.compile(r"python3?\s+(?:-\s*)?<<\s*-?\s*(['\"]?)(\w+)\1\s*\n(.*?)\n\2\s*(?:$|\n)", re.S)
CAT_TO_FILE = re.compile(r"cat\s*>\s*(\S+\.py)\s*<<\s*-?\s*(['\"]?)(\w+)\2\s*\n(.*?)\n\3\s*(?:$|\n)", re.S)
PY_C = re.compile(r"""python3?\s+-c\s+(['"])(.*)\1\s*(?:2>&1)?\s*$""", re.S)


def python_programs(command: str) -> list[str]:
    """Python source(s) a bash command runs: heredocs, `cat > x.py <<EOF` + `python x.py`, `python -c`."""
    progs = [m.group(3) for m in HEREDOC.finditer(command)]
    for m in CAT_TO_FILE.finditer(command):
        if re.search(rf"python3?\s+{re.escape(m.group(1))}", command):
            progs.append(m.group(4))
    if not progs:
        m = PY_C.search(command.strip())
        if m and m.group(1) == '"' and "\\\"" not in m.group(2):
            progs.append(m.group(2))
        elif m and m.group(1) == "'":
            progs.append(m.group(2))
    return progs


def convert(max_lines: int) -> dict:
    train = pd.read_parquet(ROOT / "data/SmolDataEnvs/data/train-00000-of-00001.parquet")
    train["files"] = train.files.map(list)
    rows = {r["task_id"]: r for r in train.to_dict(orient="records")}
    sft = pd.read_parquet(ROOT / "data/SmolDataEnvs-sft/data/train-00000-of-00001.parquet")
    stats = Counter()
    kept = []
    for rec in sft.itertuples():
        stats["trajectories"] += 1
        row = rows[rec.task_id]
        msgs = [json.loads(m) for m in rec.messages]
        outputs = {m.get("tool_call_id"): (m.get("content") or "") for m in msgs if m["role"] == "tool"}
        cands = []
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                args = tc["function"]["arguments"]
                cmd = args.get("command") if isinstance(args, dict) else json.loads(args).get("command")
                for prog in python_programs(cmd or ""):
                    cands.append((prog, outputs.get(tc["id"], "")))
        if not cands:
            stats["no_python_call"] += 1
            continue
        stats["has_python_call"] += 1
        chosen = None
        for prog, out in reversed(cands):  # latest program first: the one the agent trusted
            pred = last_line(out)
            r1, _ = grade_with_method(row, pred)
            if r1 == 1.0:
                chosen = (prog, pred)
                break
        if not chosen:
            stats["no_program_prints_gold_last"] += 1
            continue
        prog, pred = chosen
        n_lines = sum(1 for ln in prog.splitlines() if ln.strip())
        r2, _ = check_v2(row["answer"], pred, row["reward_mode"], row["atol"], row["rtol"])
        stats["gold_last_line"] += 1
        stats["gold_last_line_and_v2"] += int(r2 == 1.0)
        if n_lines > max_lines:
            stats[f"dropped_over_{max_lines}_lines"] += 1
            continue
        try:
            compile(prog, "<sft>", "exec")
        except SyntaxError:
            stats["dropped_syntax_error"] += 1
            continue
        stats["kept"] += 1
        kept.append({"task_id": rec.task_id, "difficulty_tier": row["difficulty_tier"], "n_lines": n_lines,
                     "v2_agrees": r2 == 1.0, "recorded_last_line": pred, "gold": row["answer"],
                     "messages": build_prompt(row) + [{"role": "assistant", "content": f"```python\n{prog.strip()}\n```"}]})
    out_dir = ROOT / "data/sft_program"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w") as fh, open(out_dir / "train_strict.jsonl", "w") as fs:
        for k in kept:
            fh.write(json.dumps(k) + "\n")
            if k["v2_agrees"]:  # last printed line is the bare answer, not "Label: 42"
                fs.write(json.dumps(k) + "\n")
    tiers = Counter(k["difficulty_tier"] for k in kept)
    lines = Counter(min(k["n_lines"] // 10 * 10, 60) for k in kept)
    report = {"stats": dict(stats), "kept_by_tier": dict(tiers), "kept_n_lines_hist(bucket=10)": dict(sorted(lines.items())),
              "kept_v2_agrees": sum(k["v2_agrees"] for k in kept), "out": str(out_dir / "train.jsonl"),
              "out_strict": str(out_dir / "train_strict.jsonl")}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/sft_program_conversion.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return report


def verify(limit: int) -> None:
    """Re-execute kept programs in the local sandbox and re-grade (needs staged train files)."""
    import smolenv_local as env

    train = pd.read_parquet(ROOT / "data/SmolDataEnvs/data/train-00000-of-00001.parquet")
    train["files"] = train.files.map(list)
    rows = {r["task_id"]: r for r in train.to_dict(orient="records")}
    recs = [json.loads(l) for l in (ROOT / "data/sft_program/train.jsonl").read_text().splitlines()][: limit or None]
    ok = Counter()
    cfg = env.SandboxConfig()
    for k in recs:
        row = rows[k["task_id"]]
        sb = env.LocalSandbox(row, ROOT / "data/task_files", ROOT / "runs/sft_program_verify/sandboxes", cfg)
        if sb.missing_files:
            ok["missing_data"] += 1
            continue
        res = env.program_rollout(sb, row, k["messages"][-1]["content"])
        ok["v1_pass" if res["reward_v1"] == 1.0 else "v1_fail"] += 1
    print(dict(ok))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lines", type=int, default=60, help="the program prompt says 'under 40 lines'; allow some slack")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.verify:
        verify(a.limit)
    else:
        convert(a.max_lines)
