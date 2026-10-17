"""How many 'verified-correct' SFT trajectories are correct only under the lenient grader?

Extracts the literal each trajectory writes to /workdir/answer.txt (echo / printf forms;
programmatic writes are counted as unparsed) and grades it with v1 (dataset grader)
and checker_v2 (strict). Offline, executes nothing.

    .venv/bin/python scripts/audit_sft_answers.py
"""
from __future__ import annotations

import json
import re
import shlex
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checker_v2 import check as check_v2  # noqa: E402
from vendor.fineenvs_rollout import grade_with_method  # noqa: E402

WRITE = re.compile(r"(?:echo|printf)\b[^>]*?>\s*/workdir/answer\.txt")


def literal_answer(cmd: str) -> str | None:
    for seg in re.split(r"&&|;|\n", cmd):
        if "/workdir/answer.txt" not in seg or not WRITE.search(seg):
            continue
        left = seg.split(">")[0].strip()
        try:
            toks = shlex.split(left)
        except ValueError:
            return None
        if not toks:
            return None
        if toks[0] == "echo":
            toks = [t for t in toks[1:] if t not in ("-n", "-e", "-ne", "-en")]
            return " ".join(toks)
        if toks[0] == "printf":
            args = toks[1:]
            if len(args) >= 2 and args[0] in ("%s", "%s\\n"):
                return args[1]
            return args[0] if args else None
    return None


def main() -> None:
    train = pd.read_parquet(ROOT / "data/SmolDataEnvs/data/train-00000-of-00001.parquet")
    rows = {r["task_id"]: r for r in train.to_dict(orient="records")}
    sft = pd.read_parquet(ROOT / "data/SmolDataEnvs-sft/data/train-00000-of-00001.parquet")
    c = Counter()
    ex = []
    for rec in sft.itertuples():
        row = rows[rec.task_id]
        ans = None
        for m in map(json.loads, rec.messages):
            for tc in m.get("tool_calls") or []:
                a = tc["function"]["arguments"]
                cmd = a.get("command") if isinstance(a, dict) else json.loads(a).get("command")
                if cmd and "/workdir/answer.txt" in cmd:
                    got = literal_answer(cmd)
                    if got is not None:
                        ans = got
        if ans is None:
            c["unparsed_answer_write"] += 1
            continue
        c["parsed"] += 1
        r1, m1 = grade_with_method(row, ans.strip())
        r2, m2 = check_v2(row["answer"], ans.strip(), row["reward_mode"], row["atol"], row["rtol"])
        c[f"v1={r1:.0f},v2={r2:.0f}"] += 1
        if r1 == 1.0 and r2 == 0.0:
            c[f"v1_only_via:{m1}"] += 1
            if len(ex) < 25:
                ex.append({"task_id": rec.task_id, "mode": row["reward_mode"], "atol": row["atol"], "rtol": row["rtol"],
                           "gold": row["answer"], "submitted": ans.strip()[:80], "v1_method": m1, "v2_reason": m2})
    out = {"counts": dict(c), "examples_v1_only": ex}
    (ROOT / "reports/sft_answer_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(c), indent=1))
    for e in ex[:12]:
        print(f"  {e['mode']:11} gold={e['gold']!r:22} submitted={e['submitted']!r:34} v1:{e['v1_method']} v2:{e['v2_reason']}")


if __name__ == "__main__":
    main()
