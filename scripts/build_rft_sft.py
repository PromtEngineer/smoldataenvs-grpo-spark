"""Build program-format SFT data from a teacher's eval_harness run on TRAIN tasks
(rejection-sampling fine-tuning). Keeps a completion only if BOTH checkers pass it, so the
warm start never teaches "Label: 42"-style answers that only the lenient grader accepts.

    # 1. teacher run (e.g. Qwen3.8-27B on Spark node 1), program mode, 2 samples per task:
    .venv/bin/python eval_harness.py run --endpoint http://SPARK_HOST:8000/v1 --model Qwen/Qwen3.8-27B \
        --split train --limit 2000 --n-samples 2 --temperature 0.7 --mode program --no-think \
        --out runs/teacher-qwen38-27b-train
    # 2. filter:
    .venv/bin/python scripts/build_rft_sft.py runs/teacher-qwen38-27b-train
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vendor.fineenvs_rollout import build_prompt, extract_code  # noqa: E402


def main() -> None:
    run = Path(sys.argv[1])
    train = pd.read_parquet(ROOT / "data/SmolDataEnvs/data/train-00000-of-00001.parquet")
    train["files"] = train.files.map(list)
    rows = {r["task_id"]: r for r in train.to_dict(orient="records")}
    seen, kept = set(), []
    for l in (run / "results.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r.get("reward_v1") == 1.0 and r.get("reward_v2") == 1.0 and r["task_id"] not in seen and r["task_id"] in rows:
            code = extract_code(r.get("completion", ""))
            if code:
                seen.add(r["task_id"])
                kept.append({"task_id": r["task_id"], "messages": build_prompt(rows[r["task_id"]]) +
                             [{"role": "assistant", "content": f"```python\n{code}\n```"}]})
    out = ROOT / "data/sft_program/rft_teacher.jsonl"
    out.write_text("".join(json.dumps(k) + "\n" for k in kept))
    print(f"kept {len(kept)} tasks -> {out}")


if __name__ == "__main__":
    main()
