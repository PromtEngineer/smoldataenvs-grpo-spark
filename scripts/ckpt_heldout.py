"""Turn a checkpoint's 394-task exam into the training run's held-out point for the dashboard.
The dashboard's base line is the untrained 4B on the 144 eval-split tasks, so `overall` here is the
eval split; the full 394-task numbers ride along under `exam_394`.
    python scripts/ckpt_heldout.py runs/T20-4b-agent-long-agent runs/grpo-agent-4b-long 20"""
import json
import sys
from pathlib import Path

exam, train_run, step = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
s = json.loads((exam / "summary.json").read_text())
out = {"step": step, "exam_run": exam.name, "overall": s["by_split"]["eval"], "exam_394": s["overall"],
       "by_tier": s["by_tier"], "by_dataset_seen_in_train": s["by_dataset_seen_in_train"]}
train_run.mkdir(parents=True, exist_ok=True)
(train_run / f"heldout_step{step:04d}.json").write_text(json.dumps(out, indent=1))
print(f"step {step}: eval-split strict {out['overall']['pass@1_v2']:.3f}, 394-task strict {s['overall']['pass@1_v2']:.3f}")
