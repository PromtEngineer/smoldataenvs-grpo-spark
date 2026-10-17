"""Candidate train tasks for an agent-mode pre-screen, split into one shard per node.
Same candidates the trainer would draw (profile filters, seed-42 shuffle), interleaved so
each shard gets the same mix of tiers.
    .venv/bin/python scripts/make_prescreen_shards.py --profile agent_4b_lora --n 1000 --shards 2"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_grpo_local import build_train_rows, load_profile  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--profile", default="agent_4b_lora")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--shards", type=int, default=2)
ap.add_argument("--out", default="data/prescreen_agent")
a = ap.parse_args()

prof = load_profile(a.profile)
prof["task_pool_file"] = None  # candidates come from the whole train split, not an earlier pool
rows = build_train_rows(prof, n_tasks=a.n)
ids = [r["task_id"] for r in rows]
Path(ROOT / f"{a.out}_candidates.json").write_text(json.dumps(ids))
for i in range(a.shards):
    shard = ids[i::a.shards]
    Path(ROOT / f"{a.out}_shard{i + 1}.json").write_text(json.dumps(shard))
    tiers = collections.Counter(r["difficulty_tier"] for r in rows[i::a.shards])
    print(f"shard {i + 1}: {len(shard)} tasks {dict(tiers)} -> {a.out}_shard{i + 1}.json")
