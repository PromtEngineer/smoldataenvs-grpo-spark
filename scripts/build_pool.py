"""Pool of GRPO tasks from a pre-screen run: keep tasks the base model solves on some but
not all samples (under the reward the training optimises), since all-0 or all-1 groups
carry no GRPO gradient. Several runs (e.g. one shard per node) are merged.
    python scripts/build_pool.py runs/prescreen-4b data/grpo_pool_4b.json [--reward reward_v2]
    python scripts/build_pool.py runs/prescreen-4b-agent-node1 runs/prescreen-4b-agent-node2 \
        data/grpo_pool_4b_agent.json --min-samples 4"""
import argparse, collections, json
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+"); ap.add_argument("out"); ap.add_argument("--reward", default="reward_v2")
ap.add_argument("--min-samples", type=int, default=8)
a = ap.parse_args()
by = collections.defaultdict(dict)
for run in a.runs:
    for l in open(f"{run}/results.jsonl"):
        r = json.loads(l)
        if r.get(a.reward) is not None:
            by[r["task_id"]][(run, r["sample"])] = r[a.reward]
full = {t: list(v.values()) for t, v in by.items() if len(v) >= a.min_samples}
pool = sorted(t for t, v in full.items() if 0 < sum(v) < len(v))
never = sum(1 for v in full.values() if sum(v) == 0); always = sum(1 for v in full.values() if sum(v) == len(v))
json.dump(pool, open(a.out, "w"))
print(f"{len(full)} tasks with >= {a.min_samples} samples: {len(pool)} mixed -> pool, {never} never solved, "
      f"{always} always solved -> {a.out}")
