"""Task lists for M2 (both after milestone M1), disjoint from each other and from M1's training pool:
  data/m2_rl_candidates.json  RL pre-screen for the M1 model: the tasks the BASE never solved in M1's
                              pre-screen (the frontier) + the next unscreened tasks in the trainer's order
  data/m2_27b_tasks.json      train tasks for the 27B to solve, whose correct transcripts become SFT data
    .venv/bin/python scripts/make_m2_task_lists.py --n-rl-new 929 --n-27b 2000"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_grpo_local import build_train_rows, load_profile  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-rl-new", type=int, default=929)
ap.add_argument("--n-27b", type=int, default=2000)
a = ap.parse_args()

prof = load_profile("agent_4b_lora")
prof["task_pool_file"] = None
order = [r["task_id"] for r in build_train_rows(prof, n_tasks=100000)]
m1_candidates = json.loads((ROOT / "data/prescreen_agent_candidates.json").read_text())
assert order[:len(m1_candidates)] == m1_candidates, "trainer order changed since M1"
m1_pool = set(json.loads((ROOT / "data/grpo_pool_4b_agent.json").read_text()))

by = collections.defaultdict(list)
for n in (1, 2):
    for l in open(ROOT / f"runs/prescreen-4b-agent-node{n}/results.jsonl"):
        r = json.loads(l)
        if r.get("reward_v2") is not None:
            by[r["task_id"]].append(r["reward_v2"])
base_never = sorted(t for t, v in by.items() if len(v) >= 4 and sum(v) == 0)

start = len(m1_candidates)
rl_new = order[start:start + a.n_rl_new]
d27 = order[start + a.n_rl_new:start + a.n_rl_new + a.n_27b]
rl = base_never + rl_new
assert not (set(rl) & set(d27)) and not (set(rl) & m1_pool) and not (set(d27) & m1_pool)
(ROOT / "data/m2_rl_candidates.json").write_text(json.dumps(rl))
(ROOT / "data/m2_27b_tasks.json").write_text(json.dumps(d27))
print(f"trainer order: {len(order)} tasks; M1 screened {start}")
print(f"m2_rl_candidates: {len(rl)} = {len(base_never)} base-never + {len(rl_new)} new (order {start}-{start + len(rl_new) - 1})")
print(f"m2_27b_tasks: {len(d27)} (order {start + len(rl_new)}-{start + len(rl_new) + len(d27) - 1}); "
      f"left unused after both: {len(order) - start - len(rl_new) - len(d27)}")
