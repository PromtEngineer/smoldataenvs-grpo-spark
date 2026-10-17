"""Pre-screen candidates for M3 (RL from S2 = M1 + SFT on 27B transcripts), split into one shard per node:
the tasks M1 never solved in M2's pre-screen (S2 is stronger on hard tasks, so this is its frontier) + every
train task no earlier stage has used. Excludes M1's and M2's RL pools and the 27B transcript tasks.
    .venv/bin/python scripts/make_m3_task_lists.py"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_grpo_local import build_train_rows, load_profile  # noqa: E402

prof = load_profile("agent_4b_lora")
prof["task_pool_file"] = None
order = [r["task_id"] for r in build_train_rows(prof, n_tasks=100000)]
used = set(json.loads((ROOT / "data/prescreen_agent_candidates.json").read_text()))
used |= set(json.loads((ROOT / "data/m2_rl_candidates.json").read_text()))
d27 = set(json.loads((ROOT / "data/m2_27b_tasks.json").read_text()))
unused = [t for t in order if t not in used and t not in d27]

by = collections.defaultdict(list)
for l in open(ROOT / "runs/prescreen-m1-agent/results.jsonl"):
    r = json.loads(l)
    if r.get("reward_v2") is not None:
        by[r["task_id"]].append(r["reward_v2"])
m1_never = sorted(t for t, v in by.items() if len(v) >= 4 and sum(v) == 0)
pools = set(json.loads((ROOT / "data/grpo_pool_4b_agent.json").read_text())) | set(json.loads((ROOT / "data/grpo_pool_m2.json").read_text()))
cand = [t for t in m1_never + unused if t not in pools and t not in d27]
assert len(cand) == len(set(cand))
(ROOT / "data/m3_candidates.json").write_text(json.dumps(cand))
for i in (0, 1):
    (ROOT / f"data/m3_shard{i + 1}.json").write_text(json.dumps(cand[i::2]))
print(f"m3 candidates: {len(cand)} = {len(m1_never)} M1-never + {len(unused)} unused; shards {len(cand[0::2])} / {len(cand[1::2])}")
