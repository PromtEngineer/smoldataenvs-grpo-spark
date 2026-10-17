import json, random, re, collections, sys
run = sys.argv[1]
R = [json.loads(l) for l in open(f"{run}/results.jsonl")]
ok = [r for r in R if r["reward_v2"] == 1]
random.Random(1).shuffle(ok)
kinds = collections.Counter()
def load(r):
    return json.load(open(f"{run}/transcripts/{r['task_id']}-s0.json"))
def cmds_of(msgs):
    out = []
    for m in msgs:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                try:
                    out.append(json.loads(tc["function"]["arguments"]).get("command", ""))
                except Exception:
                    pass
    return out
for r in ok:
    cmds = cmds_of(load(r))
    uses_py = any(k in c for c in cmds for k in ("python", "pandas", "awk", "sqlite"))
    gold = str(r["gold"]).lower()
    grep_ans = any(re.search(r"\b(grep|find)\b", c) and gold in c.lower() for c in cmds)
    kinds["computed with python/awk/sqlite" if uses_py else "no computation tool"] += 1
    if grep_ans:
        kinds["searched for the gold string itself"] += 1
    if any(re.search(r"\.\./|/home/(?!user/)|parquet|answer_key|/sandbox", c) for c in cmds):
        kinds["touched paths outside the task"] += 1
print(len(ok), "strict-correct:", dict(kinds))
for r in ok[:2]:
    msgs = load(r)
    print("\n=== TASK", r["task_id"], "gold:", r["gold"])
    print("Q:", msgs[1]["content"].split("Question:")[-1][:200].strip())
    for m in msgs[2:]:
        if m["role"] == "assistant":
            for c in cmds_of([m]):
                print("  $", c[:260].replace("\n", " | "))
        elif m["role"] == "tool":
            print("    ->", m["content"][:160].replace("\n", " | "))
