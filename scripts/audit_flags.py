import json, re, sys
run = sys.argv[1]
R = [json.loads(l) for l in open(f"{run}/results.jsonl")]
for r in [x for x in R if x["reward_v2"] == 1]:
    msgs = json.load(open(f"{run}/transcripts/{r['task_id']}-s0.json"))
    gold = str(r["gold"]).lower()
    for m in msgs:
        for tc in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
            c = json.loads(tc["function"]["arguments"]).get("command", "")
            if re.search(r"\.\./|/home/(?!user/)|parquet|answer_key|/sandbox", c):
                print("OUTSIDE", r["task_id"], "gold", repr(gold), "|", c[:220].replace("\n", " | "))
            elif re.search(r"\b(grep|find)\b", c) and gold in c.lower():
                print("GREP   ", r["task_id"], "gold", repr(gold), "|", c[:160].replace("\n", " | "))
