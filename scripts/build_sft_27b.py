"""Build agent-format SFT data from the 27B teacher's strict-correct transcripts (arm A2).

Reads runs/distill-27b-agent-train (results.jsonl + transcripts/<task_id>-s<sample>.json, written by
eval_harness.py agent mode: Qwen3.5-27B-FP8, thinking off, temperature 0.7, 12 model calls x 1,024 tokens) and
writes data/sft_27b/{train,val}.jsonl plus meta.json. One record per episode:

    {"task_id", "tier", ..., "messages": [system, user, assistant(tool_calls), tool, ..., assistant],
     "tools": [smolenv_local.BASH_TOOL], "no_loss_msgs": [...], "n_tokens", "n_loss_tokens", ...}

Conversions, each one matching what vLLM 0.27.1 does before it applies the chat template:
- tool_calls[].function.arguments: JSON string -> dict (vllm.entrypoints.chat_utils._postprocess_messages; the
  Qwen3.5 template iterates `arguments|items`, so a string would not render). Empty -> {}.
- assistant content, tool content, system and user messages are kept verbatim (the template trims them itself).

Filters: reward_v2 == 1 (strict checker); episodes that hit the 12-call cap (they end on a tool response, the
teacher never stopped) are dropped unless --keep-hit-max-turns. Assistant turns whose tool call has no arguments
(the teacher was cut off at 1,024 tokens inside the call; the harness answered "bash requires a string `command`")
are kept as context but listed in no_loss_msgs, so the student never learns a malformed call.

Split: 5% of tasks to val, seeded shuffle of task ids. Stats are measured with the student tokenizer and the same
render_and_mask that train_sft_agent.py trains with (it also proves every turn is prefix-consistent).

    # node 2, CPU only, inside smol-train:v2
    python3 scripts/build_sft_27b.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import smolenv_local as env  # noqa: E402
from train_sft_agent import DEFAULT_INIT, encode_rows, template_sha  # noqa: E402

META_KEYS = ("task_id", "sample", "split", "tier", "level", "reward_mode", "kaggle_dataset", "gold", "prediction",
             "n_model_calls", "n_tool_calls")


def convert(messages: list[dict]) -> tuple[list[dict], list[int], str | None]:
    """Transcript -> template messages. Returns (messages, no_loss_msgs, problem or None)."""
    if [m["role"] for m in messages[:2]] != ["system", "user"] or messages[0]["content"] != env.SFT_SYSTEM:
        return [], [], "unexpected system/user header"
    out, no_loss = [dict(messages[0]), dict(messages[1])], []
    pending: list[str] = []  # tool_call ids awaiting a tool response
    for i, m in enumerate(messages[2:], start=2):
        if m["role"] == "assistant":
            if pending:
                return [], [], f"msg {i}: assistant before all tool responses"
            a = {"role": "assistant", "content": m.get("content") or ""}
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                args = fn.get("arguments")
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                args = args if args is not None else {}
                if fn["name"] != "bash" or set(args) != {"command"} or not isinstance(args["command"], str):
                    no_loss.append(i)
                calls.append({"id": tc.get("id"), "type": "function", "function": {"name": fn["name"], "arguments": args}})
                pending.append(tc.get("id"))
            if calls:
                a["tool_calls"] = calls
            out.append(a)
        elif m["role"] == "tool":
            if not pending or m.get("tool_call_id") != pending[0]:
                return [], [], f"msg {i}: tool response out of order"
            pending.pop(0)
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "name": m.get("name", "bash"),
                        "content": m["content"]})
        else:
            return [], [], f"msg {i}: unexpected role {m['role']!r}"
    if pending:
        return [], [], "ends with unanswered tool calls"
    return out, sorted(set(no_loss)), None


def pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=str(ROOT / "runs/distill-27b-agent-train"))
    ap.add_argument("--out", default=str(ROOT / "data/sft_27b"))
    ap.add_argument("--tokenizer", default=str(DEFAULT_INIT), help="the student's tokenizer + chat template")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=12288, help="only for the 'over max length' stat")
    ap.add_argument("--keep-hit-max-turns", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    run, out = Path(a.run), Path(a.out)
    results = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines() if l.strip()]
    drop = Counter()
    rows = []
    for r in results:
        if r.get("reward_v2") != 1:
            drop["not strict-correct" if r.get("reward_v2") is not None else "ungraded"] += 1
            continue
        if r.get("hit_max_turns") and not a.keep_hit_max_turns:
            drop["hit the 12-call cap"] += 1
            continue
        tpath = run / "transcripts" / f"{r['task_id']}-s{r.get('sample', 0)}.json"
        if not tpath.exists():
            drop["no transcript"] += 1
            continue
        msgs = json.loads(tpath.read_text())
        if msgs and msgs[-1]["role"] == "tool":  # kept hit-max-turns episode: no final turn to learn
            while msgs[-1]["role"] == "tool":
                msgs.pop()
        conv, no_loss, problem = convert(msgs)
        if problem:
            drop[f"malformed transcript ({problem})"] += 1
            continue
        rec = {k: r.get(k) for k in META_KEYS}
        rec.update(messages=conv, tools=[env.BASH_TOOL], no_loss_msgs=no_loss,
                   n_assistant_turns=sum(m["role"] == "assistant" for m in conv),
                   n_tool_calls=sum(len(m.get("tool_calls") or []) for m in conv))
        rows.append(rec)

    # render + mask with the student tokenizer (asserts prefix consistency for every assistant turn)
    enc = {e["task_id"]: e for e in encode_rows(rows, a.tokenizer, a.workers)}
    for tid, e in enc.items():
        if "error" in e:
            print(f"TEMPLATE DRIFT, dropped: {tid}: {e['error']}")
            drop["template drift"] += 1
    rows = [r for r in rows if "error" not in enc[r["task_id"]]]
    turn_lens = []
    for rec in rows:
        e = enc[rec["task_id"]]
        rec["n_tokens"] = len(e["input_ids"])
        rec["n_loss_tokens"] = sum(l != -100 for l in e["labels"])
        rec["assistant_turn_tokens"] = e["turn_lengths"]
        turn_lens += e["turn_lengths"]

    tasks = sorted({r["task_id"] for r in rows})
    random.Random(a.seed).shuffle(tasks)
    val_tasks = set(tasks[: math.ceil(a.val_frac * len(tasks))])
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": [r for r in rows if r["task_id"] not in val_tasks],
              "val": [r for r in rows if r["task_id"] in val_tasks]}
    for name, rs in splits.items():
        with open(out / f"{name}.jsonl", "w") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ntok = [r["n_tokens"] for r in rows]
    nloss = [r["n_loss_tokens"] for r in rows]
    turns = [r["n_assistant_turns"] for r in rows]
    long_turns = [(r["task_id"], i, n) for r in rows for i, n in enumerate(r["assistant_turn_tokens"]) if n >= 1000]
    stats = {
        "source": str(run), "episodes": len(results), "kept": len(rows), "dropped": dict(drop),
        "train": len(splits["train"]), "val": len(splits["val"]), "val_frac": a.val_frac, "seed": a.seed,
        "tokenizer": a.tokenizer, "chat_template_sha16": template_sha(tok), "tools": [env.BASH_TOOL],
        "tokens_per_example": {"p50": pct(ntok, .5), "p90": pct(ntok, .9), "max": max(ntok), "mean": round(sum(ntok) / len(ntok))},
        "loss_tokens_per_example": {"p50": pct(nloss, .5), "p90": pct(nloss, .9), "max": max(nloss),
                                    "share_of_all_tokens": round(sum(nloss) / sum(ntok), 4)},
        "assistant_turns_per_example": {"p50": pct(turns, .5), "p90": pct(turns, .9), "max": max(turns),
                                        "mean": round(sum(turns) / len(turns), 2), "hist": dict(sorted(Counter(turns).items()))},
        "tokens_per_assistant_turn": {"p50": pct(turn_lens, .5), "p90": pct(turn_lens, .9), "max": max(turn_lens),
                                      "turns_ge_1000": len(long_turns)},
        "over_max_length": {"max_length": a.max_length, "n": sum(n > a.max_length for n in ntok),
                            "train": sum(r["n_tokens"] > a.max_length for r in splits["train"]),
                            "val": sum(r["n_tokens"] > a.max_length for r in splits["val"])},
        "no_loss_turns": {"n": sum(len(r["no_loss_msgs"]) for r in rows),
                          "episodes": [r["task_id"] for r in rows if r["no_loss_msgs"]]},
        "total_tokens": sum(ntok), "total_loss_tokens": sum(nloss),
    }
    (out / "meta.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "tools"}, indent=1))
    if long_turns:
        print("assistant turns >= 1000 tokens (teacher cap was 1,024):", long_turns[:20])
    print(f"wrote {out}/train.jsonl ({len(splits['train'])}), val.jsonl ({len(splits['val'])}), meta.json")


if __name__ == "__main__":
    main()
