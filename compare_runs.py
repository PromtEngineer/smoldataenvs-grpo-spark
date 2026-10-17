"""Paired comparison of two eval runs on the same tasks (e.g. base vs GRPO, or 4B vs 27B).

Same tasks, same harness, so the right test is paired: McNemar's exact test on the
discordant tasks, plus a paired bootstrap CI for the accuracy difference. Reports both
checkers and the unseen-dataset (transfer) slice separately.

    .venv/bin/python compare_runs.py runs/base-4b-test runs/grpo-4b-test
"""
from __future__ import annotations

import json
import random
import sys
from math import comb
from pathlib import Path


def load(run: Path) -> dict[str, dict]:
    recs = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines() if l.strip()]
    by = {}
    for r in recs:
        if r.get("reward_v1") is None:
            continue
        b = by.setdefault(r["task_id"], {"v1": [], "v2": [], "seen": r.get("seen_dataset_in_train"), "tier": r.get("tier")})
        b["v1"].append(r["reward_v1"])
        b["v2"].append(r["reward_v2"])
    return {k: {**v, "v1": sum(v["v1"]) / len(v["v1"]), "v2": sum(v["v2"]) / len(v["v2"])} for k, v in by.items()}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(a: dict, b: dict, key: str, ids: list[str], boot: int = 5000) -> dict:
    xa = [a[i][key] for i in ids]
    xb = [b[i][key] for i in ids]
    diff = sum(xb) / len(ids) - sum(xa) / len(ids)
    rng = random.Random(0)
    ds = []
    for _ in range(boot):
        s = [rng.randrange(len(ids)) for _ in ids]
        ds.append(sum(xb[j] - xa[j] for j in s) / len(ids))
    ds.sort()
    only_a = sum(1 for p, q in zip(xa, xb) if p >= 0.5 > q)
    only_b = sum(1 for p, q in zip(xa, xb) if q >= 0.5 > p)
    return {"n": len(ids), "acc_A": round(sum(xa) / len(ids), 4), "acc_B": round(sum(xb) / len(ids), 4),
            "B_minus_A": round(diff, 4), "ci95": (round(ds[int(0.025 * boot)], 4), round(ds[int(0.975 * boot)], 4)),
            "solved_only_by_A": only_a, "solved_only_by_B": only_b, "mcnemar_p": round(mcnemar_exact(only_a, only_b), 4)}


def main() -> None:
    ra, rb = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = load(ra), load(rb)
    common = sorted(set(a) & set(b))
    out = {"A": str(ra), "B": str(rb), "common_tasks": len(common)}
    for key in ("v1", "v2"):
        out[f"all_{key}"] = compare(a, b, key, common)
        unseen = [i for i in common if not a[i]["seen"]]
        if unseen:
            out[f"unseen_dataset_{key}"] = compare(a, b, key, unseen)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
