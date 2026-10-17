"""Profile the SmolDataEnvs splits, fully offline.

Reads the three parquet splits, the SFT parquet, the harbor task.toml samples,
and the metadata-only bucket listing (scripts/list_bucket_sizes.py). Writes:

    reports/dataset_profile.json   machine-readable numbers
    reports/dataset_profile.txt    the same, readable
    data/slices.json               task-id lists for the eval slices used in the plan

    .venv/bin/python scripts/profile_dataset.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vendor.fineenvs_rollout import looks_like_a_command  # noqa: E402

SPLITS = ("train", "test", "eval")


def load() -> pd.DataFrame:
    frames = []
    for s in SPLITS:
        df = pd.read_parquet(ROOT / f"data/SmolDataEnvs/data/{s}-00000-of-00001.parquet")
        df["split"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def ext(name: str) -> str:
    n = name.lower()
    for e in (".csv.gz", ".csv.zip", ".tar.gz"):
        if n.endswith(e):
            return e
    return Path(n).suffix or "(none)"


def pct(a) -> dict:
    a = np.asarray(list(a), dtype=float)
    if not len(a):
        return {}
    return {"p50": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "mean": float(a.mean())}


CLEAN_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def main() -> None:
    df = load()
    out: dict = {}
    txt: list[str] = []

    def section(title):
        txt.append("")
        txt.append("=" * 78)
        txt.append(title)
        txt.append("=" * 78)

    # ── counts ────────────────────────────────────────────────────────────────
    section("Task counts")
    tiers = pd.crosstab(df.split, df.difficulty_tier).reindex(SPLITS)[["easy", "medium", "hard"]]
    levels = pd.crosstab(df.split, df.difficulty_level).reindex(SPLITS)
    modes = pd.crosstab(df.reward_mode, df.split)[list(SPLITS)]
    out["n_tasks"] = df.split.value_counts().reindex(SPLITS).to_dict()
    out["tiers"] = tiers.to_dict(orient="index")
    out["levels"] = levels.to_dict(orient="index")
    out["reward_modes"] = modes.to_dict(orient="index")
    out["n_kaggle_datasets"] = int(df.kaggle_dataset.nunique())
    out["n_bucket_prefixes"] = int(df.bucket_prefix.nunique())
    txt += [f"total {len(df)} tasks; {out['n_kaggle_datasets']} Kaggle datasets; "
            f"{out['n_bucket_prefixes']} bucket prefixes; bucket(s): {df.hf_bucket.unique().tolist()}",
            "", "by tier:", tiers.to_string(), "", "by difficulty_level (1-5):", levels.to_string(),
            "", "by reward_mode (answer type):", modes.to_string()]

    # ── tolerances ────────────────────────────────────────────────────────────
    section("Numeric tolerances (atol / rtol) per reward_mode, all splits")
    tol = {}
    for m, g in df.groupby("reward_mode"):
        pairs = Counter(zip(g.atol, g.rtol))
        tol[m] = {f"atol={a:g},rtol={r:g}": int(c) for (a, r), c in pairs.most_common()}
        txt.append(f"{m:12} {tol[m]}")
    out["tolerances"] = tol

    # ── answer shapes ─────────────────────────────────────────────────────────
    section("Gold answer shapes")
    ans = df.answer.astype(str)
    shapes = {
        "len_chars": pct(ans.str.len()),
        "numeric_mode_gold_not_clean_number": int(((df.reward_mode == "numeric") & ~ans.str.strip().str.fullmatch(CLEAN_NUM.pattern)).sum()),
        "gold_with_percent_sign": int(ans.str.contains("%").sum()),
        "gold_with_comma": int(ans.str.contains(",").sum()),
        "gold_multiline": int(ans.str.contains("\n").sum()),
        "gold_trips_rollout_command_filter": int(ans.map(looks_like_a_command).sum()),
        "gold_is_not_applicable": int((ans.str.strip().str.lower() == "not applicable").sum()),
    }
    out["answer_shapes"] = shapes
    for k, v in shapes.items():
        txt.append(f"{k:40} {v}")
    bad_cmd = df[ans.map(looks_like_a_command)]
    out["gold_trips_command_filter_examples"] = bad_cmd[["split", "task_id", "reward_mode", "answer"]].head(15).to_dict(orient="records")
    txt.append("golds that rollout.py's looks_like_a_command() would reject (scored 0 even when printed exactly):")
    for r in out["gold_trips_command_filter_examples"]:
        txt.append(f"   {r['split']:5} {r['task_id']:28} {r['reward_mode']:12} {r['answer']!r}")
    txt.append(f"   by split: {bad_cmd.split.value_counts().to_dict()}")
    out["gold_trips_command_filter_by_split"] = bad_cmd.split.value_counts().to_dict()
    nn = df[(df.reward_mode == "numeric") & ~ans.str.strip().str.fullmatch(CLEAN_NUM.pattern)]
    txt.append("numeric-mode golds that are not a bare number (sample):")
    for r in nn.head(10).itertuples():
        txt.append(f"   {r.split:5} {r.task_id:28} {r.answer!r}")

    # ── files and sizes ───────────────────────────────────────────────────────
    section("Input files (from the metadata-only bucket listing)")
    listing_path = ROOT / "data/bucket_listing.json"
    listing = json.loads(listing_path.read_text()) if listing_path.exists() else {}
    size_of: dict[tuple[str, str], int] = {}
    prefix_total: dict[str, int] = {}
    prefix_nfiles: dict[str, int] = {}
    for key, files in listing.items():
        _bucket, prefix = key.split("::", 1)
        files = files or []
        prefix_total[prefix] = sum(f["size"] for f in files)
        prefix_nfiles[prefix] = len(files)
        for f in files:
            size_of[(prefix, f["path"].split("/")[-1])] = f["size"]
    df["n_files"] = df.files.map(len)
    out["empty_files_list_tasks"] = df[df.n_files == 0].split.value_counts().reindex(SPLITS).fillna(0).astype(int).to_dict()
    exts = Counter(ext(f) for fl in df.files for f in fl)
    need_bytes, missing = [], []
    for r in df.itertuples():
        b = 0
        for f in r.files:
            s = size_of.get((r.bucket_prefix, f))
            if s is None:
                missing.append((r.split, r.task_id, r.bucket_prefix, f))
            else:
                b += s
        need_bytes.append(b)
    df["need_bytes"] = need_bytes
    df["prefix_bytes"] = df.bucket_prefix.map(prefix_total).fillna(0)
    files_out = {"extensions_in_task_file_lists": dict(exts.most_common()),
                 "files_per_task": pct(df.n_files),
                 "listed_files_missing_from_bucket": len(missing),
                 "missing_examples": missing[:10],
                 "bucket_total_GB_for_all_prefixes": round(sum(prefix_total.values()) / 1e9, 2),
                 "bucket_total_files_for_all_prefixes": int(sum(prefix_nfiles.values())),
                 "per_split": {}}
    for s in SPLITS:
        d = df[df.split == s]
        uniq = {}
        for r in d.itertuples():
            for f in r.files:
                if (r.bucket_prefix, f) in size_of:
                    uniq[(r.bucket_prefix, f)] = size_of[(r.bucket_prefix, f)]
        pref = d.bucket_prefix.unique()
        files_out["per_split"][s] = {
            "needed_MB_per_task": {k: round(v / 1e6, 2) for k, v in pct(d.need_bytes).items()},
            "unique_needed_files": len(uniq),
            "unique_needed_GB": round(sum(uniq.values()) / 1e9, 3),
            "whole_prefix_GB_pulled_by_pull_bucket_py": round(sum(prefix_total.get(p, 0) for p in pref) / 1e9, 3),
            "n_prefixes": int(len(pref)),
            "tasks_needing_over_100MB": int((d.need_bytes > 100e6).sum()),
            "tasks_needing_over_1GB": int((d.need_bytes > 1e9).sum()),
        }
    out["files"] = files_out
    txt.append(f"file types named in task file lists: {dict(exts.most_common())}")
    txt.append(f"files per task: {files_out['files_per_task']}")
    txt.append(f"task-listed files missing from bucket listing: {len(missing)}  e.g. {missing[:3]}")
    txt.append(f"bucket, all {len(prefix_total)} prefixes: {files_out['bucket_total_GB_for_all_prefixes']} GB in "
               f"{files_out['bucket_total_files_for_all_prefixes']} files (what pull_bucket.py would pull in total)")
    for s, v in files_out["per_split"].items():
        txt.append(f"{s:5} {v}")
    big = df.sort_values("need_bytes", ascending=False).head(5)
    txt.append("largest per-task inputs:")
    for r in big.itertuples():
        txt.append(f"   {r.split:5} {r.task_id:28} {r.need_bytes/1e6:8.1f} MB  {list(r.files)[:3]}")
    out["largest_inputs"] = [{"split": r.split, "task_id": r.task_id, "MB": round(r.need_bytes / 1e6, 1), "files": list(r.files)} for r in big.itertuples()]

    # ── package tier ──────────────────────────────────────────────────────────
    out["package_tier"] = pd.crosstab(df.package_tier, df.split)[list(SPLITS)].to_dict(orient="index")
    section("package_tier (environment sizing hint) by split")
    txt.append(pd.crosstab(df.package_tier, df.split)[list(SPLITS)].to_string())

    # ── overlap / transfer ────────────────────────────────────────────────────
    section("Train vs held-out overlap (what 'never trained on' really means)")
    tr = df[df.split == "train"]
    train_ds = set(tr.kaggle_dataset)
    train_nb = set(tr.source_row_id.map(lambda x: x.split("_qa_")[0]))
    train_q = set(tr.question.str.strip().str.lower())
    ov = {}
    slices = {}
    for s in ("test", "eval"):
        d = df[df.split == s]
        seen = d.kaggle_dataset.isin(train_ds)
        ov[s] = {
            "tasks": int(len(d)),
            "kaggle_datasets": int(d.kaggle_dataset.nunique()),
            "kaggle_datasets_also_in_train": int(len(set(d.kaggle_dataset) & train_ds)),
            "tasks_on_train_seen_dataset": int(seen.sum()),
            "tasks_on_unseen_dataset": int((~seen).sum()),
            "tasks_from_notebook_also_in_train": int(d.source_row_id.map(lambda x: x.split("_qa_")[0]).isin(train_nb).sum()),
            "identical_question_text_in_train": int(d.question.str.strip().str.lower().isin(train_q).sum()),
            "unseen_dataset_tiers": d[~seen].difficulty_tier.value_counts().to_dict(),
        }
        slices[f"{s}_all"] = d.task_id.tolist()
        slices[f"{s}_unseen_dataset"] = d[~seen].task_id.tolist()
        slices[f"{s}_seen_dataset"] = d[seen].task_id.tolist()
        txt.append(f"{s}: {ov[s]}")
    slices["transfer_unseen_dataset"] = slices["test_unseen_dataset"] + slices["eval_unseen_dataset"]
    slices["eval_plus_test"] = slices["eval_all"] + slices["test_all"]
    # the held-out slice checker_v2 is designed for: the whole test split (never used during training)
    slices["heldout_v2"] = slices["test_all"]
    rng = np.random.default_rng(0)
    slices["eval_pilot_48"] = sorted(rng.choice(slices["eval_all"], 48, replace=False).tolist())
    dup = df[(df.split == "test") & df.question.str.strip().str.lower().isin(train_q)]
    ov["identical_question_examples"] = []
    for r in dup.itertuples():
        m = tr[tr.question.str.strip().str.lower() == r.question.strip().lower()].iloc[0]
        ov["identical_question_examples"].append({"test_task": r.task_id, "question": r.question, "test_gold": r.answer,
                                                  "train_task": m.task_id, "train_gold": m.answer,
                                                  "same_dataset": r.kaggle_dataset == m.kaggle_dataset})
    for e in ov["identical_question_examples"]:
        txt.append(f"   dup: {e['question'][:70]!r} test={e['test_gold']!r} train={e['train_gold']!r} same_ds={e['same_dataset']}")
    out["overlap"] = ov
    (ROOT / "data/slices.json").write_text(json.dumps(slices, indent=1))
    txt.append(f"slices written to data/slices.json: { {k: len(v) for k, v in slices.items()} }")

    # ── SFT set ───────────────────────────────────────────────────────────────
    section("SmolDataEnvs-sft trajectories")
    sft = pd.read_parquet(ROOT / "data/SmolDataEnvs-sft/data/train-00000-of-00001.parquet")
    chars = sft.messages.map(lambda ms: sum(len(m) for m in ms))
    sft_out = {"rows": int(len(sft)), "tiers": sft.difficulty_tier.value_counts().to_dict(),
               "source_agent": sft.source_agent.value_counts().to_dict(), "n_turns": pct(sft.n_turns),
               "json_chars_per_trajectory": pct(chars),
               "approx_tokens_per_trajectory(chars/3.5)": {k: round(v / 3.5) for k, v in pct(chars).items()},
               "task_ids_in_train": int(sft.task_id.isin(set(tr.task_id)).sum()),
               "task_ids_in_test_or_eval": int(sft.task_id.isin(set(df[df.split != "train"].task_id)).sum())}
    out["sft"] = sft_out
    for k, v in sft_out.items():
        txt.append(f"{k:40} {v}")

    # ── sandbox requirements ──────────────────────────────────────────────────
    section("Sandbox requirements (harbor task.toml + scripts/rollout.py)")
    sandbox = {
        "image": "docker.io/savatar101/env-data-agent-train:base (Dockerfile: FROM that image, WORKDIR /workdir)",
        "harbor_task_toml": {"cpus": 1, "memory_mb": 1024, "storage_mb": 5120, "gpus": 0,
                             "allow_internet": True, "build_timeout_sec": 600, "agent_timeout_sec": 600,
                             "verifier_timeout_sec": 120,
                             "data_staging": "healthcheck runs /opt/pull_bucket.py: pulls the WHOLE bucket prefix into /home/user/input (needs HF_TOKEN)",
                             "answer_channel": "/workdir/answer.txt, graded by tests/test.sh -> tests/grader.py (--json: correctness, submission, tool_efficiency with TOOL_BUDGET=15)"},
        "rollout_py": {"sandbox": "huggingface_hub.Sandbox, flavor cpu-basic, idle_timeout 30m, one VM per process, reused across rollouts",
                       "per_program_timeout_sec": 90, "per_command_timeout_sec": 120,
                       "answer_channel": "last non-empty stdout line",
                       "network": "not blocked (the same VM pulls data from the Hub)"},
        "installed_per_instruction": "pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed)",
    }
    out["sandbox"] = sandbox
    txt.append(json.dumps(sandbox, indent=1))

    # ── examples ──────────────────────────────────────────────────────────────
    section("Five example tasks with gold answers")
    ex_ids = []
    ev = df[df.split.isin(["eval", "test"])]
    for mode, tier in (("numeric", "easy"), ("exact_short", "medium"), ("exact_bool", "medium"), ("numeric", "hard"), ("list_csv", None)):
        g = ev[(ev.reward_mode == mode) & ((ev.difficulty_tier == tier) if tier else True)]
        if len(g):
            ex_ids.append(g.sample(1, random_state=7).task_id.iloc[0])
    examples = []
    for tid in ex_ids:
        r = df[df.task_id == tid].iloc[0]
        e = {"split": r.split, "task_id": tid, "tier": r.difficulty_tier, "level": int(r.difficulty_level),
             "kaggle": r.kaggle_dataset, "files": list(r.files), "need_MB": round(r.need_bytes / 1e6, 2),
             "question": r.question, "gold": r.answer, "reward_mode": r.reward_mode, "atol": r.atol, "rtol": r.rtol}
        examples.append(e)
        txt.append(json.dumps(e, ensure_ascii=False))
    out["examples"] = examples

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/dataset_profile.json").write_text(json.dumps(out, indent=1, default=str))
    (ROOT / "reports/dataset_profile.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
