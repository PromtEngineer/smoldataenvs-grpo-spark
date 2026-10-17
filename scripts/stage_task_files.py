"""Stage the input files SmolDataEnvs tasks need, into data/task_files/<bucket_prefix>/.

NOT RUN YET (needs approval: this is the one step that downloads task data).
Dry run by default: prints what would be fetched and how big it is, using the
metadata-only listing in data/bucket_listing.json. Add --yes to download.

Unlike the image's /opt/pull_bucket.py (which pulls the WHOLE bucket prefix, 37 GB for
all 526 prefixes), this fetches only the files each task lists in `files`.

    .venv/bin/python scripts/stage_task_files.py --splits eval,test          # dry run: ~4.4 GB
    .venv/bin/python scripts/stage_task_files.py --splits eval,test --yes    # download
    .venv/bin/python scripts/stage_task_files.py --splits train --yes        # ~11 GB, for GRPO
    .venv/bin/python scripts/stage_task_files.py --splits train --max-mb 100 --yes   # skip >100 MB inputs

On the DGX Spark, run it there (same script) or rsync data/task_files over ConnectX.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="eval,test")
    ap.add_argument("--dest", default=str(ROOT / "data/task_files"))
    ap.add_argument("--max-mb", type=float, default=0, help="skip tasks whose inputs exceed this (0 = no cap)")
    ap.add_argument("--yes", action="store_true", help="actually download")
    a = ap.parse_args()

    listing = json.loads((ROOT / "data/bucket_listing.json").read_text())
    size = {}
    for key, files in listing.items():
        _b, prefix = key.split("::", 1)
        for f in files or []:
            size[(prefix, f["path"].split("/")[-1])] = (f["path"], f["size"])
    want: dict[tuple[str, str], tuple[str, str, int]] = {}
    skipped = 0
    for s in a.splits.split(","):
        df = pd.read_parquet(ROOT / f"data/SmolDataEnvs/data/{s.strip()}-00000-of-00001.parquet")
        by_prefix = {}
        for (pfx, name) in size:
            by_prefix.setdefault(pfx, []).append(name)
        for r in df.itertuples():
            # 424 tasks (398 train / 11 test / 15 eval) have an EMPTY `files` list; Harbor's
            # pull_bucket.py pulls the whole prefix, so do the same for those tasks only.
            names = list(r.files) if len(r.files) else by_prefix.get(r.bucket_prefix, [])
            files = [(r.bucket_prefix, f) for f in names]
            need = sum(size.get(k, ("", 0))[1] for k in files)
            if a.max_mb and need > a.max_mb * 1e6:
                skipped += 1
                continue
            for k in files:
                if k in size:
                    want[k] = (r.hf_bucket, size[k][0], size[k][1])
    dest = Path(a.dest)
    todo = {k: v for k, v in want.items() if not (dest / k[0] / k[1]).exists()}
    tot = sum(v[2] for v in todo.values())
    print(f"{len(want)} files needed, {len(todo)} not yet staged, {tot / 1e9:.2f} GB to download"
          f"{f'; {skipped} tasks skipped by --max-mb' if skipped else ''} -> {dest}")
    if not a.yes:
        print("dry run; add --yes to download")
        return
    from huggingface_hub import download_bucket_files

    by_bucket: dict[str, list[tuple[str, str]]] = {}
    for (prefix, name), (bucket, path, _sz) in todo.items():
        (dest / prefix).mkdir(parents=True, exist_ok=True)
        by_bucket.setdefault(bucket, []).append((path, str(dest / prefix / name)))
    for bucket, files in by_bucket.items():
        for i in range(0, len(files), 50):
            download_bucket_files(bucket, files=files[i:i + 50])
            print(f"  {bucket}: {min(i + 50, len(files))}/{len(files)}", flush=True)
    print("staged.")


if __name__ == "__main__":
    main()
