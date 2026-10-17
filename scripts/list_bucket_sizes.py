"""Metadata-only listing of the task data files in the SmolDataEnvs bucket.

Calls list_bucket_tree (a directory listing, no file content) once per unique
bucket_prefix across all three splits and caches the result. Nothing is
downloaded. Resumable: prefixes already in the cache are skipped.

    .venv/bin/python scripts/list_bucket_sizes.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import list_bucket_tree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "bucket_listing.json"


def main() -> None:
    frames = [pd.read_parquet(ROOT / f"data/SmolDataEnvs/data/{s}-00000-of-00001.parquet") for s in ("train", "test", "eval")]
    df = pd.concat(frames)
    pairs = sorted(set(zip(df["hf_bucket"], df["bucket_prefix"])))
    cache = json.loads(OUT.read_text()) if OUT.exists() else {}
    print(f"{len(pairs)} unique (bucket, prefix) pairs; {len(cache)} cached")
    for n, (bucket, prefix) in enumerate(pairs):
        key = f"{bucket}::{prefix}"
        if key in cache:
            continue
        for attempt in range(5):
            try:
                items = list(list_bucket_tree(bucket, prefix=prefix.rstrip("/") + "/", recursive=True))
                cache[key] = [{"path": i.path, "size": i.size} for i in items if getattr(i, "type", None) == "file"]
                break
            except Exception as exc:  # rate limit or transient
                wait = 10 * (attempt + 1)
                print(f"  {prefix}: {type(exc).__name__} {str(exc)[:120]} -> retry in {wait}s")
                time.sleep(wait)
        else:
            cache[key] = None
        if n % 25 == 0:
            OUT.write_text(json.dumps(cache))
            print(f"  {n}/{len(pairs)}")
        time.sleep(0.25)
    OUT.write_text(json.dumps(cache))
    missing = [k for k, v in cache.items() if v is None]
    print(f"done: {len(cache)} prefixes, {len(missing)} failed")


if __name__ == "__main__":
    main()
