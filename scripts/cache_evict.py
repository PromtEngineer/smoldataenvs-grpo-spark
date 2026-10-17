#!/usr/bin/env python3
"""Drop clean page-cache pages for big read-only file trees (no root needed: posix_fadvise
DONTNEED). GB10 GPU allocations can hang when the kernel must reclaim page cache first.
    cache_evict.py [--loop SECONDS]      logs to <repo>/logs/cache-evict.log"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREES = [os.path.expanduser(p) for p in ("~/.cache/huggingface", "~/models", os.path.join(ROOT, "data/task_files"))]
def mem():
    d = dict(l.split(":", 1) for l in open("/proc/meminfo"))
    return int(d["MemFree"].split()[0]) / 1e6, int(d["Cached"].split()[0]) / 1e6
def sweep():
    n = 0
    for t in TREES:
        for root, _, files in os.walk(t):
            for f in files:
                p = os.path.join(root, f)
                try:
                    if os.path.getsize(p) < (1 << 20):
                        continue
                    fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
                except OSError:
                    pass
    return n
log = open(os.path.join(ROOT, "logs/cache-evict.log"), "a", buffering=1)
loop = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--loop" else 0
i = 0
while True:
    f0, c0 = mem(); n = sweep(); f1, c1 = mem()
    if i % 6 == 0 or not loop:
        log.write(f"{time.strftime('%T')} files={n} free {f0:.1f}->{f1:.1f} GB cached {c0:.1f}->{c1:.1f} GB\n")
        if not loop: print(f"files={n} free {f0:.1f}->{f1:.1f} GB, cached {c0:.1f}->{c1:.1f} GB")
    if not loop:
        break
    i += 1; time.sleep(loop)
