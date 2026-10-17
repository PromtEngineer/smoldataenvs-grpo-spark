"""Apply rlimits, then exec the real command. Used instead of preexec_fn (thread-safe).

    launch.py <cpu_seconds> <fsize_mb> <mem_mb|0> -- argv...
"""
import os
import resource
import sys


def main() -> None:
    cpu, fsize_mb, mem_mb = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    argv = sys.argv[sys.argv.index("--") + 1:]
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb << 20, fsize_mb << 20))
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
    except (ValueError, OSError):
        pass
    if mem_mb:
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
