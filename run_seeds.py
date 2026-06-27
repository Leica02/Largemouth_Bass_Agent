#!/usr/bin/env python3
"""
Parallel Seed Launcher for Aquaculture Batch Simulation
========================================================
Spawns multiple independent subprocesses of aquaculture_batch_sim.py, each
with a distinct random seed.  Each process writes its full output to a
per-seed log file while the main console displays only progress summaries
and a final status report.

Default seeds: 42, 43, 44.  Custom seeds and per-process thread limits are
configurable via command-line arguments.

Usage
-----
    python run_seeds.py                 # run default seeds (42, 43, 44)
    python run_seeds.py 1 2 3 4         # run arbitrary seeds
    python run_seeds.py --threads 2 ... # limit BLAS/torch threads per process
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "aquaculture_batch_sim.py")
DEFAULT_SEEDS: List[int] = [42, 43, 44]


def main() -> None:
    """Parse arguments and launch parallel simulation processes.

    Each seed is run in a separate subprocess with stdout redirected to a
    per-seed log file.  The launcher polls until all processes complete,
    then prints a summary table and exits with a non-zero code if any
    process failed.
    """
    parser = argparse.ArgumentParser(
        description="Run multiple aquaculture simulation seeds in parallel")
    parser.add_argument("seeds", nargs="*", type=int, default=None,
                        help="List of seeds to run (default: 42 43 44)")
    parser.add_argument("--threads", type=int, default=1,
                        help="OMP/MKL/torch thread count per process "
                             "(default 1 to avoid contention; 0 = unlimited)")
    args = parser.parse_args()

    seeds: List[int] = args.seeds if args.seeds else DEFAULT_SEEDS

    if not os.path.exists(SCRIPT):
        print(f"[ERROR] Simulation script not found: {SCRIPT}")
        sys.exit(1)

    # Restrict numerical library threads to prevent N processes from
    # saturating CPU with N x num_cores threads
    env = os.environ.copy()
    if args.threads and args.threads > 0:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = str(args.threads)

    print(f"Launching {len(seeds)} parallel seeds: {seeds}")
    if args.threads and args.threads > 0:
        print(f"Thread limit per process: {args.threads}")
    print("=" * 55)

    procs: List[Dict] = []
    t_start = time.time()
    for s in seeds:
        log_path = f"log_seed{s}.txt"
        log_file = open(log_path, "w", encoding="utf-8")
        p = subprocess.Popen(
            [sys.executable, SCRIPT, "--seed", str(s)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        procs.append({"seed": s, "proc": p, "log": log_file,
                      "log_path": log_path})
        print(f"  Started seed {s:>3d}  (PID {p.pid})  log -> {log_path}")

    print("=" * 55)
    print("All processes launched.  Waiting for completion... "
          "(use `tail -f log_seed42.txt` in another terminal for live output)")

    # Poll until all processes finish
    results: Dict[int, int] = {}
    remaining = {item["seed"]: item for item in procs}
    while remaining:
        for s, item in list(remaining.items()):
            rc = item["proc"].poll()
            if rc is not None:
                item["log"].close()
                results[s] = rc
                elapsed = time.time() - t_start
                status = "OK" if rc == 0 else f"FAILED (exit code {rc})"
                print(f"  seed {s:>3d} finished: {status}  "
                      f"(elapsed {elapsed/60:.1f} min)")
                del remaining[s]
        time.sleep(2)

    total = time.time() - t_start
    print("=" * 55)
    print("All processes complete.  Summary:")
    for s in seeds:
        rc = results.get(s)
        if rc == 0:
            print(f"  seed {s:>3d}: OK  -> aquaculture_results_seed{s}/")
        else:
            print(f"  seed {s:>3d}: FAILED (exit code {rc}), "
                  f"see log_seed{s}.txt")
    print(f"Total elapsed time: {total/60:.1f} min")

    # Exit non-zero if any child process failed
    if any(rc != 0 for rc in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
