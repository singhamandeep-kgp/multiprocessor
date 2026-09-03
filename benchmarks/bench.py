"""mpengine performance benchmarks, against joblib and the raw stdlib.

Four workloads, each isolating one dispatch cost that mpengine 0.3.1 paid and
0.4.0 does not:

  1. many tiny jobs      - per-job submission overhead        (chunksize)
  2. one shared payload  - per-job serialization of a closure (broadcast)
  3. BLAS-heavy jobs     - native thread oversubscription     (blas_threads)
  4. repeated dispatch   - pool spawn/teardown per call       (reuse_pool)

Run it::

    python benchmarks/bench.py            # all four
    python benchmarks/bench.py --quick    # smaller sizes, for a fast check
    python benchmarks/bench.py 1 3        # only those numbered benchmarks

Numbers are wall-clock on whatever machine you run it on and will not match
the README's, which were taken on 8 cores with OpenBLAS. What should hold
anywhere is the *shape*: 'before' is what mpengine did prior to 0.4.0, kept
runnable here by passing the parameters that switch each optimization back
off, so the comparison stays honest rather than being a remembered number.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from joblib import Parallel, delayed

from mpengine.engine import process_jobs, shutdown_pools

WORKERS = os.cpu_count() or 4


def _time(label: str, fn) -> float:
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    print(f"    {label:<44} {elapsed:8.2f}s")
    return elapsed


def _verdict(before: float, after: float, reference: float | None = None) -> None:
    print(f"    {'-' * 44} {'-' * 8}")
    print(f"    speedup vs mpengine 0.3.1 behaviour          {before / after:7.1f}x")
    if reference is not None:
        print(f"    ratio to reference (lower is better)         {after / reference:7.2f}x")
    print()


# --------------------------------------------------------------------------
# 1. Many tiny jobs - submission overhead
# --------------------------------------------------------------------------
def tiny(x: int) -> int:
    return x + 1


def bench_tiny_jobs(quick: bool) -> None:
    n = 4_000 if quick else 20_000
    print(f"[1] {n:,} trivial jobs on {WORKERS} workers - per-job dispatch overhead")
    jobs = [{"func": tiny, "x": i} for i in range(n)]
    common = dict(task="tiny", n_workers=WORKERS, text_progress=False, milestones=False)

    before = _time(
        "mpengine, chunksize=1 (0.3.1 behaviour)",
        lambda: process_jobs(jobs, chunksize=1, **common),
    )
    after = _time(
        "mpengine, chunksize='auto'",
        lambda: process_jobs(jobs, **common),
    )
    ref = _time(
        "joblib (auto batching)",
        lambda: Parallel(n_jobs=WORKERS)(delayed(tiny)(i) for i in range(n)),
    )
    _verdict(before, after, ref)


# --------------------------------------------------------------------------
# 2. One large payload shared by every job - serialization cost
# --------------------------------------------------------------------------
def _column_mean(col: int, panel: np.ndarray) -> float:
    return float(panel[:, col].mean())


def bench_shared_payload(quick: bool) -> None:
    n = 60 if quick else 200
    shape = (4_000, 1_000) if quick else (10_000, 1_000)
    panel = np.random.default_rng(0).standard_normal(shape)
    print(
        f"[2] {n} jobs sharing a {panel.nbytes / 1e6:.0f} MB panel on {WORKERS} workers "
        "- payload serialization"
    )

    # 0.3.1 had no broadcast, so the panel could only travel by closure capture
    # - i.e. once per job. Note that batching already improves this case on its
    # own: cloudpickle memoizes a repeated object within a single dumps() call,
    # so the panel now travels once per BATCH rather than once per job. That is
    # why the honest comparison needs a realistically large payload - below
    # roughly 10 MB, batching alone has already closed the gap and broadcast
    # costs more than it saves on a cold pool.
    def captured(col: int) -> float:
        return float(panel[:, col].mean())

    common = dict(task="panel", n_workers=WORKERS, text_progress=False, milestones=False)
    jobs = [{"func": _column_mean, "col": i} for i in range(n)]

    before = _time(
        "mpengine, closure capture (0.3.1 behaviour)",
        lambda: process_jobs([{"func": captured, "col": i} for i in range(n)], **common),
    )
    after = _time(
        "mpengine, broadcast=",
        lambda: process_jobs(jobs, broadcast={"panel": panel}, **common),
    )
    # The payload is delivered by the pool initializer, so a pool kept alive
    # pays for it once ever rather than once per call - which is the shape any
    # real sweep has, and where this stops being a tuning knob and starts being
    # a different order of magnitude.
    process_jobs(jobs, broadcast={"panel": panel}, reuse_pool=True, **common)
    warm = _time(
        "mpengine, broadcast= + reuse_pool=True (warm)",
        lambda: process_jobs(jobs, broadcast={"panel": panel}, reuse_pool=True, **common),
    )
    shutdown_pools()
    ref = _time(
        "joblib (auto memmap)",
        lambda: Parallel(n_jobs=WORKERS)(delayed(captured)(i) for i in range(n)),
    )
    print(f"    panel copies piped: ~{min(WORKERS * 4, n)} by closure capture "
          f"(one per batch), {WORKERS} by broadcast, 0 once the pool is warm")
    _verdict(before, after, ref)
    print(f"    with the pool warm, broadcast is {before / warm:.1f}x the 0.3.1 "
          f"behaviour and {warm / ref:.2f}x joblib\n")


# --------------------------------------------------------------------------
# 3. BLAS-heavy jobs - native thread oversubscription
# --------------------------------------------------------------------------
def _svd(k: int) -> float:
    a = np.random.default_rng(k).standard_normal((700, 700))
    return float(np.linalg.svd(a, compute_uv=False)[0])


def bench_blas(quick: bool) -> None:
    n = 8 if quick else 24
    print(f"[3] {n} SVD jobs on {WORKERS} workers - BLAS thread oversubscription")
    jobs = [{"func": _svd, "k": i} for i in range(n)]
    common = dict(task="svd", n_workers=WORKERS, text_progress=False, milestones=False)

    before = _time(
        "mpengine, blas_threads=None (0.3.1 behaviour)",
        lambda: process_jobs(jobs, blas_threads=None, **common),
    )
    after = _time(
        "mpengine, blas_threads='auto'",
        lambda: process_jobs(jobs, **common),
    )
    _verdict(before, after)


# --------------------------------------------------------------------------
# 4. Repeated dispatch - pool spawn/teardown
# --------------------------------------------------------------------------
def bench_pool_reuse(quick: bool) -> None:
    calls = 5 if quick else 10
    print(f"[4] {calls} back-to-back dispatches on {WORKERS} workers - pool spawn cost")
    jobs = [{"func": tiny, "x": i} for i in range(WORKERS)]
    common = dict(task="reuse", n_workers=WORKERS, text_progress=False, milestones=False)

    before = _time(
        "mpengine, fresh pool each call (0.3.1 behaviour)",
        lambda: [process_jobs(jobs, **common) for _ in range(calls)],
    )
    # Warm the cached pool first, so this measures steady-state reuse rather
    # than one spawn amortized over the loop.
    process_jobs(jobs, reuse_pool=True, **common)
    after = _time(
        "mpengine, reuse_pool=True",
        lambda: [process_jobs(jobs, reuse_pool=True, **common) for _ in range(calls)],
    )
    shutdown_pools()
    print(f"    saved per call: {(before - after) / calls * 1000:,.0f} ms")
    _verdict(before, after)


BENCHMARKS = {
    1: bench_tiny_jobs,
    2: bench_shared_payload,
    3: bench_blas,
    4: bench_pool_reuse,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", nargs="*", type=int, choices=list(BENCHMARKS))
    parser.add_argument("--quick", action="store_true", help="smaller sizes")
    args = parser.parse_args()

    print(f"python {sys.version.split()[0]} | {WORKERS} cores | "
          f"numpy {np.__version__}{' | quick' if args.quick else ''}\n")
    for key in args.which or sorted(BENCHMARKS):
        BENCHMARKS[key](args.quick)


if __name__ == "__main__":
    main()
