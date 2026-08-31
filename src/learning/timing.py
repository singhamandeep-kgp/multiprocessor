"""Shared timeit harness + table printer, used by every exercise."""

from __future__ import annotations

import time
import timeit
from typing import Callable


def time_runs(fn: Callable[[], object], *, repeat: int = 3) -> dict:
    """Time `fn` as `repeat` whole, independent calls; report the best.

    For variants whose per-call cost is whole seconds (e.g. spinning up a
    process pool), `timeit`'s `number=` would just repeat that setup cost
    needlessly. Each call here is timed once with `perf_counter`, and the
    setup cost (e.g. pool startup) is deliberately included each repeat since
    it's part of what's being measured, not noise to average away.
    """
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    return {"min_total_s": best, "per_call_ms": best * 1000, "number": 1}


def time_it(fn: Callable[[], object], *, repeat: int = 5, number: int = 100) -> dict:
    """Time `fn` `number` times per repeat, `repeat` times over, report the best.

    Following the book's own convention (min of several `timeit.Timer.repeat`
    runs) rather than a mean: the minimum isolates the method's true cost from
    scheduling noise, since noise can only ever slow a run down, never speed
    one up below its floor.
    """
    times = timeit.repeat(fn, repeat=repeat, number=number)
    best = min(times)
    return {"min_total_s": best, "per_call_ms": best / number * 1000, "number": number}


def print_table(results: dict[str, dict], baseline: str) -> None:
    base = results[baseline]["per_call_ms"]
    print(f"{'method':32s} {'per-call (ms)':>14s} {'speedup vs ' + baseline:>18s}")
    for name, r in results.items():
        speedup = base / r["per_call_ms"]
        print(f"{name:32s} {r['per_call_ms']:14.4f} {speedup:17.1f}x")
