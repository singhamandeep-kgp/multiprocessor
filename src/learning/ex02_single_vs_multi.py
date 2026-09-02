"""Ch.20 SS20.3 - single-thread vs. multithreading vs. multiprocessing.

Atom: one security's full raw close-price history -> its W-day simple moving
average (SMA). ~300 securities pulled from the curated Delta lake in one
query, split into per-security arrays, then computed 4 ways:

  A. single-thread   - plain loop, no executor,           naive atom
  B. multi-threading - ThreadPoolExecutor over molecules,  naive atom (same fn as A)
  C. multiprocessing - ProcessPoolExecutor over molecules, naive atom (same fn as A/B)
  D. multiprocessing - ProcessPoolExecutor over molecules, vectorized atom

A vs. B isolates the GIL (identical function, only the executor changes).
B vs. C isolates threads vs. processes. C vs. D isolates the vectorization
win on top of processes.

Run:
    python -m learning.ex02_single_vs_multi
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import polars as pl
from statarb.data.api import DataAPI

from mpengine.partition import equal_chunks
from learning.timing import print_table, time_runs

N_SECURITIES = 1200
MIN_OBS = 2000
WINDOW = 20
N_WORKERS = os.cpu_count() or 4


# -- the atom, two ways -------------------------------------------------------


def sma_naive(prices: np.ndarray, window: int) -> np.ndarray:
    n = len(prices)
    out = np.empty(n - window + 1, dtype=np.float64)
    for i in range(window - 1, n):
        out[i - window + 1] = sum(prices[i - window + 1 : i + 1]) / window
    return out


def sma_vectorized(prices: np.ndarray, window: int) -> np.ndarray:
    c = np.concatenate(([0.0], np.cumsum(prices, dtype=np.float64)))
    return (c[window:] - c[:-window]) / window


# -- molecule runners (module-level: Windows `spawn` needs these importable) -


def run_molecule_naive(molecule: tuple[list[int], dict[int, np.ndarray], int]) -> dict[int, np.ndarray]:
    sids, prices, window = molecule
    return {sid: sma_naive(prices[sid], window) for sid in sids}


def run_molecule_vectorized(molecule: tuple[list[int], dict[int, np.ndarray], int]) -> dict[int, np.ndarray]:
    sids, prices, window = molecule
    return {sid: sma_vectorized(prices[sid], window) for sid in sids}


def build_molecules(sids, prices_by_sid, window, n_workers):
    return [
        (chunk, {sid: prices_by_sid[sid] for sid in chunk}, window)
        for chunk in equal_chunks(sids, n_workers)
    ]


# -- the 4 variants ------------------------------------------------------------


def run_single_thread(sids, prices_by_sid, window):
    return {sid: sma_naive(prices_by_sid[sid], window) for sid in sids}


def run_pooled(molecules, executor_cls, mapper_fn, n_workers):
    results: dict[int, np.ndarray] = {}
    with executor_cls(max_workers=n_workers) as ex:
        for part in ex.map(mapper_fn, molecules):
            results.update(part)
    return results


# -- data ----------------------------------------------------------------------


def load_universe(n: int, min_obs: int) -> tuple[list[int], dict[int, np.ndarray]]:
    api = DataAPI()
    sec = api.securities.filter(pl.col("n_obs") > min_obs).sort("sid").head(n)
    sids = sec["sid"].to_list()
    px = api.get_prices(sids=sids, adjusted="raw", columns=["sid", "d", "px_close"]).sort(["sid", "d"])
    prices_by_sid = {sid: px.filter(pl.col("sid") == sid)["px_close"].to_numpy() for sid in sids}
    return sids, prices_by_sid


def main() -> None:
    sids, prices_by_sid = load_universe(N_SECURITIES, MIN_OBS)
    total_days = sum(len(p) for p in prices_by_sid.values())
    print(
        f"{len(sids)} securities, {total_days:,} total price rows, "
        f"window={WINDOW}, N_WORKERS={N_WORKERS}\n"
    )

    # correctness: naive vs vectorized atom, one security
    sample_sid = sids[0]
    naive_sample = sma_naive(prices_by_sid[sample_sid], WINDOW)
    vec_sample = sma_vectorized(prices_by_sid[sample_sid], WINDOW)
    max_diff = np.max(np.abs(naive_sample - vec_sample))
    print(f"correctness check (sid={sample_sid}) - max abs diff naive vs vectorized: {max_diff:.3e}\n")

    molecules = build_molecules(sids, prices_by_sid, WINDOW, N_WORKERS)
    employed_workers = min(N_WORKERS, len(molecules))
    print(
        f"C/D multiprocessing: employing {employed_workers} worker process(es) "
        f"for {len(molecules)} molecule(s) (N_WORKERS={N_WORKERS})\n"
    )

    results = {
        "A single-thread": time_runs(lambda: run_single_thread(sids, prices_by_sid, WINDOW)),
        "B multi-threading": time_runs(
            lambda: run_pooled(molecules, ThreadPoolExecutor, run_molecule_naive, N_WORKERS)
        ),
        "C multiprocessing": time_runs(
            lambda: run_pooled(molecules, ProcessPoolExecutor, run_molecule_naive, N_WORKERS)
        ),
        "D multiproc+vectorized": time_runs(
            lambda: run_pooled(molecules, ProcessPoolExecutor, run_molecule_vectorized, N_WORKERS)
        ),
    }
    print_table(results, baseline="A single-thread")

    print(
        "\nExpected: B ~= A (GIL blocks real parallelism for this CPU-bound atom); "
        "C meaningfully > A (processes run truly in parallel); "
        "D the largest (C's process win compounded with ex01's vectorization win). "
        "Read the printed numbers rather than assuming this - process-spawn "
        "overhead on Windows is real and can eat into C/D at this problem size."
    )

if __name__ == "__main__":
    main()
