"""Ch.20 SS20.4 - linParts vs. nestedParts (exercises 3+4+5).

Atom: row i (i = 1..N) of a triangular-cost workload over one security's raw
close-price history - the max simple return over every lookback window
length k=1..i ending at day i. Row i costs O(i) (i inner-loop iterations),
same shape as the book's motivating cases (expanding-window SADF, multi-window
barrier touches), deliberately kept unvectorized so the cost is real (same
spirit as ex02's naive SMA).

The N rows are split into N_WORKERS molecules two ways:
  - lin_parts    - equal *row-count* molecules (book's linParts)
  - nested_parts - equal *work* molecules (book's nestedParts), since row
                   cost increases with row index here (upper_triang=False)

Both schemes run the identical atom function via ProcessPoolExecutor - only
how atoms are grouped into molecules differs. A single-thread baseline is
included for context.

Run:
    python -m learning.ex03_partitioning
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from learning.data import load_close_series
from mpengine.partition import lin_parts, nested_parts, parts_to_molecules
from learning.timing import print_table, time_runs

TICKER = "AAPL"
N = 7500
N_WORKERS = os.cpu_count() or 4


# -- the atom, two ways --------------------------------------------------------


def max_return_row(prices: np.ndarray, i: int) -> float:
    """Row i (1-indexed): max simple return over every lookback k=1..i ending at day i."""
    best = -np.inf
    for k in range(1, i + 1):
        r = prices[i] / prices[i - k] - 1.0
        if r > best:
            best = r
    return best


def max_return_row_reference(prices: np.ndarray, i: int) -> float:
    return float(np.max(prices[i] / prices[:i] - 1.0))


# -- molecule runner (module-level: Windows `spawn` needs this importable) ----


def run_molecule(molecule: tuple[list[int], np.ndarray]) -> tuple[dict[int, float], float]:
    rows, prices = molecule
    t0 = time.perf_counter()
    out = {i: max_return_row(prices, i) for i in rows}
    return out, time.perf_counter() - t0


def build_molecules(parts: np.ndarray, atoms: list[int], prices: np.ndarray) -> list[tuple[list[int], np.ndarray]]:
    return [(mol, prices) for mol in parts_to_molecules(atoms, parts)]


# -- the 3 variants -------------------------------------------------------------


def run_single_thread(atoms: list[int], prices: np.ndarray) -> dict[int, float]:
    return {i: max_return_row(prices, i) for i in atoms}


def run_pooled(molecules, n_workers: int) -> tuple[dict[int, float], list[float]]:
    results: dict[int, float] = {}
    elapsed: list[float] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for part, dt in ex.map(run_molecule, molecules):
            results.update(part)
            elapsed.append(dt)
    return results, elapsed


# -- diagnostics ----------------------------------------------------------------


def print_cost_diagnostic(name: str, atoms: list[int], parts: np.ndarray) -> None:
    molecules = parts_to_molecules(atoms, parts)
    costs = [sum(m) for m in molecules]
    print(f"{name} - {len(molecules)} molecules")
    for idx, (mol, c) in enumerate(zip(molecules, costs), 1):
        print(f"  molecule {idx}: size={len(mol):5,d} rows   cost={c:12,d}")
    lo, hi = min(costs), max(costs)
    print(f"  cost min={lo:,}  max={hi:,}  ratio(max/min)={hi / lo:.3f}x\n")


def main() -> None:
    prices = load_close_series(TICKER, adjusted="raw")
    assert len(prices) >= N + 1, f"{TICKER} has only {len(prices)} rows, need >= {N + 1}"
    prices = prices[-(N + 1):]
    atoms = list(range(1, N + 1))
    print(f"{TICKER}: using last {N + 1:,} raw close prices ({N:,} atoms), N_WORKERS={N_WORKERS}\n")

    # correctness: naive vs vectorized reference, a handful of sample rows
    sample_rows = [1, N // 4, N // 2, 3 * N // 4, N]
    max_diff = max(abs(max_return_row(prices, i) - max_return_row_reference(prices, i)) for i in sample_rows)
    print(f"correctness check (sample rows {sample_rows}) - max abs diff naive vs reference: {max_diff:.3e}\n")

    lin = lin_parts(N, N_WORKERS)
    nested = nested_parts(N, N_WORKERS, upper_triang=False)
    employed_workers = min(N_WORKERS, N)
    print(f"multiprocessing: employing {employed_workers} worker process(es) for {employed_workers} molecule(s)\n")

    print_cost_diagnostic("lin_parts (equal row-count)", atoms, lin)
    print_cost_diagnostic("nested_parts (equal work)", atoms, nested)

    lin_molecules = build_molecules(lin, atoms, prices)
    nested_molecules = build_molecules(nested, atoms, prices)

    _, lin_elapsed = run_pooled(lin_molecules, N_WORKERS)
    _, nested_elapsed = run_pooled(nested_molecules, N_WORKERS)
    print("observed per-molecule elapsed time (seconds), same molecule order as diagnostic above:")
    print(f"  lin_parts:    {[round(t, 3) for t in lin_elapsed]}")
    print(f"  nested_parts: {[round(t, 3) for t in nested_elapsed]}\n")

    results = {
        "A single-thread": time_runs(lambda: run_single_thread(atoms, prices)),
        "B multiproc+lin_parts": time_runs(lambda: run_pooled(lin_molecules, N_WORKERS)),
        "C multiproc+nested_parts": time_runs(lambda: run_pooled(nested_molecules, N_WORKERS)),
    }
    print_table(results, baseline="A single-thread")

    lin_ms = results["B multiproc+lin_parts"]["per_call_ms"]
    nested_ms = results["C multiproc+nested_parts"]["per_call_ms"]
    print(f"\nnested_parts vs lin_parts speedup: {lin_ms / nested_ms:.2f}x")

    print(
        "\nExpected: B degrades toward the cost of its single busiest molecule, since the "
        "other workers finish early and idle; C should track close to the ideal "
        "N_WORKERS-fold speedup, since all molecules finish together. Read the printed "
        "numbers rather than assuming this - process-spawn overhead on Windows can "
        "compress both differences below their theoretical values, as ex02 found."
    )


if __name__ == "__main__":
    main()
