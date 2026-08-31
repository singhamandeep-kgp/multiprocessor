"""Ch.20 Exercise 1 — vectorization.

Compute daily simple returns, ret[i] = price[i]/price[i-1] - 1, over one
security's full raw close-price history, four different ways, and time each:

  1. scalar loop            - explicit Python for-loop with indexing
  2. matrix algebra          - numpy elementwise array division
  3. compiled iterator       - map()/zip() (C-implemented iteration protocol)
  4. compiled generator      - a generator expression over zip()

Run:
    python -m learning.ex01_vectorization_returns
"""

from __future__ import annotations

import numpy as np

from learning.data import load_close_series
from learning.timing import print_table, time_it

TICKER = "AAPL"


def returns_scalar_loop(prices: np.ndarray) -> list[float]:
    n = len(prices)
    out = [0.0] * (n - 1)
    for i in range(1, n):
        out[i - 1] = prices[i] / prices[i - 1] - 1.0
    return out


def returns_matrix_algebra(prices: np.ndarray) -> np.ndarray:
    return prices[1:] / prices[:-1] - 1.0


def returns_compiled_iterator(prices: np.ndarray) -> list[float]:
    return list(map(lambda pq: pq[1] / pq[0] - 1.0, zip(prices[:-1], prices[1:])))


def returns_compiled_generator(prices: np.ndarray) -> list[float]:
    return list(p1 / p0 - 1.0 for p0, p1 in zip(prices[:-1], prices[1:]))


def main() -> None:
    prices = load_close_series(TICKER, adjusted="raw")
    n = len(prices)
    print(f"{TICKER}: loaded {n:,} raw close prices\n")

    methods = {
        "scalar loop": returns_scalar_loop,
        "compiled generator": returns_compiled_generator,
        "compiled iterator (map/zip)": returns_compiled_iterator,
        "vectorized (numpy)": returns_matrix_algebra,
    }

    outputs = {name: np.asarray(fn(prices), dtype=np.float64) for name, fn in methods.items()}
    ref = outputs["vectorized (numpy)"]
    max_diff = max(np.max(np.abs(out - ref)) for out in outputs.values())
    print(f"correctness check - max abs diff across methods: {max_diff:.3e}\n")

    results = {name: time_it(lambda fn=fn: fn(prices)) for name, fn in methods.items()}
    print_table(results, baseline="scalar loop")


if __name__ == "__main__":
    main()
