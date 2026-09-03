"""Atom -> molecule partitioning helpers, used by every exercise."""

from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def equal_chunks(seq: Sequence[T], n: int) -> list[list[T]]:
    """Split `seq` into up to `n` contiguous, roughly equal-size chunks.

    Appropriate when atoms have uniform cost (linParts territory) - not the
    book's actual triangular-cost `nestedParts`, which gets its own exercise.
    """
    n = max(1, min(n, len(seq)))
    q, r = divmod(len(seq), n)
    chunks = []
    start = 0
    for i in range(n):
        size = q + (1 if i < r else 0)
        if size == 0:
            break
        chunks.append(list(seq[start:start + size]))
        start += size
    return chunks


def lin_parts(num_atoms: int, num_threads: int) -> np.ndarray:
    """Ch.20 Snippet 20.5 (`linParts`) - boundary indices for up to
    `num_threads` equal-*count* partitions of `num_atoms` atoms of uniform
    cost. Molecule m (1-indexed) is `atoms[parts[m-1]:parts[m]]` - see
    `parts_to_molecules`.
    """
    num_threads_ = min(num_threads, num_atoms)
    parts = np.linspace(0, num_atoms, num_threads_ + 1)
    return np.ceil(parts).astype(int)


def _integer_boundaries(float_parts: list[float], num_atoms: int, num_molecules: int) -> np.ndarray:
    """Turn exact float boundaries into integer ones without collapsing any
    molecule to zero width.

    Rounding each boundary independently (the obvious `np.round(parts)`) lets
    two adjacent float boundaries land on the same integer, which yields an
    empty molecule - i.e. a dispatched job with no work in it. That is not
    rare: for `nested_parts` it happened for ~68% of (num_atoms, num_threads)
    pairs under 80, including such ordinary cases as (10, 8).

    Instead, allocate integer *widths* by largest remainder: floor each float
    width, force a floor of 1 atom per molecule, then reconcile the total back
    to `num_atoms` by handing surplus atoms to the largest fractional
    remainders (or reclaiming from the smallest). A valid all-non-empty
    assignment always exists here because callers clamp
    `num_molecules <= num_atoms` first.
    """
    widths = np.diff(np.asarray(float_parts, dtype=float))
    base = np.floor(widths).astype(np.int64)
    frac = widths - np.floor(widths)
    base = np.maximum(base, 1)

    shortfall = int(num_atoms) - int(base.sum())
    if shortfall > 0:
        order = np.argsort(-frac, kind="stable")
        for k in range(shortfall):
            base[order[k % num_molecules]] += 1
    elif shortfall < 0:
        order = np.argsort(frac, kind="stable")
        k = 0
        guard = 0
        while shortfall < 0 and guard < 100 * num_molecules:
            idx = order[k % num_molecules]
            if base[idx] > 1:
                base[idx] -= 1
                shortfall += 1
            k += 1
            guard += 1

    return np.concatenate(([0], np.cumsum(base))).astype(int)


def nested_parts(num_atoms: int, num_threads: int, upper_triang: bool = False) -> np.ndarray:
    """Ch.20 Snippet 20.6 (`nestedParts`) - boundary indices for up to
    `num_threads` equal-*work* partitions of a triangular-cost workload where
    atom i costs O(i) (e.g. an expanding-window computation over i atoms).

    Each boundary r_m is the positive root of
        (1/2)(r_m + r_{m-1} + 1)(r_m - r_{m-1}) = N(N+1) / (2M)
    solved iteratively from r_0 = 0, keeping intermediate boundaries as
    floats and rounding once at the end (rounding progressively would
    compound error).

    `upper_triang=True` reverses which end carries the heavy rows - use it
    when atom cost *decreases* with atom index instead of increasing.

    The float boundaries are solved exactly and converted to integers by
    `_integer_boundaries`, which guarantees no molecule comes back empty -
    rounding each boundary independently silently produced zero-width
    molecules for the majority of input pairs.
    """
    num_threads_ = min(num_threads, num_atoms)
    if num_atoms <= 0 or num_threads_ <= 0:
        return np.array([0], dtype=int)
    parts = [0.0]
    for _ in range(num_threads_):
        prev = parts[-1]
        part = 1 + 4 * (prev**2 + prev + num_atoms * (num_atoms + 1) / num_threads_)
        part = (-1 + part**0.5) / 2
        parts.append(part)
    parts_arr = _integer_boundaries(parts, num_atoms, num_threads_)
    if upper_triang:
        parts_arr = np.concatenate(([0], np.cumsum(np.diff(parts_arr)[::-1])))
    return parts_arr


def parts_to_molecules(atoms: Sequence[T], parts: np.ndarray) -> list[list[T]]:
    """Slice a flat atom list into molecules given boundary indices `parts`
    (as returned by `lin_parts`/`nested_parts`): molecule m = atoms[parts[m-1]:parts[m]].
    """
    return [list(atoms[parts[i - 1]:parts[i]]) for i in range(1, len(parts))]
