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
    """
    num_threads_ = min(num_threads, num_atoms)
    parts = [0.0]
    for _ in range(num_threads_):
        prev = parts[-1]
        part = 1 + 4 * (prev**2 + prev + num_atoms * (num_atoms + 1) / num_threads_)
        part = (-1 + part**0.5) / 2
        parts.append(part)
    parts_arr = np.round(parts).astype(int)
    if upper_triang:
        parts_arr = np.concatenate(([0], np.cumsum(np.diff(parts_arr)[::-1])))
    return parts_arr


def parts_to_molecules(atoms: Sequence[T], parts: np.ndarray) -> list[list[T]]:
    """Slice a flat atom list into molecules given boundary indices `parts`
    (as returned by `lin_parts`/`nested_parts`): molecule m = atoms[parts[m-1]:parts[m]].
    """
    return [list(atoms[parts[i - 1]:parts[i]]) for i in range(1, len(parts))]
