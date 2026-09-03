"""Partitioning: atoms in, molecule boundaries out.

Pure functions with no pool involved, so these are the fast, exhaustive part
of the suite - they can afford to sweep whole grids of inputs rather than
sample a few.
"""

from __future__ import annotations

import numpy as np
import pytest

from mpengine.partition import (
    _integer_boundaries,
    equal_chunks,
    lin_parts,
    nested_parts,
    parts_to_molecules,
)


# --------------------------------------------------------------------------
# equal_chunks
# --------------------------------------------------------------------------
@pytest.mark.parametrize("length,n", [(10, 3), (10, 1), (10, 10), (7, 4), (1, 5), (100, 8)])
def test_equal_chunks_conserves_and_balances(length, n):
    seq = list(range(length))
    chunks = equal_chunks(seq, n)
    assert [x for c in chunks for x in c] == seq, "atoms lost, duplicated or reordered"
    assert all(chunks), "an empty chunk is a worker handed no work"
    sizes = [len(c) for c in chunks]
    assert max(sizes) - min(sizes) <= 1, f"unbalanced: {sizes}"


def test_equal_chunks_never_exceeds_available_atoms():
    # Asking for more chunks than atoms must clamp, not emit empties.
    assert len(equal_chunks(list(range(3)), 10)) == 3


def test_equal_chunks_handles_degenerate_n():
    assert equal_chunks([1, 2, 3], 0) == [[1, 2, 3]]
    assert equal_chunks([], 4) == []


# --------------------------------------------------------------------------
# lin_parts
# --------------------------------------------------------------------------
@pytest.mark.parametrize("atoms,threads", [(10, 3), (100, 8), (5, 5), (5, 9), (1, 1)])
def test_lin_parts_boundaries_are_well_formed(atoms, threads):
    parts = lin_parts(atoms, threads)
    assert parts[0] == 0 and parts[-1] == atoms
    assert np.all(np.diff(parts) >= 0), "boundaries must not go backwards"
    assert len(parts) == min(threads, atoms) + 1


def test_lin_parts_molecules_reassemble_the_original():
    atoms = list(range(37))
    molecules = parts_to_molecules(atoms, lin_parts(len(atoms), 6))
    assert [a for m in molecules for a in m] == atoms


# --------------------------------------------------------------------------
# nested_parts - the equal-work partitioner
# --------------------------------------------------------------------------
@pytest.mark.parametrize("atoms", range(1, 60))
@pytest.mark.parametrize("threads", [1, 2, 3, 4, 7, 8, 16])
def test_nested_parts_never_emits_an_empty_molecule(atoms, threads):
    """Regression for the rounding defect fixed in 0.3.0.

    Rounding each float boundary independently let two adjacent boundaries
    land on the same integer, producing a zero-width molecule - a job
    dispatched with no work in it. It affected the majority of (atoms,
    threads) pairs, (10, 8) among them, so this sweeps a grid rather than
    trusting a couple of examples.
    """
    parts = nested_parts(atoms, threads)
    widths = np.diff(parts)
    assert np.all(widths >= 1), f"empty molecule for ({atoms}, {threads}): {parts}"


@pytest.mark.parametrize("atoms,threads", [(10, 8), (3, 3), (7500, 8), (100, 7), (1, 4)])
def test_nested_parts_conserves_every_atom(atoms, threads):
    parts = nested_parts(atoms, threads)
    assert parts[0] == 0
    assert parts[-1] == atoms, "the last boundary must account for every atom"
    assert np.all(np.diff(parts) >= 0)


def test_nested_parts_actually_equalises_triangular_work():
    """The whole point of nestedParts over linParts: when atom i costs O(i),
    equal *counts* leave one worker with most of the work. Molecule work here
    is the sum of atom indices it covers."""
    atoms, threads = 7500, 8
    parts = nested_parts(atoms, threads)
    work = [sum(range(parts[i - 1], parts[i])) for i in range(1, len(parts))]
    assert max(work) / min(work) < 1.05, f"work imbalance {max(work) / min(work):.3f}: {work}"

    # ... and demonstrate the contrast, so the test documents why it exists.
    lin = lin_parts(atoms, threads)
    lin_work = [sum(range(lin[i - 1], lin[i])) for i in range(1, len(lin))]
    assert max(lin_work) / min(lin_work) > 5, "linParts should be badly imbalanced here"


def test_nested_parts_upper_triang_reverses_the_widths():
    normal = np.diff(nested_parts(100, 5))
    reversed_ = np.diff(nested_parts(100, 5, upper_triang=True))
    assert list(reversed_) == list(normal[::-1])
    assert nested_parts(100, 5, upper_triang=True)[-1] == 100


@pytest.mark.parametrize("atoms", [0, -1])
def test_nested_parts_handles_no_atoms(atoms):
    assert list(nested_parts(atoms, 4)) == [0]


def test_nested_parts_clamps_threads_to_atoms():
    parts = nested_parts(3, 10)
    assert len(parts) == 4, "at most one molecule per atom"
    assert np.all(np.diff(parts) >= 1)


# --------------------------------------------------------------------------
# _integer_boundaries - the largest-remainder allocator behind nested_parts
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "floats,atoms,molecules",
    [
        ([0.0, 1.4, 1.6, 3.0], 3, 3),   # two boundaries that would collide
        ([0.0, 5.5, 10.0], 10, 2),
        ([0.0, 0.3, 0.6, 1.0], 3, 3),   # every width below 1
    ],
)
def test_integer_boundaries_are_total_and_non_empty(floats, atoms, molecules):
    parts = _integer_boundaries(floats, atoms, molecules)
    assert parts[0] == 0
    assert parts[-1] == atoms, "widths must sum back to the atom count"
    assert len(parts) == molecules + 1
    assert np.all(np.diff(parts) >= 1)


# --------------------------------------------------------------------------
# parts_to_molecules
# --------------------------------------------------------------------------
def test_parts_to_molecules_slices_exactly_as_documented():
    atoms = list("abcdefgh")
    assert parts_to_molecules(atoms, np.array([0, 3, 5, 8])) == [
        ["a", "b", "c"], ["d", "e"], ["f", "g", "h"],
    ]


def test_parts_to_molecules_round_trips_nested_parts():
    atoms = list(range(500))
    molecules = parts_to_molecules(atoms, nested_parts(len(atoms), 8))
    assert [a for m in molecules for a in m] == atoms
    assert all(m for m in molecules)
