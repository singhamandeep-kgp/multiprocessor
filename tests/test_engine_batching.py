"""Batching: the 0.4.0 dispatch rewrite.

Jobs now travel to workers in batches rather than one submission each. That
closed a 5.3x overhead gap, but it moved the boundary that per-job reporting
and per-job failure attribution are built on - so these tests pin down that
nothing downstream can tell the difference.
"""

from __future__ import annotations

import threading

import pytest

from mpengine.engine import _resolve_chunksize, process_jobs
from tests import workers
from tests.util import log_text


# --------------------------------------------------------------------------
# chunksize resolution
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_jobs,n_pool,expected",
    [
        (20_000, 8, 625),   # enough batches to keep 8 workers fed several times
        (8, 8, 1),          # small run: one job per batch, per-job granularity kept
        (100, 8, 3),
        (1, 1, 1),
        (32, 8, 1),         # exactly at the 4x-per-worker threshold
    ],
)
def test_resolve_chunksize_auto(n_jobs, n_pool, expected):
    assert _resolve_chunksize("auto", n_jobs, n_pool) == expected


def test_resolve_chunksize_honours_an_explicit_value():
    assert _resolve_chunksize(25, 20_000, 8) == 25


@pytest.mark.parametrize("bad", [0, -1, "big", 2.5, True, None])
def test_resolve_chunksize_rejects_nonsense(bad):
    with pytest.raises(ValueError, match="chunksize"):
        _resolve_chunksize(bad, 100, 8)


def test_auto_chunksize_keeps_per_job_granularity_on_small_runs():
    """A human watching 8 jobs land should see 8 events, not 1. This is why
    'auto' collapses to 1 rather than always batching."""
    assert _resolve_chunksize("auto", 8, 8) == 1


# --------------------------------------------------------------------------
# Per-job reporting survives batching
# --------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("chunksize", [1, 5, 25])
def test_on_progress_fires_once_per_job_whatever_the_batch_size(chunksize):
    n = 50
    seen: list[tuple[int, float]] = []
    process_jobs(
        [{"func": workers.square, "x": i} for i in range(n)],
        task="prog", n_workers=2, chunksize=chunksize,
        on_progress=lambda pid, secs, _r: seen.append((pid, secs)),
        text_progress=False, milestones=False,
    )
    assert len(seen) == n, "a batch must be unpacked before anything downstream sees it"
    assert all(secs >= 0 for _pid, secs in seen)


@pytest.mark.slow
def test_atom_seconds_measures_the_job_not_the_batch():
    """Timing is taken around each individual call inside the worker. If it
    were taken around the batch, twenty 50 ms jobs sharing a batch would each
    report ~1s instead of ~0.05s."""
    seen: list[float] = []
    process_jobs(
        [{"func": workers.slow, "x": i, "seconds": 0.05} for i in range(12)],
        task="timing", n_workers=2, chunksize=6,
        on_progress=lambda _p, secs, _r: seen.append(secs),
        text_progress=False, milestones=False,
    )
    assert len(seen) == 12
    assert all(0.02 < s < 0.5 for s in seen), f"batch time leaked into atom time: {seen}"


@pytest.mark.slow
def test_batching_returns_every_result():
    n = 60
    out = process_jobs(
        [{"func": workers.square, "x": i} for i in range(n)],
        task="batched", n_workers=2, chunksize=7,
        text_progress=False, milestones=False,
    )
    assert sorted(out) == sorted(i * i for i in range(n))


@pytest.mark.slow
def test_milestones_count_jobs_not_futures(caplog_mpengine):
    """With 100 jobs in batches of 10 the run must report done=100/100, not
    done=10/10 - the counter tracks futures unless it is told otherwise."""
    process_jobs(
        [{"func": workers.square, "x": i} for i in range(100)],
        task="miles", n_workers=2, chunksize=10,
        text_progress=False, milestones=True,
    )
    milestones = [
        line for line in log_text(caplog_mpengine).splitlines() if "progress task=" in line
    ]
    assert milestones, "milestone logging did not fire at all"
    # Every milestone must be denominated in jobs (/100), never in the ten
    # batches those jobs travelled in.
    assert all("/100 " in line for line in milestones), milestones
    assert any("done=100/100" in line for line in milestones), milestones


@pytest.mark.slow
def test_milestones_can_be_suppressed(caplog_mpengine):
    """A caller driving its own live display passes milestones=False, so the
    engine's fallback logging does not print alongside a tqdm bar."""
    process_jobs(
        [{"func": workers.square, "x": i} for i in range(40)],
        task="quiet", n_workers=2, text_progress=False, milestones=False,
    )
    assert "progress task=quiet" not in log_text(caplog_mpengine)


# --------------------------------------------------------------------------
# Failure attribution survives batching
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_unpicklable_job_is_isolated_without_sinking_its_batch():
    """The optimistic path dumps a whole batch in one call. When that fails it
    has to fall back and walk the batch job by job, or one bad payload takes
    its four batch-mates down with it and the index is lost."""
    jobs = [{"func": workers.square, "x": i} for i in range(20)]
    jobs[7]["x"] = threading.Lock()   # cannot be cloudpickled

    failed: list[int] = []
    out = process_jobs(
        jobs, task="isolate", n_workers=2, chunksize=5,
        on_job_error=lambda i, _exc: failed.append(i),
        text_progress=False, milestones=False,
    )
    assert failed == [7], "the culprit must be named by its original index"
    assert len(out) == 19, "the other four jobs in batch 5-9 must still have run"


@pytest.mark.slow
def test_unpicklable_job_still_fails_fast_without_a_handler():
    """Without on_job_error the book-faithful behaviour is preserved: raise
    rather than silently drop work."""
    jobs = [{"func": workers.square, "x": i} for i in range(10)]
    jobs[3]["x"] = threading.Lock()
    with pytest.raises(TypeError):
        process_jobs(jobs, task="strict", n_workers=2,
                     text_progress=False, milestones=False)


@pytest.mark.slow
def test_all_jobs_unpicklable_returns_empty(caplog_mpengine):
    jobs = [{"func": workers.square, "x": threading.Lock()} for _ in range(3)]
    out = process_jobs(
        jobs, task="none", n_workers=2, on_job_error=lambda *_: None,
        text_progress=False, milestones=False,
    )
    assert out == []
    assert "nothing dispatchable" in log_text(caplog_mpengine)
