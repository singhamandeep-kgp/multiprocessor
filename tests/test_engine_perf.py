"""BLAS thread governance, executor reuse, and dead-worker detection.

These are the 0.4.0 features whose whole value is what happens inside a real
worker process, so most of this file is necessarily marked slow.
"""

from __future__ import annotations

import os
import time
from concurrent.futures.process import BrokenProcessPool

import pytest

from mpengine.engine import (
    _BLAS_ENV_VARS,
    _blas_env,
    _pool_cache,
    _resolve_blas_threads,
    process_jobs,
    shutdown_pools,
)
from tests import workers
from tests.util import log_text


# --------------------------------------------------------------------------
# The thread budget arithmetic
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_pool", [8, 4, 2, 1])
def test_auto_budget_divides_the_machine_by_concurrent_jobs(n_pool):
    """`n_pool` is the divisor - not n_workers, and not the job count. It is
    how many jobs actually run at once: with 4 jobs, 8 cores and n_workers=2,
    only two run concurrently, so each deserves 4 threads. Dividing by the job
    count would say 2 and leave half the machine idle.
    """
    cores = os.cpu_count() or 1
    assert _resolve_blas_threads("auto", n_pool) == max(1, cores // n_pool)


def test_auto_budget_is_one_when_jobs_outnumber_cores():
    """The ordinary case, and the one worth 6.7x: every core is running a job,
    so each job's BLAS gets exactly one thread."""
    cores = os.cpu_count() or 1
    assert _resolve_blas_threads("auto", cores * 4) == 1


def test_explicit_budget_is_passed_through():
    assert _resolve_blas_threads(3, 8) == 3


def test_none_disables_governance():
    assert _resolve_blas_threads(None, 8) is None


@pytest.mark.parametrize("bad", [0, -1, "some", 2.5, True])
def test_budget_rejects_nonsense(bad):
    with pytest.raises(ValueError, match="blas_threads"):
        _resolve_blas_threads(bad, 8)


# --------------------------------------------------------------------------
# The environment context manager
# --------------------------------------------------------------------------
def test_blas_env_sets_every_variable_and_restores_them():
    before = {v: os.environ.get(v) for v in _BLAS_ENV_VARS}
    with _blas_env(2):
        assert all(os.environ[v] == "2" for v in _BLAS_ENV_VARS)
    assert {v: os.environ.get(v) for v in _BLAS_ENV_VARS} == before


def test_blas_env_restores_a_preexisting_value_rather_than_deleting_it():
    """A caller who deliberately set OMP_NUM_THREADS must get it back exactly,
    not have it removed because mpengine borrowed the variable for a run."""
    os.environ["OMP_NUM_THREADS"] = "7"
    try:
        with _blas_env(1):
            assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["OMP_NUM_THREADS"] == "7"
    finally:
        os.environ.pop("OMP_NUM_THREADS", None)


def test_blas_env_is_a_no_op_when_disabled():
    os.environ.pop("OMP_NUM_THREADS", None)
    with _blas_env(None):
        assert "OMP_NUM_THREADS" not in os.environ


def test_blas_env_restores_even_if_the_body_raises():
    before = {v: os.environ.get(v) for v in _BLAS_ENV_VARS}
    with pytest.raises(RuntimeError):
        with _blas_env(4):
            raise RuntimeError("boom")
    assert {v: os.environ.get(v) for v in _BLAS_ENV_VARS} == before


# --------------------------------------------------------------------------
# What the worker actually sees
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_the_budget_reaches_the_worker_process():
    """The parent setting an environment variable is only a means; what
    matters is the value a spawned child reads before it loads numpy."""
    out = process_jobs(
        [{"func": workers.env_var, "name": "OMP_NUM_THREADS"} for _ in range(4)],
        task="env", n_workers=2, blas_threads=3,
        text_progress=False, milestones=False,
    )
    assert set(out) == {"3"}


@pytest.mark.slow
def test_native_libraries_in_the_worker_honour_the_budget():
    """Ground truth, rather than the environment variable standing in for it.

    This is the assertion that covers the fork path, where the child inherits
    an already-loaded OpenBLAS and the environment variable arrives far too
    late - only the threadpoolctl call inside the worker reaches it there.
    """
    out = process_jobs(
        [{"func": workers.blas_threads_seen} for _ in range(4)],
        task="blas", n_workers=2, blas_threads=1,
        text_progress=False, milestones=False,
    )
    reported = [n for counts in out for n in counts]
    if not reported:
        pytest.skip("no threadpoolctl-visible native libraries in this environment")
    assert set(reported) == {1}, f"a worker BLAS was left unconstrained: {reported}"


@pytest.mark.slow
def test_disabling_governance_leaves_the_worker_environment_alone():
    os.environ.pop("OMP_NUM_THREADS", None)
    out = process_jobs(
        [{"func": workers.env_var, "name": "OMP_NUM_THREADS"} for _ in range(2)],
        task="noenv", n_workers=2, blas_threads=None,
        text_progress=False, milestones=False,
    )
    assert set(out) == {None}


@pytest.mark.slow
def test_the_resolved_budget_is_logged(caplog_mpengine):
    """A run's own log has to record what it actually did, or the number is
    unauditable after the fact."""
    process_jobs(
        [{"func": workers.square, "x": 1} for _ in range(2)],
        task="logged", n_workers=2, blas_threads=2,
        text_progress=False, milestones=False,
    )
    assert "blas_threads=2" in log_text(caplog_mpengine)


# --------------------------------------------------------------------------
# Executor reuse
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_reuse_pool_keeps_the_same_workers_between_calls():
    jobs = [{"func": workers.current_pid} for _ in range(6)]
    common = dict(task="reuse", n_workers=2, text_progress=False, milestones=False)

    first = set(process_jobs(jobs, reuse_pool=True, **common))
    second = set(process_jobs(jobs, reuse_pool=True, **common))
    assert first == second, "a reused pool must not respawn its workers"
    shutdown_pools()


@pytest.mark.slow
def test_without_reuse_each_call_gets_fresh_workers():
    jobs = [{"func": workers.current_pid} for _ in range(6)]
    common = dict(task="fresh", n_workers=2, text_progress=False, milestones=False)
    first = set(process_jobs(jobs, **common))
    second = set(process_jobs(jobs, **common))
    assert not (first & second), "a torn-down pool's pids must not come back"


@pytest.mark.slow
def test_shutdown_pools_empties_the_cache():
    process_jobs(
        [{"func": workers.square, "x": 1} for _ in range(2)],
        task="cache", n_workers=2, reuse_pool=True,
        text_progress=False, milestones=False,
    )
    assert _pool_cache, "reuse_pool=True should have parked an executor"
    shutdown_pools()
    assert not _pool_cache


@pytest.mark.slow
def test_pools_with_different_settings_are_not_shared():
    """The cache key covers everything the executor was built from, so a pool
    built for 2 workers can never be handed to a call asking for 3."""
    common = dict(task="keys", text_progress=False, milestones=False, reuse_pool=True)
    jobs = [{"func": workers.square, "x": 1} for _ in range(4)]
    process_jobs(jobs, n_workers=2, **common)
    process_jobs(jobs, n_workers=3, **common)
    assert len(_pool_cache) == 2
    shutdown_pools()


# --------------------------------------------------------------------------
# Dead-worker detection - the highest-stakes behaviour in the library
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_a_dead_worker_raises_instead_of_hanging_forever():
    """`mp.Pool` cannot detect a worker killed mid-job: it only resolves a task
    slot when a result reaches the output queue, and a dead process posts
    nothing - so the run blocked forever, with no exception and no log line.
    An unattended overnight sweep hanging silently is far worse than a loud
    failure, which is why dispatch moved to ProcessPoolExecutor.
    """
    started = time.perf_counter()
    with pytest.raises(BrokenProcessPool, match="a worker process died"):
        process_jobs(
            [{"func": workers.suicide, "x": i, "victim": 3} for i in range(12)],
            task="death", n_workers=3, chunksize=1,
            text_progress=False, milestones=False,
        )
    assert time.perf_counter() - started < 30, "detection must be prompt, not eventual"


@pytest.mark.slow
def test_the_death_message_says_how_far_the_run_got(caplog_mpengine):
    with pytest.raises(BrokenProcessPool) as excinfo:
        process_jobs(
            [{"func": workers.suicide, "x": i, "victim": 2} for i in range(8)],
            task="context", n_workers=2, chunksize=1,
            text_progress=False, milestones=False,
        )
    assert "/8 jobs completed" in str(excinfo.value)
    assert "orchestrator.run" in str(excinfo.value), "point the reader at durability"
    assert "worker process died" in log_text(caplog_mpengine)


@pytest.mark.slow
def test_a_broken_reused_pool_is_evicted_from_the_cache():
    """A broken pool is permanently unusable. Leaving it parked would make
    every later call inherit the corpse."""
    with pytest.raises(BrokenProcessPool):
        process_jobs(
            [{"func": workers.suicide, "x": i, "victim": 1} for i in range(6)],
            task="evict", n_workers=2, chunksize=1, reuse_pool=True,
            text_progress=False, milestones=False,
        )
    assert not _pool_cache, "the dead pool must not be handed to the next caller"

    # ... and the next call must simply work.
    out = process_jobs(
        [{"func": workers.square, "x": 3} for _ in range(2)],
        task="after", n_workers=2, reuse_pool=True,
        text_progress=False, milestones=False,
    )
    assert out == [9, 9]
    shutdown_pools()
