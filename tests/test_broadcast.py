"""Broadcast: shipping one shared payload per worker instead of per job."""

from __future__ import annotations

import cloudpickle
import numpy as np
import pytest

from mpengine.engine import BroadcastRef, process_jobs
from mpengine.orchestrator import _validate_broadcast, run
from tests import workers


@pytest.fixture
def panel() -> np.ndarray:
    return np.arange(200, dtype=float).reshape(20, 10)


# --------------------------------------------------------------------------
# Validation - before any directory is created
# --------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["not an identifier", "", "2leading", "with-dash", 7])
def test_broadcast_keys_must_be_usable_as_keyword_arguments(key):
    with pytest.raises(ValueError, match="valid Python identifier"):
        _validate_broadcast({key: 1}, [{"x": 1}])


def test_broadcast_may_not_collide_with_a_param_set_key():
    """A job carrying both would be two competing values for one argument, and
    silently picking either is worse than refusing."""
    with pytest.raises(ValueError, match="two values for one argument"):
        _validate_broadcast({"panel": 1}, [{"panel": 2}, {"panel": 3}])


def test_a_non_colliding_broadcast_validates():
    _validate_broadcast({"panel": 1}, [{"col": 0}, {"col": 1}])


def test_run_rejects_a_colliding_broadcast_before_creating_anything(base_dir):
    with pytest.raises(ValueError, match="two values for one argument"):
        run(workers.add, [{"x": 1, "y": 2}], base_dir=base_dir, broadcast={"y": 9})
    assert not base_dir.exists(), "a rejected call must leave no run directory behind"


# --------------------------------------------------------------------------
# The reference itself
# --------------------------------------------------------------------------
def test_a_broadcast_reference_is_tiny_on_the_wire(panel):
    """The entire point: what travels with each job is a name, not the payload.

    A job carrying the array directly is roughly the array's size; one
    carrying a reference is a few hundred bytes however large the payload is.
    """
    with_payload = cloudpickle.dumps({"func": workers.column_mean, "col": 0, "panel": panel})
    with_ref = cloudpickle.dumps(
        {"func": workers.column_mean, "col": 0, "panel": BroadcastRef("panel")}
    )
    assert len(with_ref) < len(with_payload) / 2
    assert len(with_ref) < 2_000


# --------------------------------------------------------------------------
# Delivery through a real pool
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_process_jobs_delivers_the_payload_as_a_keyword_argument(panel):
    out = process_jobs(
        [{"func": workers.column_mean, "col": i} for i in range(4)],
        task="bcast", n_workers=2, broadcast={"panel": panel},
        text_progress=False, milestones=False,
    )
    assert sorted(out) == sorted(float(panel[:, i].mean()) for i in range(4))


@pytest.mark.slow
def test_broadcast_survives_pool_reuse(panel):
    """The payload is installed by the initializer, so a kept-alive pool must
    still have it on the second call - that is what makes reuse the thing to
    pair broadcast with."""
    jobs = [{"func": workers.column_mean, "col": i} for i in range(4)]
    common = dict(task="bcast", n_workers=2, broadcast={"panel": panel},
                  reuse_pool=True, text_progress=False, milestones=False)
    first = sorted(process_jobs(jobs, **common))
    second = sorted(process_jobs(jobs, **common))
    assert first == second == sorted(float(panel[:, i].mean()) for i in range(4))


@pytest.mark.slow
def test_run_delivers_a_broadcast_to_the_inner_function(base_dir, panel):
    """`run` wraps the caller's job inside its own bookkeeping job, so the
    reference has to land on the inner one - placed on the wrapper it would be
    a keyword argument `_run_and_save_job` has no parameter for."""
    summary = run(
        workers.column_mean, [{"col": i} for i in range(5)],
        base_dir=base_dir, task="bcastrun", broadcast={"panel": panel},
    )
    assert summary.n_ok == 5 and summary.n_failed == 0


@pytest.mark.slow
def test_broadcast_works_in_debug_mode(base_dir, panel):
    """Debug mode runs sequentially in the calling process, where no pool
    initializer ever fires - the payload has to be installed locally instead
    or every reference resolves against nothing."""
    summary = run(
        workers.panel_shape, [{"_x": 0}],
        base_dir=base_dir, task="dbg", broadcast={"panel": panel}, debug=True,
    )
    assert summary.n_ok == 1, summary.results


@pytest.mark.slow
def test_multiple_broadcast_values_are_all_delivered(base_dir):
    summary = run(
        workers.add, [{}], base_dir=base_dir, task="multi",
        broadcast={"x": 2, "y": 40},
    )
    assert summary.n_ok == 1, summary.results
