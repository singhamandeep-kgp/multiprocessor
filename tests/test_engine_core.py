"""The dispatch primitives: expand_call, process_jobs_, process_jobs."""

from __future__ import annotations

import functools
import os

import pytest

from mpengine.engine import (
    BroadcastRef,
    _infer_task_name,
    expand_call,
    install_broadcast,
    process_jobs,
    process_jobs_,
)
from tests import workers
from tests.util import log_text


# --------------------------------------------------------------------------
# expand_call - the hinge the whole engine turns on
# --------------------------------------------------------------------------
def test_expand_call_dispatches_the_job_dict():
    assert expand_call({"func": workers.add, "x": 2, "y": 3}) == 5


def test_expand_call_does_not_mutate_the_caller_s_job():
    """The book's literal `kargs.pop('func')` mutates in place, so the same job
    list can only ever be dispatched once - a second pass raises KeyError.
    ex04 runs one list through both process_jobs_ and process_jobs, so the
    defensive copy is load-bearing, not decoration."""
    job = {"func": workers.square, "x": 4}
    assert expand_call(job) == 16
    assert job == {"func": workers.square, "x": 4}
    assert expand_call(job) == 16, "the job must survive being dispatched twice"


def test_expand_call_resolves_a_broadcast_reference():
    install_broadcast({"y": 40})
    assert expand_call({"func": workers.add, "x": 2, "y": BroadcastRef("y")}) == 42


def test_expand_call_reports_a_missing_broadcast_value_clearly():
    with pytest.raises(KeyError, match="never installed by the pool initializer"):
        expand_call({"func": workers.identity, "x": BroadcastRef("absent")})


def test_broadcast_ref_repr_names_its_key():
    assert repr(BroadcastRef("panel")) == "BroadcastRef('panel')"


# --------------------------------------------------------------------------
# process_jobs_ - the sequential debugging path
# --------------------------------------------------------------------------
def test_process_jobs_preserves_submission_order():
    jobs = [{"func": workers.square, "x": i} for i in range(6)]
    assert process_jobs_(jobs) == [0, 1, 4, 9, 16, 25]


def test_process_jobs_raises_at_the_failing_job():
    """No pool, so the traceback is live and attributable - that clarity is
    the entire reason this mode exists."""
    jobs = [{"func": workers.boom_on, "x": i, "victim": 2} for i in range(5)]
    with pytest.raises(ValueError, match="job 2 exploded"):
        process_jobs_(jobs)


def test_process_jobs_accepts_an_empty_list():
    assert process_jobs_([]) == []


# --------------------------------------------------------------------------
# Task-name inference
# --------------------------------------------------------------------------
def test_infer_task_name_uses_the_function_name():
    assert _infer_task_name([{"func": workers.square}]) == "square"


def test_infer_task_name_survives_callables_without_a_name():
    """functools.partial and callable objects have no __name__ - and they are
    exactly what this engine advertises support for, so falling over on them
    would be self-defeating."""
    assert _infer_task_name([{"func": functools.partial(workers.add, y=1)}]) == "partial"
    assert _infer_task_name([{"func": workers.CallableObject()}]) == "CallableObject"


# --------------------------------------------------------------------------
# process_jobs - the parallel path
# --------------------------------------------------------------------------
def test_process_jobs_returns_empty_for_no_jobs():
    """Previously an IndexError, from inferring a task name off jobs[0]."""
    assert process_jobs([]) == []


@pytest.mark.slow
def test_process_jobs_returns_every_result():
    jobs = [{"func": workers.square, "x": i} for i in range(20)]
    out = process_jobs(jobs, task="sq", n_workers=2, text_progress=False, milestones=False)
    # Completion order, by design - compare as a multiset.
    assert sorted(out) == [i * i for i in range(20)]


@pytest.mark.slow
def test_process_jobs_dispatches_closures_and_lambdas():
    """The headline feature: cloudpickle serializes a function by value, so a
    worker can run something that has no importable module path."""
    offset = 100

    def closure(x: int) -> int:
        return x + offset

    out = process_jobs(
        [{"func": closure, "x": i} for i in range(4)]
        + [{"func": lambda x: x * -1, "x": 7}],
        task="closures", n_workers=2, text_progress=False, milestones=False,
    )
    assert sorted(out) == [-7, 100, 101, 102, 103]


@pytest.mark.slow
def test_process_jobs_runs_work_in_other_processes():
    out = process_jobs(
        [{"func": workers.current_pid, "_x": i} for i in range(8)],
        task="pids", n_workers=2, text_progress=False, milestones=False,
    )
    assert os.getpid() not in out, "jobs must not run in the parent"


@pytest.mark.slow
def test_process_jobs_clamps_workers_to_the_job_count(caplog_mpengine):
    """Surplus workers pay full spawn cost to do nothing."""
    process_jobs(
        [{"func": workers.square, "x": 1}], task="one", n_workers=8,
        text_progress=False, milestones=False,
    )
    assert "n_workers=8 clamped to 1" in log_text(caplog_mpengine)
