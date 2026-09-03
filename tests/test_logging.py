"""The terminal/production logging split.

mpengine deliberately routes three kinds of output to three different places:
lifecycle chatter (captured in run.log, never propagated), the run summary
(propagated, so a deployment's own handlers see it), and the banner plus live
display (a real terminal only). None of that is visible from a return value,
so it needs testing directly or it will quietly rot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from mpengine.orchestrator import _has_external_handler, _safe_task_name, run
from tests import workers


# --------------------------------------------------------------------------
# Logger wiring
# --------------------------------------------------------------------------
def test_the_package_installs_only_a_null_handler():
    """A library must never configure logging for its host application - it
    names its loggers and stays silent until the caller asks."""
    import mpengine  # noqa: F401  (import for the side effect under test)

    handlers = logging.getLogger("mpengine").handlers
    assert handlers, "without a NullHandler, callers see 'no handlers' warnings"
    assert all(isinstance(h, logging.NullHandler) for h in handlers)


@pytest.mark.parametrize("name", ["mpengine.engine", "mpengine.orchestrator"])
def test_lifecycle_loggers_do_not_propagate(name):
    """This is what stops a caller who ran logging.basicConfig() for their own
    unrelated reasons from also receiving mpengine's internal chatter."""
    assert logging.getLogger(name).propagate is False


def test_the_summary_logger_does_propagate():
    """The summary is the one thing meant to reach a deployment's own
    handlers, which is why it is a sibling of the lifecycle loggers rather
    than a child of one."""
    summary = logging.getLogger("mpengine.summary")
    assert summary.propagate is True
    assert summary.parent is logging.getLogger("mpengine")


# --------------------------------------------------------------------------
# External-handler detection
# --------------------------------------------------------------------------
def test_a_root_handler_counts_as_externally_configured():
    root = logging.getLogger()
    handler = logging.NullHandler()
    root.addHandler(handler)
    try:
        assert _has_external_handler() is True
    finally:
        root.removeHandler(handler)


def test_the_packages_own_null_handler_does_not_count():
    """Otherwise mpengine would mistake its own no-op handler for the caller
    having taken control, and never show the summary on a terminal.

    pytest's own logging plugin always attaches a handler to root, so the
    unconfigured state has to be staged deliberately - skipping instead would
    mean this never actually ran anywhere.
    """
    root = logging.getLogger()
    stashed, root.handlers = root.handlers, []
    try:
        assert _has_external_handler() is False
    finally:
        root.handlers = stashed


def test_a_real_handler_on_the_package_logger_counts():
    """A caller who attached a handler to `mpengine` directly, rather than via
    basicConfig, has still taken control and must not be overridden."""
    root = logging.getLogger()
    package = logging.getLogger("mpengine")
    stashed, root.handlers = root.handlers, []
    handler = logging.StreamHandler()
    package.addHandler(handler)
    try:
        assert _has_external_handler() is True
    finally:
        package.removeHandler(handler)
        root.handlers = stashed


# --------------------------------------------------------------------------
# run.log - the durable per-run record
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_run_log_captures_the_whole_lifecycle(base_dir):
    """The per-worker files record what each worker did; run.log records what
    the run as a whole did, so one run is one self-contained record. It has to
    capture the loggers that do NOT propagate, which is why the handler is
    attached to each of them directly rather than to the shared parent."""
    summary = run(workers.square, [{"x": i} for i in range(4)],
                  base_dir=base_dir, task="record")
    text = (Path(summary.log_dir) / "run.log").read_text(encoding="utf-8")

    assert "run start" in text
    assert "dispatch start" in text          # mpengine.engine, propagate=False
    assert "dispatch done" in text
    assert "run done" in text                # mpengine.orchestrator, propagate=False
    assert "Output stored here" in text      # mpengine.summary, propagates
    assert summary.run_id in text


@pytest.mark.slow
def test_run_log_records_a_failure_the_parent_would_otherwise_never_see(base_dir):
    """Until this existed a failed job was written only to that worker's own
    file, so an operator watching the run's logs saw nothing at all until
    someone read the manifest."""
    summary = run(workers.boom_on, [{"x": i, "victim": 1} for i in range(3)],
                  base_dir=base_dir, task="failrec")
    text = (Path(summary.log_dir) / "run.log").read_text(encoding="utf-8")
    assert "job FAILED" in text
    assert "job_0001" in text


@pytest.mark.slow
def test_run_log_is_written_regardless_of_caller_configuration(base_dir):
    """No basicConfig anywhere in this test - the file must still exist."""
    summary = run(workers.square, [{"x": 1}], base_dir=base_dir, task="always")
    assert (Path(summary.log_dir) / "run.log").stat().st_size > 0


@pytest.mark.slow
def test_handlers_are_removed_again_after_a_run(base_dir):
    """Attached and removed per run - otherwise every run would leave another
    FileHandler behind and the tenth run would write ten copies of every line
    into nine stale files."""
    before = {
        name: list(logging.getLogger(name).handlers)
        for name in ("mpengine", "mpengine.engine", "mpengine.orchestrator")
    }
    run(workers.square, [{"x": 1}], base_dir=base_dir, task="cleanup")
    after = {
        name: list(logging.getLogger(name).handlers)
        for name in ("mpengine", "mpengine.engine", "mpengine.orchestrator")
    }
    assert after == before


@pytest.mark.slow
def test_logger_levels_are_restored_after_a_run(base_dir):
    """run.log lowers each logger to INFO to capture the lifecycle. Left
    lowered, it would make a caller's own output noisier than they configured
    for every run after the first."""
    levels_before = {
        name: logging.getLogger(name).level
        for name in ("mpengine", "mpengine.engine", "mpengine.orchestrator")
    }
    run(workers.square, [{"x": 1}], base_dir=base_dir, task="levels")
    levels_after = {
        name: logging.getLogger(name).level
        for name in ("mpengine", "mpengine.engine", "mpengine.orchestrator")
    }
    assert levels_after == levels_before


# --------------------------------------------------------------------------
# What a caller with their own logging setup receives
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_a_configured_caller_gets_the_summary_but_not_the_chatter(base_dir):
    """The whole point of the split. A caller who configured logging for their
    own reasons should see where the run put things, and nothing else."""
    root = logging.getLogger()
    captured: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record):
            captured.append(record)

    handler = Collector(level=logging.DEBUG)
    root.addHandler(handler)
    previous = root.level
    root.setLevel(logging.DEBUG)
    try:
        run(workers.square, [{"x": i} for i in range(3)],
            base_dir=base_dir, task="split")
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)

    names = {r.name for r in captured}
    assert "mpengine.summary" in names, "the caller must be told where things landed"
    assert "mpengine.engine" not in names, "lifecycle chatter must not leak"
    assert "mpengine.orchestrator" not in names

    messages = " ".join(r.getMessage() for r in captured)
    assert "Output stored here" in messages
    assert "Manifest stored here" in messages
    assert "dispatch start" not in messages


@pytest.mark.slow
def test_worker_logs_name_the_process_that_wrote_them(base_dir):
    """One log file per worker *process*, not per job, so a failure is
    attributable to a specific process rather than just a stack trace."""
    summary = run(workers.boom_on, [{"x": i, "victim": 0} for i in range(4)],
                  base_dir=base_dir, task="attrib", n_workers=2, chunksize=1)
    worker_logs = list(Path(summary.log_dir).glob("worker_*.log"))
    assert worker_logs
    combined = "\n".join(p.read_text() for p in worker_logs)
    assert "failed job_0000" in combined
    assert "job 0 exploded" in combined
