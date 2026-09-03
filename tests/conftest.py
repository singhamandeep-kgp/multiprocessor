"""Shared fixtures and cross-cutting test setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cloudpickle
import pytest

import mpengine.engine as engine
from tests import workers

# Ship the job targets by value rather than by import path. A spawned worker is
# a fresh interpreter, and whether it could import `tests.workers` depends on
# how pytest arranged sys.path - which is not something the suite should be
# betting on. Registering the module makes cloudpickle embed the function in
# the job itself, so the child needs nothing on disk to reconstruct it.
cloudpickle.register_pickle_by_value(workers)


@pytest.fixture(autouse=True)
def _no_pool_leaks():
    """Never let a cached executor outlive the test that created it.

    `reuse_pool=True` deliberately parks a live pool in a module global. Left
    there, it would leak worker processes between tests and - worse - let one
    test's warm workers silently serve another's assertions about pids.
    """
    yield
    engine.shutdown_pools()


@pytest.fixture(autouse=True)
def _clean_broadcast():
    """The broadcast registry is a module global in the parent too (debug mode
    installs into it), so clear it between tests."""
    yield
    engine._BROADCAST.clear()


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    """A throwaway root for one run's outputs/logs/manifests."""
    return tmp_path / "runs"


@pytest.fixture
def caplog_mpengine(caplog):
    """Capture mpengine's own lifecycle logging.

    `mpengine.engine` and `mpengine.orchestrator` set `propagate = False` on
    purpose, so their records never reach the root logger that pytest's caplog
    attaches to. Reaching them means attaching caplog's handler to each logger
    directly - and restoring propagation afterwards, since the propagate flag
    is itself under test elsewhere.
    """
    targets = [
        logging.getLogger("mpengine.engine"),
        logging.getLogger("mpengine.orchestrator"),
        logging.getLogger("mpengine.summary"),
    ]
    previous = [(lg, lg.level) for lg in targets]
    for lg in targets:
        lg.addHandler(caplog.handler)
        lg.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG)
    try:
        yield caplog
    finally:
        for lg, level in previous:
            lg.removeHandler(caplog.handler)
            lg.setLevel(level)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: spawns real process pools; deselect with -m 'not slow'"
    )
