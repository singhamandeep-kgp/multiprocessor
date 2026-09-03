"""Job targets for the test suite.

These live in their own module, not inline in the test files, for one reason:
a spawned worker is a fresh interpreter that has to be able to reconstruct
whatever it is asked to run. `conftest.py` registers this module for
cloudpickle's pickle-by-value, so these functions travel with the job rather
than as an import path the child would have to resolve - which keeps the tests
working regardless of how pytest happens to have arranged sys.path.

Anything here may be executed inside a worker process, so keep it importable,
side-effect-free at import time, and cheap.
"""

from __future__ import annotations

import os
import time
from typing import Any


def square(x: int) -> int:
    return x * x


def add(x: int, y: int) -> int:
    return x + y


def identity(x: Any) -> Any:
    return x


def boom(x: int) -> int:
    """Fails predictably, for failure-isolation tests."""
    raise ValueError(f"job {x} exploded")


def boom_on(x: int, victim: int = 0) -> int:
    """Fails for exactly one input, so the other jobs can be asserted intact."""
    if x == victim:
        raise ValueError(f"job {x} exploded")
    return x


def slow(x: int, seconds: float = 0.05) -> int:
    time.sleep(seconds)
    return x


def suicide(x: int, victim: int = 3) -> int:
    """Kills its own worker process outright - no exception, no traceback, the
    way an OOM kill or a native segfault behaves. `mp.Pool` cannot detect this
    and hangs forever; `ProcessPoolExecutor` must raise `BrokenProcessPool`.
    """
    if x == victim:
        os._exit(1)
    time.sleep(0.05)
    return x


def current_pid(_x: Any = None) -> int:
    return os.getpid()


def env_var(name: str) -> str | None:
    """Read one environment variable from inside the worker - how the BLAS
    thread budget is observed without needing a real BLAS call."""
    return os.environ.get(name)


def blas_threads_seen(_x: Any = None) -> list[int]:
    """What the native libraries actually loaded in this worker report, which
    is the ground truth the environment variables are only a means to."""
    try:
        from threadpoolctl import threadpool_info

        return [i["num_threads"] for i in threadpool_info()]
    except Exception:
        return []


def column_mean(col: int, panel: Any) -> float:
    """Takes its large argument by name, so it can be supplied by `broadcast`."""
    return float(panel[:, col].mean())


def panel_shape(_x: Any = None, panel: Any = None) -> tuple[int, ...]:
    return tuple(panel.shape)


def save_boom(_obj: Any, _path: Any) -> None:
    """A save_fn that fails after `run` has already decided the job succeeded,
    which is what the atomic-write path exists to survive."""
    raise OSError("save deliberately failed")


class CallableObject:
    """Has no __name__ - neither does functools.partial. Task-name inference
    has to cope with both, since the library advertises support for them."""

    def __call__(self, x: int) -> int:
        return x + 1
