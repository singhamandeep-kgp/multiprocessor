"""mpengine - a small, general-purpose multiprocessing engine.

Two layers, which you can use at whichever level suits you:

- ``engine``        the primitives. A job is just a dict carrying its own
                    callback plus that callback's kwargs; ``expand_call`` turns
                    such a dict into a call, and ``process_jobs`` /
                    ``process_jobs_`` dispatch them in parallel or sequentially.
- ``orchestrator``  the production layer. ``run()`` adds a manifest of exactly
                    what was launched, one log file per worker process, results
                    saved to disk, and failure isolation so one bad job cannot
                    lose the rest of the run.

Most callers only ever need ``run``::

    from mpengine import run

    def my_task(x, y):
        return x * y

    summary = run(
        my_task,
        [{"x": 2, "y": 3}, {"x": 4, "y": 5}],
        base_dir="runs",
    )
    print(summary.n_ok, summary.n_failed)

That writes ``runs/manifests/<run_id>.txt``, ``runs/outputs/<run_id>/`` and
``runs/logs/<run_id>/``.

Worker targets - ``func`` itself, and any custom ``save_fn`` - can be closures
or lambdas, not just module-level functions. Jobs are serialized with
``cloudpickle`` before crossing the process boundary, which can pickle a
function by value (not just by reference). The one caveat: whatever a closure
captures travels with every job that uses it.

The dispatch core follows Lopez de Prado, *Advances in Financial Machine
Learning*, Ch.20 - the docstrings name the specific snippets - but nothing here
is finance-specific; it parallelizes any callable.
"""

from mpengine.engine import expand_call, process_jobs, process_jobs_, report_progress
from mpengine.orchestrator import (
    JobResult,
    RunSummary,
    WorkerStats,
    load_pickle,
    load_run_outputs,
    run,
    save_pickle,
)
from mpengine.partition import equal_chunks, lin_parts, nested_parts, parts_to_molecules

__all__ = [
    # the usual entry point
    "run",
    "RunSummary",
    "JobResult",
    "WorkerStats",
    "save_pickle",
    # reading a finished run's outputs back
    "load_run_outputs",
    "load_pickle",
    # lower-level dispatch, if you want to drive the pool yourself
    "expand_call",
    "process_jobs",
    "process_jobs_",
    "report_progress",
    # atom -> molecule partitioning
    "equal_chunks",
    "lin_parts",
    "nested_parts",
    "parts_to_molecules",
]

__version__ = "0.1.2"
