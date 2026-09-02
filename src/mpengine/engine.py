"""Ch.20 SS20.5 - the multiprocessing engine (exercises 6+7).

expand_call / process_jobs_ / process_jobs / report_progress - a generic
job-dict-to-function-call pipeline, reusable by any exercise rather than
tied to one atom/molecule shape. This is the book's own point: stop writing
a bespoke parallelization wrapper per function, and build one library that
can parallelize unknown functions regardless of their arguments or output.

A "job" here is just a dict: one 'func' entry (the callback) plus whatever
kwargs that callback needs. `expand_call` turns such a dict into a call,
which is the hinge the whole engine turns on.
"""

from __future__ import annotations

import datetime as dt
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Callable

import cloudpickle


def expand_call(kargs: dict[str, Any]) -> Any:
    """Ch.20 Snippet 20.10 (`expandCall`) - turn a job dict into a call.

    Deviation from the book: `kargs` is copied before popping. The book's
    literal `kargs.pop('func')` mutates the *caller's* dict in place, so the
    same job list can only ever be run once - a second pass raises
    KeyError('func'). ex04 deliberately runs one job list through both
    `process_jobs_` and `process_jobs` to compare them, so the copy matters.
    """
    kargs = dict(kargs)
    func = kargs.pop("func")
    return func(**kargs)


def process_jobs_(jobs: list[dict[str, Any]]) -> list[Any]:
    """Ch.20 Snippet 20.8 (`processJobs_`) - sequential fallback, for debugging.

    Runs every job in-process, one at a time, in submission order. No pool,
    no pickling, no spawn - so a failing job raises immediately, at its exact
    position in the list, with an ordinary live traceback you can attach a
    debugger to. That clarity is the entire reason this mode exists.
    """
    return [expand_call(job) for job in jobs]


def report_progress(job_num: int, num_jobs: int, time0: float, task: str) -> None:
    """Ch.20 Snippet 20.9's `reportProgress`, modernized.

    Overwrites its own line via '\\r' until the final job, then emits '\\n'.
    Units stay the book's minutes even though a fast demo run always shows
    "0.00 minutes" - the snippet is sized for real multi-minute workloads.
    """
    frac = job_num / num_jobs
    minutes_elapsed = (time.time() - time0) / 60.0
    minutes_remaining = minutes_elapsed * (1 / frac - 1) if frac > 0 else 0.0
    timestamp = dt.datetime.now().isoformat(sep=" ", timespec="seconds")
    msg = (
        f"{timestamp} {frac * 100:5.1f}% {task} done after {minutes_elapsed:.2f} "
        f"minutes. Remaining {minutes_remaining:.2f} minutes."
    )
    print(msg, end="\n" if job_num >= num_jobs else "\r", file=sys.stderr, flush=True)


def _run_from_blob(blob: bytes) -> tuple[int, float, Any]:
    """Pool target: undo the cloudpickle wrapping `process_jobs` applies before
    submission, then dispatch exactly as `expand_call` always has.

    Returns `(pid, atom_seconds, result)`. The pid attributes the completion to
    a specific worker process; `atom_seconds` is measured around the call
    itself, *inside* the worker, so it is pure compute time - it cannot include
    queueing, spawn cost, or the idle gap between this atom finishing and the
    rest of the run finishing. That distinction is the whole point: timing the
    same thing from the parent would measure when the result *arrived*, not how
    long the work actually took.
    """
    job = cloudpickle.loads(blob)
    t0 = time.perf_counter()
    result = expand_call(job)
    return os.getpid(), time.perf_counter() - t0, result


def process_jobs(
    jobs: list[dict[str, Any]],
    task: str | None = None,
    n_workers: int = os.cpu_count() or 4,
    on_progress: Callable[[int, float, Any], None] | None = None,
) -> list[Any]:
    """Ch.20 Snippet 20.9 (`processJobs`) - real `mp.Pool` + `imap_unordered`.

    Named `n_workers`, not `n_threads`: this spawns OS *processes* via
    `mp.Pool`, each with its own interpreter and memory space, not threads
    sharing one. That distinction matters here specifically - ex02 measured
    that CPython's GIL makes threads add nothing for CPU-bound work, which is
    exactly what this engine dispatches.

    Results arrive in *completion* order, not submission order, which is what
    lets `report_progress` report honestly as each job lands - with uneven job
    costs (see ex03), an early-submitted heavy job can finish long after
    several later-submitted light ones.

    Each job is serialized with `cloudpickle` before submission (rather than
    relying on `Pool`'s own stdlib-`pickle` handling of `jobs`), so a job's
    'func' - or a nested callable, like `orchestrator.run`'s `save_fn` - can be
    a closure or a lambda, not just a module-level function. `_run_from_blob`
    is itself a plain module-level function, so it still pickles by reference
    with no dependence on `__main__` spawn fixup.

    `on_progress`, if given, is called as `on_progress(pid, atom_seconds,
    result)` for every completed job - `pid` identifies which worker process
    produced it and `atom_seconds` is that job's pure compute time as measured
    inside the worker, which is what lets a caller (see `orchestrator.run`'s
    `show_progress`) drive a per-worker live display with real timings. The
    return value here is unaffected either way - still the plain `list[Any]`
    of results, never the `(pid, atom_seconds, result)` triples.
    Supplying `on_progress` means the caller is rendering its own progress
    display, so the book's own text-based `report_progress` is skipped for
    that call - the two would otherwise both write to stderr and garble each
    other's redraws.

    Uses explicit close()+join() rather than `with mp.Pool(...) as pool:`,
    whose __exit__ calls terminate() - an abrupt kill, not the book's graceful
    shutdown. The try/finally still lets a job's exception propagate while
    guaranteeing the workers are torn down.
    """
    if task is None:
        task = jobs[0]["func"].__name__
    blobs = [cloudpickle.dumps(job) for job in jobs]
    # `Pool(processes=n)` spawns exactly n OS processes immediately, whether
    # or not there's n jobs' worth of work - never spin up more workers than
    # there are jobs to hand them. This only ever clamps downward: when jobs
    # outnumber n_workers, no clamp is needed at all - the same fixed pool
    # keeps pulling jobs off the queue, one per worker at a time, until the
    # whole list is done. Worker count is sized to hardware, not job count.
    pool = mp.Pool(processes=min(n_workers, len(jobs)))
    out: list[Any] = []
    time0 = time.time()
    try:
        for i, (pid, atom_s, out_) in enumerate(pool.imap_unordered(_run_from_blob, blobs), 1):
            out.append(out_)
            if on_progress is not None:
                on_progress(pid, atom_s, out_)
            else:
                report_progress(i, len(jobs), time0, task)
    finally:
        pool.close()
        pool.join()
    return out