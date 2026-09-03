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
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable

import cloudpickle

_log = logging.getLogger("mpengine.engine")
# Lifecycle logging (dispatch start/done, worker-count clamping, progress
# milestones) is always captured in full by orchestrator.run()'s run.log, but
# is deliberately never allowed to bubble up past this logger to whatever the
# CALLER's own logging setup is (root, via logging.basicConfig or similar) -
# a caller configuring logging for their own unrelated purposes should not
# suddenly also see mpengine's internal chatter. Only "mpengine.summary" (the
# stored-here paths and the worker ranking, see orchestrator.py) is meant to
# surface there.
_log.propagate = False


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
    """Executor target: undo the cloudpickle wrapping `process_jobs` applies
    before submission, then dispatch exactly as `expand_call` always has.

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


def _infer_task_name(jobs: list[dict[str, Any]]) -> str:
    """Best-effort display name for a job list.

    The obvious `jobs[0]["func"].__name__` crashes with AttributeError on the
    very callables this engine goes out of its way to support: a
    `functools.partial` (a natural way to pin one big fitted object once) and
    any class instance implementing `__call__` have no `__name__`. Degrade to
    the type name, then to a constant, rather than refusing to run.
    """
    func = jobs[0].get("func")
    return getattr(func, "__name__", None) or type(func).__name__ or "job"


def process_jobs(
    jobs: list[dict[str, Any]],
    task: str | None = None,
    n_workers: int = os.cpu_count() or 4,
    on_progress: Callable[[int, float, Any], None] | None = None,
    on_job_error: Callable[[int, BaseException], None] | None = None,
    text_progress: bool | None = None,
    milestones: bool = True,
) -> list[Any]:
    """Ch.20 Snippet 20.9 (`processJobs`) - parallel dispatch over a process pool.

    Named `n_workers`, not `n_threads`: this spawns OS *processes*, each with
    its own interpreter and memory space, not threads sharing one. That
    distinction matters here specifically - ex02 measured that CPython's GIL
    makes threads add nothing for CPU-bound work, which is exactly what this
    engine dispatches.

    Results arrive in *completion* order, not submission order, which is what
    lets `report_progress` report honestly as each job lands - with uneven job
    costs (see ex03), an early-submitted heavy job can finish long after
    several later-submitted light ones. (`orchestrator.run` re-sorts back to
    submission order before returning, since its results are label-addressed.)

    Each job is serialized with `cloudpickle` before submission (rather than
    relying on the executor's own stdlib-`pickle` handling), so a job's 'func'
    - or a nested callable, like `orchestrator.run`'s `save_fn` - can be a
    closure or a lambda, not just a module-level function. `_run_from_blob` is
    itself a plain module-level function, so it still pickles by reference with
    no dependence on `__main__` spawn fixup.

    Deliberately built on `concurrent.futures.ProcessPoolExecutor` rather than
    the book's literal `mp.Pool` + `imap_unordered`. `mp.Pool` cannot detect a
    worker that DIES mid-job (OOM-killed, or segfaulting inside native code
    such as BLAS): CPython only resolves a task slot when a result arrives on
    the output queue, and a dead process posts nothing, so the run blocks
    forever with no exception and no log line. `ProcessPoolExecutor` watches
    its workers and raises `BrokenProcessPool` instead - a hang that never
    surfaces is far worse for an unattended run than a loud failure. The
    `with` block here is also genuinely graceful, unlike `mp.Pool.__exit__`,
    which calls `terminate()`.

    `on_progress`, if given, is called as `on_progress(pid, atom_seconds,
    result)` for every completed job - `pid` identifies which worker process
    produced it and `atom_seconds` is that job's pure compute time as measured
    inside the worker, which is what lets a caller (see `orchestrator.run`'s
    `show_progress`) drive a per-worker live display with real timings. The
    return value here is unaffected either way - still the plain `list[Any]`
    of results, never the `(pid, atom_seconds, result)` triples.
    `text_progress` controls the book's own `\\r`-overwriting stderr line.
    Left at None it is automatic: emitted only when no `on_progress` display
    is running AND stderr is a real terminal. That distinction matters in
    production - redirected to a log file, a `\\r` line per job is unreadable
    noise, so when the text display is off this logs periodic completion
    milestones at INFO instead - unless `milestones=False`, which a caller
    passes when `on_progress` is itself a live visual display (a tqdm bar,
    say) rather than merely a logging hook: the milestones would otherwise be
    redundant with - and print right alongside - that display. `on_progress`
    alone isn't the right signal for this, since a caller can legitimately
    want per-job callbacks (for logging, say) with no visual display at all.

    `on_job_error`, if given, is called as `on_job_error(index, exc)` for any
    job that cannot be *serialized* for submission, and that job is skipped
    instead of sinking the batch. Jobs are cloudpickled up front, so without
    this one unpicklable payload (a lock, a live socket, an open file handle
    captured by a closure) raises before any job runs at all - job 517 of 1000
    taking down the 999 that were perfectly runnable. Left as None, that
    original fail-fast behaviour is preserved, which keeps this layer
    book-faithful; `orchestrator.run` opts in to turn such a failure into one
    failed job, matching the per-job isolation it already promises.
    """
    if not jobs:
        return []
    if task is None:
        task = _infer_task_name(jobs)

    # Serialize per job rather than in one comprehension, so a single
    # unpicklable payload can be attributed and skipped instead of aborting
    # the whole submission (see `on_job_error`).
    blobs: list[bytes] = []
    for i, job in enumerate(jobs):
        try:
            blobs.append(cloudpickle.dumps(job))
        except Exception as exc:
            if on_job_error is None:
                raise
            _log.warning(
                "job could not be serialized for dispatch, skipping "
                "task=%s job_index=%d error=%s", task, i, exc,
            )
            on_job_error(i, exc)
    if not blobs:
        _log.warning("nothing dispatchable task=%s n_jobs=%d", task, len(jobs))
        return []

    # Never spin up more workers than there are jobs to hand them: the
    # executor starts its processes eagerly, so surplus workers pay full spawn
    # cost to do nothing. This only ever clamps downward - when jobs outnumber
    # n_workers no clamp is needed, since the same fixed pool keeps pulling
    # jobs until the list is done. Worker count is sized to hardware, not to
    # job count.
    out: list[Any] = []
    time0 = time.time()
    n_submitted = len(blobs)
    n_pool = min(n_workers, n_submitted)
    if n_pool < n_workers:
        _log.warning(
            "n_workers=%d clamped to %d task=%s reason=fewer jobs than workers",
            n_workers, n_pool, task,
        )

    # Auto: the book's carriage-return line is a terminal affordance, so only emit it when
    # a terminal is actually there and nothing else owns the display.
    if text_progress is None:
        text_progress = on_progress is None and sys.stderr.isatty()
    # When nothing is drawing live, log milestones so a redirected run still
    # shows movement - roughly ten lines, whatever the job count. `milestones`
    # is the caller's explicit say on this, separate from text_progress/
    # on_progress: it's the only signal that actually means "something else
    # is already showing a visual progress display, don't duplicate it."
    milestone = max(1, n_submitted // 10) if (not text_progress and milestones) else 0

    _log.info(
        "dispatch start task=%s n_jobs=%d n_workers=%d text_progress=%s",
        task, n_submitted, n_pool, text_progress,
    )
    with ProcessPoolExecutor(max_workers=n_pool) as executor:
        futures = [executor.submit(_run_from_blob, blob) for blob in blobs]
        try:
            for i, future in enumerate(as_completed(futures), 1):
                pid, atom_s, out_ = future.result()
                out.append(out_)
                if on_progress is not None:
                    on_progress(pid, atom_s, out_)
                if text_progress:
                    report_progress(i, n_submitted, time0, task)
                elif milestone and (i % milestone == 0 or i == n_submitted):
                    _log.info(
                        "progress task=%s done=%d/%d pct=%.0f elapsed_s=%.2f",
                        task, i, n_submitted, 100.0 * i / n_submitted,
                        time.time() - time0,
                    )
        except BrokenProcessPool as exc:
            # Cancel whatever has not started so shutdown does not block on
            # work that can never complete, then re-raise with the context the
            # bare stdlib error lacks: how far the run actually got.
            for future in futures:
                future.cancel()
            _log.error(
                "worker process died task=%s completed=%d/%d - run aborted",
                task, len(out), n_submitted,
            )
            raise BrokenProcessPool(
                f"a worker process died during task {task!r} after "
                f"{len(out)}/{n_submitted} jobs completed - it was most likely "
                f"OOM-killed or crashed inside native code (e.g. BLAS). The "
                f"completed jobs' results were lost with the pool; if you need "
                f"per-job durability use orchestrator.run, which saves each "
                f"result to disk as it lands."
            ) from exc
    _log.info(
        "dispatch done task=%s completed=%d/%d elapsed_s=%.2f",
        task, len(out), n_submitted, time.time() - time0,
    )
    return out