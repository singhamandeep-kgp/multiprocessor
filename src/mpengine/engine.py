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

import atexit
import datetime as dt
import hashlib
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from typing import Any, Callable, Iterator

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


# ---------------------------------------------------------------------------
# BLAS thread governance
# ---------------------------------------------------------------------------
#
# numpy hands matrix work to a native BLAS (OpenBLAS, MKL, Accelerate), and
# that BLAS is itself multi-threaded: one np.linalg.svd call silently spreads
# across every core it can see. Harmless in a single process - catastrophic in
# a pool, because each worker loads its OWN copy of the library, each of which
# still believes it owns the whole machine. Eight workers on eight cores means
# eight processes x eight BLAS threads = 64 threads contending for 8 cores, and
# the OS spends its time context-switching instead of computing. Measured on an
# 8-core box: 24 SVD jobs took 8.13s unconstrained and 0.86s with the budget
# applied - a 9.5x penalty, invisible and entirely self-inflicted.
#
# The fix is to hand each worker a thread budget, so parallelism comes from
# mpengine (one job per core) rather than from every job's BLAS trying to grab
# the whole box at once.
_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _resolve_blas_threads(blas_threads: int | str | None, n_pool: int) -> int | None:
    """Resolve the per-worker BLAS thread budget. None means "leave it alone".

    'auto' divides the machine by `n_pool` - the number of jobs actually
    running *concurrently*, which is the only correct divisor. It is not
    `n_workers` (the caller's request, before the clamp in `process_jobs`) and
    not the job count: with 4 jobs, 8 cores and an explicit `n_workers=2`, only
    two jobs run at a time, so each deserves 4 threads. Dividing by the job
    count would say 2 and leave half the machine idle.

    In the ordinary case - more jobs than cores - this lands on 1, which is
    what makes the 9.5x above.
    """
    if blas_threads is None:
        return None
    if blas_threads == "auto":
        return max(1, (os.cpu_count() or 1) // max(1, n_pool))
    if isinstance(blas_threads, bool) or not isinstance(blas_threads, int):
        raise ValueError(
            f"blas_threads must be 'auto', an int, or None - got {blas_threads!r}"
        )
    if blas_threads < 1:
        raise ValueError(f"blas_threads must be >= 1 - got {blas_threads}")
    return blas_threads


@contextmanager
def _blas_env(budget: int | None) -> Iterator[None]:
    """Set the BLAS thread environment for the duration of a dispatch, so that
    workers inherit it, then restore exactly what was there before.

    This is the half of the fix that covers `spawn` (the Windows and macOS
    default): a spawned child is a fresh interpreter that reads these variables
    when it first imports numpy, so setting them in the parent before the pool
    starts is what reaches it. The parent's own already-loaded BLAS is
    unaffected, since these are only ever consulted at library load time.

    It has to stay set for the whole dispatch, not just the executor's
    constructor: ProcessPoolExecutor starts its workers lazily as tasks are
    submitted, so a worker can be spawned well after construction.

    The other half - `threadpool_limits` inside `_worker_init` - is what covers
    `fork`, where the child inherits an already-loaded library and the
    environment variable arrives far too late to matter.
    """
    if budget is None:
        yield
        return
    previous = {var: os.environ.get(var) for var in _BLAS_ENV_VARS}
    for var in _BLAS_ENV_VARS:
        os.environ[var] = str(budget)
    try:
        yield
    finally:
        for var, value in previous.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


# ---------------------------------------------------------------------------
# Broadcast payloads
# ---------------------------------------------------------------------------
#
# Jobs are cloudpickled individually, which is what lets a closure or lambda
# cross the process boundary - but it also means everything a closure captures
# travels once PER JOB. Measured: 200 jobs sharing one 8 MB numpy panel
# serialized and piped 1.6 GB, for 8 MB of actual data.
#
# `broadcast` sends such a payload once per WORKER instead, via the pool's
# initializer, and each job refers to it by name. 200 x 8 MB becomes 8 x 8 MB.
_BROADCAST: dict[str, Any] = {}


class BroadcastRef:
    """A placeholder standing in a job dict for a value that was broadcast to
    the workers rather than shipped with the job.

    `expand_call` swaps it for the real object at call time. Keeping the
    substitution in `expand_call` - rather than in the dispatch loop - is what
    makes it work at any nesting depth: `orchestrator.run` wraps the caller's
    job inside its own bookkeeping job, and both are expanded through here, so
    the reference resolves against the user's function and never leaks into the
    wrapper's signature.
    """

    __slots__ = ("key",)

    def __init__(self, key: str) -> None:
        self.key = key

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BroadcastRef({self.key!r})"


def install_broadcast(payload: dict[str, Any]) -> None:
    """Publish broadcast values into this process's registry.

    Called inside each worker by `_worker_init`, and in the parent by
    `orchestrator.run` when `debug=True` - debug mode runs sequentially in the
    calling process, where no worker initializer ever fires, so without this
    the references would have nothing to resolve against.
    """
    _BROADCAST.update(payload)


def _worker_init(blob: bytes) -> None:
    """Pool initializer: runs once per worker process, before its first job.

    Everything that should happen once per worker rather than once per job is
    funnelled through here - the BLAS budget, the broadcast payload, and the
    caller's own initializer. It takes a single cloudpickled blob so that the
    caller's initializer may be a closure or a lambda, consistent with the rest
    of the library; `_worker_init` itself stays a plain module-level function,
    so the executor can pickle it by reference with no `__main__` fixup.
    """
    budget, payload, user_init, user_args = cloudpickle.loads(blob)

    if budget is not None:
        for var in _BLAS_ENV_VARS:
            os.environ[var] = str(budget)
        try:
            from threadpoolctl import threadpool_limits

            # Deliberately not used as a context manager: the limit must hold
            # for this worker's whole life, not one block. Kept referenced on
            # the module so it cannot be collected mid-run.
            global _thread_limiter
            _thread_limiter = threadpool_limits(limits=budget)
        except Exception:  # pragma: no cover - threadpoolctl is best-effort
            # The environment variables above already cover spawn; this is the
            # fork-specific belt-and-braces, and a missing or unhappy
            # threadpoolctl must never take down a run over a tuning hint.
            pass

    if payload:
        install_broadcast(payload)

    if user_init is not None:
        user_init(*user_args)


_thread_limiter: Any = None


# ---------------------------------------------------------------------------
# Executor reuse
# ---------------------------------------------------------------------------
#
# Building a pool costs real time - roughly 280 ms per `process_jobs` call on
# an 8-core Windows box, paid again on every call because the executor is
# created and torn down inside the dispatch. A caller invoking `run()` in a
# loop pays it every iteration for nothing.
_pool_cache: dict[tuple[Any, ...], ProcessPoolExecutor] = {}


def _new_executor(n_pool: int, init_blob: bytes, max_tasks_per_child: int | None):
    kwargs: dict[str, Any] = {
        "max_workers": n_pool,
        "initializer": _worker_init,
        "initargs": (init_blob,),
    }
    # max_tasks_per_child recycles a worker periodically, which is how a slow
    # memory leak in someone's job stops being a run-ending problem. Added in
    # 3.11; the package floor is 3.10, so it cannot simply be passed.
    if max_tasks_per_child is not None and sys.version_info >= (3, 11):
        kwargs["max_tasks_per_child"] = max_tasks_per_child
    return ProcessPoolExecutor(**kwargs)


def shutdown_pools() -> None:
    """Tear down every executor held by `reuse_pool=True`.

    Registered with atexit, and safe to call directly - a caller that has
    finished with the pool and wants the processes gone now (before a fork, or
    to release memory held by a broadcast payload) can say so.
    """
    while _pool_cache:
        _, executor = _pool_cache.popitem()
        try:
            executor.shutdown(wait=True)
        except Exception:  # pragma: no cover - best-effort teardown
            pass


atexit.register(shutdown_pools)


def expand_call(kargs: dict[str, Any]) -> Any:
    """Ch.20 Snippet 20.10 (`expandCall`) - turn a job dict into a call.

    Deviation from the book: `kargs` is copied before popping. The book's
    literal `kargs.pop('func')` mutates the *caller's* dict in place, so the
    same job list can only ever be run once - a second pass raises
    KeyError('func'). ex04 deliberately runs one job list through both
    `process_jobs_` and `process_jobs` to compare them, so the copy matters.

    Any argument that is a `BroadcastRef` is resolved here, against the
    payload this process was given by `_worker_init` - see `BroadcastRef`.
    """
    kargs = dict(kargs)
    func = kargs.pop("func")
    for name, value in kargs.items():
        if type(value) is BroadcastRef:
            try:
                kargs[name] = _BROADCAST[value.key]
            except KeyError:
                raise KeyError(
                    f"broadcast value {value.key!r} is not available in this "
                    f"process (pid {os.getpid()}) - it was never installed by "
                    f"the pool initializer"
                ) from None
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


def _run_batch_from_blob(blob: bytes) -> list[tuple[int, float, Any]]:
    """Executor target: run one batch of jobs, returning a triple per job.

    Takes a whole batch rather than a single job, because a submit-per-job
    costs an IPC round-trip and a future each - measured at 6.3x slower than
    joblib on 20,000 trivial jobs, essentially all of it dispatch overhead
    rather than work. Batching amortizes that.

    Returns `(pid, atom_seconds, result)` per job - unchanged in meaning from
    when this ran one job at a time. The pid attributes the completion to a
    specific worker process; `atom_seconds` is measured around each individual
    call, *inside* the worker, so it stays pure compute time even in a batch -
    it cannot include queueing, spawn cost, or time spent on the other jobs
    sharing this batch. That distinction is the whole point: timing the same
    thing from the parent would measure when the batch *arrived*, not how long
    any one job took.
    """
    jobs = cloudpickle.loads(blob)
    pid = os.getpid()
    out: list[tuple[int, float, Any]] = []
    for job in jobs:
        t0 = time.perf_counter()
        result = expand_call(job)
        # Bound before appending, deliberately: inside a tuple literal the
        # elapsed-time expression would be evaluated before the call it is
        # meant to be timing.
        elapsed = time.perf_counter() - t0
        out.append((pid, elapsed, result))
    return out


def _dump_batches(
    batches: list[list[dict[str, Any]]],
    batch_size: int,
    task: str,
    on_job_error: Callable[[int, BaseException], None] | None,
) -> tuple[list[bytes], int]:
    """Serialize each batch to one blob, falling back to per-job attribution
    only for a batch that actually fails.

    The obvious implementation - dump every job separately so an unpicklable
    payload can be named by index - costs a `cloudpickle.dumps` call per job,
    which on 20,000 trivial jobs is most of the parent's time and is paid on
    every run to buy an error path that almost never fires. So take the batch
    in one call optimistically, and only when that raises go back over the
    batch job by job to find which one is at fault. A run with no bad jobs pays
    one dump per batch; a run with one pays double for that batch alone, and
    still gets the exact index.
    """
    blobs: list[bytes] = []
    n_submitted = 0
    for b_i, batch in enumerate(batches):
        try:
            blobs.append(cloudpickle.dumps(batch))
            n_submitted += len(batch)
            continue
        except Exception:
            if on_job_error is None:
                raise

        good: list[dict[str, Any]] = []
        for k, job in enumerate(batch):
            try:
                cloudpickle.dumps(job)
            except Exception as exc:
                _log.warning(
                    "job could not be serialized for dispatch, skipping "
                    "task=%s job_index=%d error=%s", task, b_i * batch_size + k, exc,
                )
                on_job_error(b_i * batch_size + k, exc)
                continue
            good.append(job)
        if good:
            blobs.append(cloudpickle.dumps(good))
            n_submitted += len(good)
    return blobs, n_submitted


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


def _resolve_chunksize(chunksize: int | str, n_submitted: int, n_pool: int) -> int:
    """Jobs per batch. 'auto' mirrors the heuristic `ProcessPoolExecutor.map`
    uses: enough batches to keep every worker fed several times over, so a
    straggler batch cannot leave the pool idle at the end.

    It collapses to 1 for small runs, which is exactly where per-job progress
    granularity matters - a human watching a live display of 8 jobs sees each
    one land. Only large runs batch, and at 20,000 jobs nobody is counting.
    """
    if chunksize == "auto":
        return max(1, n_submitted // (max(1, n_pool) * 4))
    if isinstance(chunksize, bool) or not isinstance(chunksize, int):
        raise ValueError(f"chunksize must be 'auto' or an int - got {chunksize!r}")
    if chunksize < 1:
        raise ValueError(f"chunksize must be >= 1 - got {chunksize}")
    return chunksize


def process_jobs(
    jobs: list[dict[str, Any]],
    task: str | None = None,
    n_workers: int = os.cpu_count() or 4,
    on_progress: Callable[[int, float, Any], None] | None = None,
    on_job_error: Callable[[int, BaseException], None] | None = None,
    text_progress: bool | None = None,
    milestones: bool = True,
    blas_threads: int | str | None = "auto",
    chunksize: int | str = "auto",
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
    broadcast: dict[str, Any] | None = None,
    inject_broadcast: bool = True,
    reuse_pool: bool = False,
    max_tasks_per_child: int | None = None,
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
    closure or a lambda, not just a module-level function. `_run_batch_from_blobs`
    is itself a plain module-level function, so it still pickles by reference
    with no dependence on `__main__` spawn fixup.

    Deliberately built on `concurrent.futures.ProcessPoolExecutor` rather than
    the book's literal `mp.Pool` + `imap_unordered`. `mp.Pool` cannot detect a
    worker that DIES mid-job (OOM-killed, or segfaulting inside native code
    such as BLAS): CPython only resolves a task slot when a result arrives on
    the output queue, and a dead process posts nothing, so the run blocks
    forever with no exception and no log line. `ProcessPoolExecutor` watches
    its workers and raises `BrokenProcessPool` instead - a hang that never
    surfaces is far worse for an unattended run than a loud failure.

    Performance parameters
    ----------------------
    `blas_threads` caps how many threads each worker's native BLAS (OpenBLAS,
    MKL) may use for a single numpy call. Left at `'auto'` it is
    `cpu_count // n_pool`, so eight workers on eight cores get one thread each
    and two workers get four each - parallelism comes from this engine, one job
    per core, rather than from every worker's BLAS separately trying to seize
    the whole machine. Without it, eight workers x eight BLAS threads contend
    for eight cores: measured at 9.5x slower on 24 SVD jobs. Pass an int for an
    explicit budget, or None to leave the environment untouched.

    `chunksize` is how many jobs travel per submission. `'auto'` keeps every
    worker fed several times over while collapsing to 1 on small runs, so a
    live per-job display is unaffected where a human can actually follow it.
    Batching is what closes a 6.3x dispatch-overhead gap on many tiny jobs.

    `broadcast` ships a payload once per WORKER instead of once per job. Any
    value a job would otherwise capture in a closure - a price panel, a fitted
    model - is serialized once and installed in each worker by the pool
    initializer; jobs receive it as a keyword argument.

    It is worth being precise about when this pays, because it is not always.
    Batching already helps a captured payload on its own: cloudpickle memoizes
    a repeated object within one `dumps` call, so a closure-captured panel
    travels once per batch (~32 copies) rather than once per job. Broadcast cuts
    that to one copy per worker, but pays for it during pool startup, where the
    payload is piped to each worker before any job can run. Measured, 200 jobs
    on 8 workers: with an 8 MB panel broadcast is a small *loss* on a cold pool
    (2.25s vs 1.41s), and with an 80 MB panel a 2.5x win (3.52s vs 8.93s).

    Paired with `reuse_pool=True` it stops being a trade-off, because the
    payload is then paid once for the life of the pool rather than once per
    call: the same 80 MB case drops to ~0.55s, 14x the old behaviour and on
    par with joblib's memmapped equivalent (0.55s against 0.58s, though
    joblib's swings with the OS disk cache). If you are broadcasting anything
    substantial and calling more than once, use both.

    Each broadcast key is added to every job as a `BroadcastRef` placeholder,
    which `expand_call` resolves to the real object inside the worker, so the
    caller's function simply receives it as a keyword argument.
    `inject_broadcast=False` suppresses that placement for a caller that
    positions the references itself - `orchestrator.run` does, because it
    wraps the caller's job inside its own bookkeeping job and the reference
    belongs on the inner one, not on the wrapper.

    `initializer`/`initargs` run once per worker before its first job, the
    idiomatic place to open a connection or warm a cache. Unlike the stdlib's,
    this initializer may be a closure or a lambda.

    `reuse_pool=True` keeps the executor alive in a module-level cache, keyed
    by pool size, thread budget and initializer payload, instead of tearing it
    down at the end of the dispatch - worth ~280 ms per call to anything that
    calls this in a loop, and it keeps a broadcast payload resident across
    calls. Off by default: a persistent worker carries state between runs.
    `max_tasks_per_child` (Python 3.11+) recycles a worker periodically, which
    keeps a slow leak in someone's job from ending the run.

    Reporting parameters
    --------------------
    `on_progress`, if given, is called as `on_progress(pid, atom_seconds,
    result)` for every completed job - `pid` identifies which worker process
    produced it and `atom_seconds` is that job's pure compute time as measured
    inside the worker, which is what lets a caller (see `orchestrator.run`'s
    `show_progress`) drive a per-worker live display with real timings. Batched
    jobs are still reported one at a time, so nothing downstream sees a batch.
    The return value here is unaffected either way - still the plain
    `list[Any]` of results, never the `(pid, atom_seconds, result)` triples.

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

    # Place a reference to each broadcast value in every job, so the caller's
    # function receives it as an ordinary keyword argument. Copies rather than
    # mutating - the caller's job dicts are theirs, and ex04 reuses one list
    # across two dispatches.
    if broadcast and inject_broadcast:
        refs = {key: BroadcastRef(key) for key in broadcast}
        jobs = [{**job, **refs} for job in jobs]

    # Never spin up more workers than there are jobs to hand them: the
    # executor starts its processes eagerly, so surplus workers pay full spawn
    # cost to do nothing. This only ever clamps downward - when jobs outnumber
    # n_workers no clamp is needed, since the same fixed pool keeps pulling
    # jobs until the list is done. Worker count is sized to hardware, not to
    # job count.
    out: list[Any] = []
    time0 = time.time()
    batch_size = _resolve_chunksize(chunksize, len(jobs), min(n_workers, len(jobs)))
    blobs, n_submitted = _dump_batches(
        [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)],
        batch_size, task, on_job_error,
    )
    if not blobs:
        _log.warning("nothing dispatchable task=%s n_jobs=%d", task, len(jobs))
        return []

    # Re-clamped against what actually survived serialization, not what was
    # asked for, so jobs dropped by `on_job_error` cannot leave idle workers.
    n_pool = min(n_workers, n_submitted)
    if n_pool < n_workers:
        _log.warning(
            "n_workers=%d clamped to %d task=%s reason=fewer jobs than workers",
            n_workers, n_pool, task,
        )

    # Resolved AFTER the clamp: n_pool is how many jobs actually run at once,
    # which is the only correct divisor for the thread budget.
    budget = _resolve_blas_threads(blas_threads, n_pool)

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
    next_milestone = milestone

    init_blob = cloudpickle.dumps((budget, broadcast or {}, initializer, initargs))
    _log.info(
        "dispatch start task=%s n_jobs=%d n_workers=%d n_batches=%d chunksize=%d "
        "blas_threads=%s reuse_pool=%s text_progress=%s",
        task, n_submitted, n_pool, len(blobs), batch_size,
        budget, reuse_pool, text_progress,
    )

    # A reused pool is only interchangeable with a fresh one if it was built
    # the same way, so the cache key covers everything `_new_executor` consumes.
    # The blob is hashed rather than held: a broadcast payload can be hundreds
    # of megabytes, and the key would otherwise pin a second reference to it.
    cache_key = (
        n_pool,
        max_tasks_per_child,
        hashlib.sha256(init_blob).hexdigest(),
    )

    with _blas_env(budget):
        if reuse_pool:
            executor = _pool_cache.get(cache_key)
            if executor is None:
                executor = _new_executor(n_pool, init_blob, max_tasks_per_child)
                _pool_cache[cache_key] = executor
            owns_executor = False
        else:
            executor = _new_executor(n_pool, init_blob, max_tasks_per_child)
            owns_executor = True

        n_done = 0
        try:
            futures = [executor.submit(_run_batch_from_blob, b) for b in blobs]
            try:
                for future in as_completed(futures):
                    # One batch lands at a time, but everything downstream -
                    # the per-worker display, the per-job logging - is written
                    # against single jobs, so unpack and report each one.
                    for pid, atom_s, out_ in future.result():
                        out.append(out_)
                        n_done += 1
                        if on_progress is not None:
                            on_progress(pid, atom_s, out_)
                    if text_progress:
                        report_progress(n_done, n_submitted, time0, task)
                    elif milestone and (n_done >= next_milestone or n_done == n_submitted):
                        _log.info(
                            "progress task=%s done=%d/%d pct=%.0f elapsed_s=%.2f",
                            task, n_done, n_submitted, 100.0 * n_done / n_submitted,
                            time.time() - time0,
                        )
                        # Recomputed rather than incremented: a batch can carry
                        # completion past several milestones at once.
                        next_milestone = ((n_done // milestone) + 1) * milestone
            except BrokenProcessPool as exc:
                # Cancel whatever has not started so shutdown does not block on
                # work that can never complete, then re-raise with the context the
                # bare stdlib error lacks: how far the run actually got.
                for future in futures:
                    future.cancel()
                # A broken pool is permanently unusable - evict it so the next
                # call builds a fresh one instead of inheriting the corpse.
                if not owns_executor:
                    _pool_cache.pop(cache_key, None)
                    owns_executor = True
                _log.error(
                    "worker process died task=%s completed=%d/%d - run aborted",
                    task, n_done, n_submitted,
                )
                raise BrokenProcessPool(
                    f"a worker process died during task {task!r} after "
                    f"{n_done}/{n_submitted} jobs completed - it was most likely "
                    f"OOM-killed or crashed inside native code (e.g. BLAS). The "
                    f"completed jobs' results were lost with the pool; if you need "
                    f"per-job durability use orchestrator.run, which saves each "
                    f"result to disk as it lands."
                ) from exc
        finally:
            if owns_executor:
                executor.shutdown(wait=True)

    _log.info(
        "dispatch done task=%s completed=%d/%d elapsed_s=%.2f",
        task, len(out), n_submitted, time.time() - time0,
    )
    return out
