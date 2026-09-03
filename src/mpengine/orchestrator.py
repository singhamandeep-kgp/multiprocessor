"""A production orchestration layer on top of `mpengine.engine`.

`engine.py` stays the pure, book-faithful engine (`expand_call`/
`process_jobs_`/`process_jobs`) and is unchanged by this module. `run()` is
the one entry point on top of it: give it a function and a list of parameter
sets, and it handles everything a real multiprocessing job needs beyond the
book's scope -

  1. a manifest .txt recording exactly what was launched (and, once done,
     what happened)
  2. every job's result saved to disk, in whatever format the caller chooses
  3. one log file per *worker process* (not per job), so a failure is
     attributable to a specific process, not just a stack trace

Unlike `engine.py`'s `process_jobs`, a single job failing here does not abort
the run - it's recorded as a failed `JobResult` and the rest continue. That is
the actual point of "know which one broke and which didn't": you want the N-1
good results even when job N blew up.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pickle
import random
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Iterator

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from mpengine.banner import print_banner
from mpengine.engine import (
    BroadcastRef,
    expand_call,
    install_broadcast,
    process_jobs,
    process_jobs_,
)

N_WORKERS = os.cpu_count() or 4

# Run reporting goes through logging rather than print, so embedding mpengine
# in a larger app or a CI job does not pollute stdout. The library adds no
# handler of its own (a library must not configure the root logger); callers
# who want to see this call logging.basicConfig(level=logging.INFO). The
# spider banner is the one deliberate exception and still prints.
_log = logging.getLogger("mpengine.orchestrator")
# See the matching note on "mpengine.engine" in engine.py: lifecycle logging
# is always captured in run.log but never bubbles up to the caller's own
# logging setup, so configuring logging for unrelated reasons doesn't also
# surface mpengine's internal chatter.
_log.propagate = False

# A separate logger, and deliberately NOT a child of "mpengine.orchestrator"
# (it sits directly under "mpengine" instead) - for the "what happened, where
# did it land" summary a person watching a terminal actually wants: the
# stored-here paths and the worker ranking. Kept apart from `_log` for two
# reasons: `_log` also carries dispatch-lifecycle chatter that would show up
# too if a handler were attached to it directly, and `_log.propagate = False`
# above would otherwise also cut this logger off from the caller's own
# logging setup, were it a child of `_log`. These are real `logging` calls
# throughout, never a print() dressed up to look like one -
# `_terminal_summary_handler` below is what makes them visible on an
# interactive terminal without the caller configuring anything, and normal
# propagation is what makes them visible via the caller's own setup too.
_summary_log = logging.getLogger("mpengine.summary")


def _has_external_handler() -> bool:
    """Whether the caller has configured logging somewhere that would already
    show these records - `logging.basicConfig()` (attaches to root) or a
    handler attached directly to `mpengine`. If so, leave it alone entirely
    rather than risk showing anything twice or fighting the caller's own
    formatting."""
    if logging.getLogger().handlers:
        return True
    return any(
        not isinstance(h, logging.NullHandler)
        for h in logging.getLogger("mpengine").handlers
    )


@contextmanager
def _terminal_summary_handler(interactive: bool) -> Iterator[None]:
    """Make `_summary_log` visible on an interactive terminal with zero
    configuration, for the duration of one run.

    Only ever active when `interactive` is True AND the caller has not
    configured logging themselves (`_has_external_handler`) - prod stays
    exactly as silent as the caller's own logging setup dictates, and a
    caller who has taken control of logging is never overridden. Attached and
    removed per run, mirroring `_run_log_file`, so nothing leaks across calls.
    """
    if not interactive or _has_external_handler():
        yield
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
    previous_level = _summary_log.level
    _summary_log.setLevel(logging.INFO)
    _summary_log.addHandler(handler)
    try:
        yield
    finally:
        _summary_log.removeHandler(handler)
        handler.close()
        _summary_log.setLevel(previous_level)

# one FileHandler-backed logger per worker *process*, lazily created on that
# process's first job and reused for every job after - keyed by pid, which is
# always fresh in a newly spawned worker, so no cross-run coordination needed
_worker_logger: logging.Logger | None = None
_worker_log_dir: Path | None = None


@dataclass
class JobResult:
    label: str
    status: str  # "ok" | "error"
    output_path: str | None = None
    error: str | None = None


@dataclass
class RunSummary:
    run_id: str
    manifest_path: str
    output_dir: str
    log_dir: str
    n_jobs: int
    n_ok: int
    n_failed: int
    elapsed_s: float
    results: list[JobResult] = field(default_factory=list)
    # keyed by worker pid; only populated when show_progress=True, since that
    # is what installs the per-atom timing hook
    worker_stats: dict[int, "WorkerStats"] = field(default_factory=dict)


def save_pickle(obj: Any, path: Path) -> None:
    """Default save_fn. A caller-supplied save_fn can be a lambda or a closure
    too - like any other job field, it crosses the process boundary via
    `cloudpickle`, not stdlib `pickle`."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """Counterpart to `save_pickle`, and the default `load_fn` for
    `load_run_outputs`. Runs in the calling process, not a worker, so the
    ordinary `pickle` caveat applies: whatever types were saved must be
    importable here to be reconstructed."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_run_outputs(
    run_dir: str | Path,
    load_fn: Callable[[Path], Any] = load_pickle,
) -> dict[str, Any]:
    """Read a finished run's outputs back as `{label: result}`.

    `run_dir` is that run's output directory - pass `RunSummary.output_dir`,
    or the path `run()` printed as "Output stored here". Keys are the labels
    `run()` assigned (`job_0000`, ...), so a result can be matched straight
    back to the `JobResult` that produced it.

    Only successful jobs appear. A failed job never wrote a file, so its label
    is simply missing here - check `RunSummary.results` to see which failed and
    why, rather than inferring it from a gap in these keys.

    `load_fn` must match whatever `save_fn` wrote the run; `save_pickle` and
    `load_pickle` are the matched default pair.
    """
    path = Path(run_dir)
    if not path.is_dir():
        raise NotADirectoryError(f"not a run output directory: {path}")
    # Skip `.<label>.partial` leftovers: those are in-flight or abandoned
    # writes from `_run_and_save_job`, never a finished result.
    return {
        p.name: load_fn(p)
        for p in sorted(path.iterdir())
        if p.is_file() and not (p.name.startswith(".") and p.name.endswith(".partial"))
    }


@contextmanager
def _run_log_file(log_path: Path) -> Iterator[None]:
    """Capture the parent's whole view of one run into `<log_dir>/run.log`.

    The per-worker files record what each worker did; this records what the
    run as a whole did - dispatch, milestones, per-job outcomes, the summary -
    so one run is one self-contained, durable record. That is the audit trail
    this library is actually differentiated on.

    Attached directly to THREE loggers, not just the shared "mpengine" parent:
    "mpengine.orchestrator" and "mpengine.engine" both set `propagate = False`
    (so their lifecycle chatter never reaches a caller's own logging setup -
    see the note by each `_log` definition), which means a handler placed
    only on "mpengine" would never see their records at all. Attaching
    directly to each stops there being any gap; "mpengine.summary" (the
    stored-here paths and ranking) still climbs normally, so attaching to
    "mpengine" as well is what captures those. Level handling is deliberately
    restrained: forcing DEBUG on any of these would also push DEBUG records
    into a caller's own handlers wherever they DO propagate (a per-job line
    for all 10,000 jobs appearing in someone's console who asked for INFO) -
    instead each logger's level is only ever lowered as far as INFO, and only
    when it was coarser than that, so the file always captures the full
    lifecycle and every failure without ever making the caller's own output
    noisier than they configured. A caller who genuinely wants per-job DEBUG
    detail in the file just sets DEBUG themselves. Everything is restored on
    the way out, including if the run raises.
    """
    loggers = [
        logging.getLogger("mpengine"),
        logging.getLogger("mpengine.orchestrator"),
        logging.getLogger("mpengine.engine"),
    ]
    handler = logging.FileHandler(log_path / "run.log", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    previous_levels = [lg.level for lg in loggers]
    for lg in loggers:
        if not lg.isEnabledFor(logging.INFO):
            lg.setLevel(logging.INFO)
        lg.addHandler(handler)
    try:
        yield
    finally:
        for lg, level in zip(loggers, previous_levels):
            lg.removeHandler(handler)
            lg.setLevel(level)
        handler.close()


def _validate_labels(labels: list[str]) -> None:
    """Reject labels that are unsafe or ambiguous as filenames.

    Each label is used directly as a filename (`output_dir / label`), and
    pathlib's `/` is unforgiving about what that permits: an absolute label
    discards the base directory entirely, `..` walks out of the run tree, and
    an empty label collapses onto the run directory itself so `save_fn` is
    handed a directory to write. None of that is reachable from hostile input
    - a caller only ever passes its own labels - but all of it silently writes
    somewhere other than the run directory, which is worth a loud error rather
    than a debugging session.

    Duplicates are rejected for a different reason: two jobs sharing a label
    write to one path, both report `ok` with that same `output_path`, and
    whichever finishes last silently wins. The other result is simply gone.
    """
    bad: list[str] = []
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            bad.append(f"{label!r} (empty)")
        elif Path(label).is_absolute() or (len(label) > 1 and label[1] == ":"):
            bad.append(f"{label!r} (absolute path)")
        elif "/" in label or "\\" in label:
            bad.append(f"{label!r} (contains a path separator)")
        elif ".." in Path(label).parts:
            bad.append(f"{label!r} (contains '..')")
    if bad:
        raise ValueError(
            "labels are used directly as output filenames, so these are not usable: "
            + ", ".join(bad)
        )

    seen: dict[str, int] = {}
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
    dupes = sorted(lbl for lbl, n in seen.items() if n > 1)
    if dupes:
        raise ValueError(
            "labels must be unique - each one names an output file, so duplicates "
            f"would overwrite each other: {', '.join(repr(d) for d in dupes)}"
        )


def _validate_broadcast(
    broadcast: dict[str, Any], param_sets: list[dict[str, Any]]
) -> None:
    """Reject a broadcast payload that cannot be delivered unambiguously.

    A broadcast value reaches the caller's function as a keyword argument, so
    its key has to be a usable one, and it must not collide with a key that
    `param_sets` already supplies - a job carrying both would be two competing
    values for one parameter, and silently picking either is worse than saying
    so. Checked here, before any directory is created, per the same
    validation-precedes-side-effects rule as `_validate_labels`.
    """
    bad = [k for k in broadcast if not isinstance(k, str) or not k.isidentifier()]
    if bad:
        raise ValueError(
            "broadcast keys are passed to your function as keyword arguments, so "
            "each must be a valid Python identifier - these are not: "
            + ", ".join(repr(k) for k in bad)
        )
    clashing = sorted({k for params in param_sets for k in params} & set(broadcast))
    if clashing:
        raise ValueError(
            "these keys are in both broadcast and param_sets, so each job would "
            f"get two values for one argument: {', '.join(repr(k) for k in clashing)}"
        )


def _get_worker_logger(log_dir: Path) -> logging.Logger:
    global _worker_logger, _worker_log_dir
    if _worker_logger is not None and _worker_log_dir == log_dir:
        return _worker_logger

    pid = os.getpid()
    logger = logging.getLogger(f"mpengine.orchestrator.worker.{pid}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(log_dir / f"worker_{pid}.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)

    _worker_logger, _worker_log_dir = logger, log_dir
    return logger


def _run_and_save_job(job: dict) -> JobResult:
    """The actual worker/sequential target. Unwraps its own bookkeeping fields,
    calls the caller's function via `expand_call` (reusing engine.py rather
    than reimplementing the dict-to-call trick), saves the result, and always
    returns a JobResult - exceptions are caught here, never re-raised, so one
    bad job cannot abort the run.
    """
    label = job["label"]
    output_dir = job["output_dir"]
    log_dir = job["log_dir"]
    save_fn = job["save_fn"]
    inner_job = job["inner_job"]

    logger = _get_worker_logger(log_dir)
    logger.info("starting %s", label)
    try:
        result = expand_call(inner_job)
    except Exception as exc:
        logger.error("failed %s: %s\n%s", label, exc, traceback.format_exc())
        return JobResult(label=label, status="error", error=str(exc))

    # Write to a sibling temp path and only then rename into place. A save_fn
    # that dies partway through (plausible for a large or custom serializer)
    # would otherwise leave partial bytes at the real output path while the job
    # is correctly recorded as failed - and a later `load_run_outputs`, which
    # loads every file it finds, would then choke on that corpse and take down
    # the read-back of an otherwise healthy run. os.replace is atomic on both
    # POSIX and Windows, so the final path only ever holds a complete result.
    output_path = output_dir / label
    tmp_path = output_dir / f".{label}.partial"
    try:
        save_fn(result, tmp_path)
        os.replace(tmp_path, output_path)
    except Exception as exc:
        logger.error("save failed %s: %s\n%s", label, exc, traceback.format_exc())
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return JobResult(label=label, status="error", error=f"save failed: {exc}")

    logger.info("completed %s -> %s", label, output_path)
    return JobResult(label=label, status="ok", output_path=str(output_path))


def _load_worker_names() -> list[str]:
    """Cool, memorable codenames for the progress display - purely cosmetic,
    so a worker reads as "Worker Jarvis (PID 12345)" rather than a bare pid.
    Lives in `worker_names.json` (not inline here) so the name pool can be
    edited without touching this module. Sized generously past any realistic
    core count; `_next_worker_name` falls back to a numbered name if a run
    somehow has more workers than names.
    """
    data = resources.files("mpengine").joinpath("worker_names.json").read_text(encoding="utf-8")
    return json.loads(data)


_WORKER_NAMES = _load_worker_names()


@dataclass
class WorkerStats:
    """Per-worker-process tallies, accumulated live as atoms complete."""

    name: str
    pid: int
    n_atoms: int = 0
    busy_s: float = 0.0
    last_atom_s: float = 0.0

    @property
    def avg_atom_s(self) -> float:
        return self.busy_s / self.n_atoms if self.n_atoms else 0.0

    @property
    def atoms_per_s(self) -> float:
        return self.n_atoms / self.busy_s if self.busy_s else 0.0

    def line(self) -> str:
        plural = "atom " if self.n_atoms == 1 else "atoms"
        return (
            f"Worker {self.name} (PID {self.pid}): {self.n_atoms} {plural} | "
            f"avg {self.avg_atom_s:.2f}s/atom | last {self.last_atom_s:.2f}s"
        )


@contextmanager
def _progress_renderer(
    n_jobs: int, task: str, stats_out: dict[int, WorkerStats]
) -> Iterator[Callable[[int, float, Any], None]]:
    """tqdm-based live display for `run(show_progress=True)`: one overall bar
    (total=n_jobs), plus a per-worker line created lazily the first time a pid
    is seen. Workers get no fixed total - `Pool` assigns jobs to them
    dynamically as they free up, so there's no way to know one in advance.

    The per-worker numbers are computed here from `atom_seconds` measured
    inside the worker, *not* from tqdm's own rate. tqdm's rate would divide by
    the bar's lifetime - creation to close - and since every bar is closed
    together at the end of the run, that span is mostly idle waiting, which
    makes an early-finishing worker look slow and the last-finishing worker
    look fast. Dividing real atom counts by real compute time avoids that
    inversion entirely.

    Every line is recomputed and redrawn the instant its worker finishes an
    atom (`mininterval=0` so tqdm cannot throttle the redraw). A worker's line
    holds steady while it is mid-atom - there is no partial-progress signal
    from inside a running atom, only completions.

    Each worker gets a random, unique codename for the run (shuffled once per
    call, so it varies run to run); the pid is shown alongside it, since the
    name itself is only decorative. `stats_out` is populated in place so the
    caller keeps the tallies after the display is torn down.
    """
    overall = tqdm(total=n_jobs, desc=task, position=0, file=sys.stderr, mininterval=0)
    worker_bars: dict[int, tqdm] = {}
    available_names = _WORKER_NAMES.copy()
    # A private Random instance, not `random.shuffle`. Shuffling via the module
    # function advances the interpreter-global RNG, so picking cosmetic worker
    # codenames would perturb any caller relying on `random` for a reproducible
    # sequence - a real hazard for the Monte Carlo work this engine is aimed at.
    random.Random().shuffle(available_names)

    def _next_worker_name() -> str:
        if available_names:
            return available_names.pop()
        return f"Worker-{len(worker_bars) + 1}"

    def on_progress(pid: int, atom_s: float, _result: Any) -> None:
        if pid not in worker_bars:
            stats_out[pid] = WorkerStats(name=_next_worker_name(), pid=pid)
            worker_bars[pid] = tqdm(
                total=None,
                position=len(worker_bars) + 1,
                bar_format="{desc}",
                file=sys.stderr,
                mininterval=0,
            )

        stats = stats_out[pid]
        stats.n_atoms += 1
        stats.busy_s += atom_s
        stats.last_atom_s = atom_s

        bar = worker_bars[pid]
        bar.set_description_str(stats.line(), refresh=True)
        overall.update(1)

    try:
        yield on_progress
    finally:
        overall.close()
        for bar in worker_bars.values():
            bar.close()


def _log_worker_ranking(stats: dict[int, WorkerStats]) -> None:
    """Rank workers fastest-to-slowest by seconds per atom.

    A real `_summary_log.info(...)` call, nothing else - visibility on an
    interactive terminal comes entirely from `_terminal_summary_handler`
    attaching a real handler for the run's duration, not from a second print
    alongside the log call. This can only ever produce output when `stats` is
    non-empty, which only happens in the first place when the terminal is
    real (see `run()` - `stats` is populated solely by the live-display
    renderer).

    Note the caveat logged alongside: with uneven atom sizes - the normal
    case in quant work - a low s/atom can mean small atoms rather than a
    genuinely faster worker, so the fastest/slowest tags are a hint, not a
    measurement.
    """
    if not stats:
        return

    ranked = sorted(stats.values(), key=lambda s: s.avg_atom_s)
    lines = ["workers, fastest to slowest (by avg seconds per atom):"]
    for i, s in enumerate(ranked):
        tag = ""
        if len(ranked) > 1:
            tag = "  <- fastest" if i == 0 else ("  <- slowest" if i == len(ranked) - 1 else "")
        lines.append(
            f"  {s.name:<12} (PID {s.pid:>6})  {s.n_atoms:>3} atoms  "
            f"avg {s.avg_atom_s:6.2f}s/atom  {s.atoms_per_s:6.2f} atoms/s{tag}"
        )
    lines.append("  (uneven atom sizes: a low s/atom can mean small atoms, not a faster worker)")
    _summary_log.info("\n".join(lines))


def run(
    func: Callable[..., Any],
    param_sets: list[dict[str, Any]],
    *,
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    manifest_dir: str | Path | None = None,
    save_fn: Callable[[Any, Path], None] = save_pickle,
    labels: list[str] | None = None,
    task: str | None = None,
    n_workers: int = N_WORKERS,
    debug: bool = False,
    show_progress: bool = False,
    text_progress: bool | None = None,
    blas_threads: int | str | None = "auto",
    chunksize: int | str = "auto",
    broadcast: dict[str, Any] | None = None,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
    reuse_pool: bool = False,
    max_tasks_per_child: int | None = None,
) -> RunSummary:
    """Run `func(**params)` for every dict in `param_sets`, organized.

    Destinations: pass `base_dir` and the three output locations are derived
    as `<base_dir>/outputs`, `<base_dir>/logs` and `<base_dir>/manifests`; or
    pass `output_dir`/`log_dir`/`manifest_dir` explicitly to place them
    independently. An explicit path always wins over the derived one, so you
    can give a `base_dir` and still redirect just the logs elsewhere.

    `func` can be a closure or a lambda, not just a module-level function -
    `engine.process_jobs` serializes jobs with `cloudpickle`, which can pickle
    a function by value. The only cost: whatever a closure captures travels
    with every job that uses it.

    Every destination directory gets a `<run_id>` subfolder (`task` + a
    timestamp), so successive runs never collide and a worker's log file
    (named by its pid) can never be confused with a previous run's.

    `show_progress=True` (only meaningful when `debug=False` - ignored
    otherwise, since debug mode is one sequential in-process run with no
    "workers" to distinguish) renders a live terminal display: one overall bar
    for the whole run, plus a per-worker line showing atoms done, average
    seconds per atom and the last atom's time, each recomputed and redrawn the
    moment that worker finishes an atom. It also prints a fastest-to-slowest
    ranking at the end, and fills `RunSummary.worker_stats` (keyed by pid) so
    the same numbers are available programmatically.

    `text_progress` overrides whether the book's `\r`-overwriting stderr line
    runs (only relevant when `show_progress=False`; the default, `None`,
    keeps today's behaviour - on when interactive, off otherwise). Force it
    to `False` if you're watching per-job DEBUG/INFO logging instead: the two
    would otherwise both write to the terminal and visually clash.

    Performance
    -----------
    `broadcast` is the one to reach for first when jobs share a large object.
    Anything a job would otherwise capture in a closure - a price panel, a
    fitted model - goes here instead, is serialized once, and is delivered to
    each *worker* rather than travelling with each *job*::

        run(score, param_sets, base_dir="runs", broadcast={"panel": panel})

    `score` is then called as `score(**params, panel=panel)`. A key here may
    not also appear in `param_sets`. Pair it with `reuse_pool=True` whenever
    you call `run()` more than once with the same payload - the panel is then
    delivered once for the life of the pool instead of once per call, which
    for a 80 MB panel over 200 jobs is 0.62s against 8.93s. On a single cold
    call with a payload under ~10 MB, plain closure capture is if anything
    slightly faster; `engine.process_jobs` documents the crossover in full.

    `blas_threads` caps the threads each worker's native BLAS may use for one
    numpy call; `'auto'` is `cpu_count // workers`, so parallelism comes from
    this engine rather than from every worker's BLAS separately trying to
    seize the whole machine (measured 9.5x on SVD-heavy jobs). `chunksize`
    is how many jobs travel per submission, `'auto'` by default and collapsing
    to 1 on small runs so a live display keeps per-job granularity.
    `initializer`/`initargs` run once per worker before its first job and may
    be closures. `reuse_pool=True` keeps the pool (and its broadcast payload)
    alive between calls, worth ~280 ms per call in a loop; `max_tasks_per_child`
    recycles a worker periodically on Python 3.11+. All are passed straight
    through to `engine.process_jobs`, which documents them in full.
    """
    if base_dir is not None:
        base = Path(base_dir)
        output_dir = output_dir if output_dir is not None else base / "outputs"
        log_dir = log_dir if log_dir is not None else base / "logs"
        manifest_dir = manifest_dir if manifest_dir is not None else base / "manifests"

    missing = [
        name
        for name, value in (
            ("output_dir", output_dir),
            ("log_dir", log_dir),
            ("manifest_dir", manifest_dir),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"pass base_dir, or all of output_dir/log_dir/manifest_dir (missing: {', '.join(missing)})"
        )

    # `func.__name__` is not safe: a functools.partial or a callable object -
    # both of which this engine explicitly supports - has no __name__.
    func_name = getattr(func, "__name__", None) or type(func).__name__ or "job"
    task = task or func_name

    # Everything above and below this point is validation; no directory,
    # manifest or banner is produced until it all passes, so a rejected call
    # cannot leave orphaned run artefacts behind.
    labels = labels or [f"job_{i:04d}" for i in range(len(param_sets))]
    if len(labels) != len(param_sets):
        raise ValueError(f"got {len(labels)} labels for {len(param_sets)} param sets")
    _validate_labels(labels)
    if broadcast:
        _validate_broadcast(broadcast, param_sets)

    # Microseconds, not seconds. At second resolution two runs of the same task
    # inside one second produced an identical run_id, and because the
    # directories are made with exist_ok=True they were silently *shared*: the
    # second run's manifest, opened 'w', truncated the first run's manifest
    # outright, and default job_NNNN labels overwrote its outputs.
    run_id = f"{task}_{dt.datetime.now():%Y%m%d_%H%M%S_%f}"

    output_path = Path(output_dir) / run_id
    log_path = Path(log_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    Path(manifest_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_dir) / f"{run_id}.txt"

    # Broadcast values go into the INNER job as lightweight references, never
    # as the objects themselves - that is the whole point, since the inner job
    # is what gets cloudpickled per job. `engine.expand_call` swaps each
    # reference for the real payload inside the worker, where the pool
    # initializer has already installed it exactly once.
    refs = {key: BroadcastRef(key) for key in (broadcast or {})}
    jobs = [
        {
            "func": _run_and_save_job,
            "job": {
                "label": label,
                "output_dir": output_path,
                "log_dir": log_path,
                "save_fn": save_fn,
                "inner_job": {"func": func, **params, **refs},
            },
        }
        for label, params in zip(labels, param_sets)
    ]

    with open(manifest_path, "w") as f:
        f.write(f"run_id: {run_id}\n")
        f.write(f"launched: {dt.datetime.now().isoformat(sep=' ', timespec='seconds')}\n")
        f.write(f"func: {func_name}\n")
        f.write(f"task: {task}\n")
        f.write(f"n_workers: {n_workers}\n")
        f.write(f"debug: {debug}\n")
        f.write(f"n_jobs: {len(jobs)}\n")
        f.write(f"output_dir: {output_path}\n")
        f.write(f"log_dir: {log_path}\n")
        f.write("\nparameters:\n")
        for label, params in zip(labels, param_sets):
            f.write(f"  {label}: {params}\n")
        f.write("\n")

    # The terminal/prod split lives here: a real terminal gets the spider, the
    # live progress display and the worker summary logs made visible with zero
    # configuration, since those are things worth seeing live. Redirected -
    # piped, under a supervisor, in CI - none of that shows; only `logging`
    # output exists, for whatever the deployment itself configures to collect it.
    interactive = sys.stderr.isatty()

    # One durable record per run: everything from here down - dispatch,
    # milestones, every job outcome, the summary - is captured to
    # <log_dir>/run.log as well as going to the caller's own handlers.
    # `_terminal_summary_handler` is what makes the stored-here paths and the
    # worker ranking (both real `_summary_log` calls, see above) actually
    # visible on an interactive terminal without any caller configuration.
    # Order matters: _terminal_summary_handler must check for an externally
    # configured handler BEFORE _run_log_file attaches its own FileHandler to
    # "mpengine" - reversed, it would mistake our own run.log handler for a
    # caller-configured one and skip attaching the terminal handler entirely.
    with _terminal_summary_handler(interactive), _run_log_file(log_path):
        worker_stats: dict[int, WorkerStats] = {}

        if interactive:
            print_banner(task, n_workers, debug)
        _log.info(
            "run start run_id=%s task=%s func=%s n_jobs=%d n_workers=%d "
            "debug=%s show_progress=%s",
            run_id, task, func_name, len(jobs), n_workers, debug, show_progress,
        )
        _log.info(
            "run dirs run_id=%s output_dir=%s log_dir=%s manifest=%s",
            run_id, output_path, log_path, manifest_path,
        )

        # A job that cannot even be cloudpickled for submission becomes one failed
        # JobResult rather than sinking the whole batch - the same per-job
        # isolation this layer already promises for jobs that fail while running.
        unsendable: list[JobResult] = []

        def on_job_error(index: int, exc: BaseException) -> None:
            unsendable.append(
                JobResult(
                    label=labels[index],
                    status="error",
                    error=f"could not be serialized for dispatch: {exc}",
                )
            )

        def log_job(pid: int, atom_s: float, result: Any) -> None:
            """Surface every job outcome in the PARENT log as it lands.

            Successes are DEBUG - a 10,000-job sweep should not write 10,000 INFO
            lines - but failures are WARNING, because until now a failed job was
            only ever written to that worker's own file and an operator watching
            the run's logs saw nothing at all until the manifest was read.
            """
            label = getattr(result, "label", "?")
            if getattr(result, "status", None) == "ok":
                _log.debug("job ok label=%s pid=%s atom_s=%.3f", label, pid, atom_s)
            else:
                _log.warning(
                    "job FAILED label=%s pid=%s atom_s=%.3f error=%s",
                    label, pid, atom_s, getattr(result, "error", None),
                )

        # The live display is a terminal affordance. Interactively it is the whole
        # point; redirected to a log file tqdm's redraws are unreadable noise, so
        # in prod we fall back to the engine's periodic INFO milestones instead.
        # (`interactive` itself was already computed above, before the banner.)
        live_display = show_progress and not debug and interactive
        if show_progress and not debug and not interactive:
            _log.info(
                "progress display disabled run_id=%s reason=stderr is not a terminal "
                "(logging milestones instead)", run_id,
            )

        # Everything the engine needs to build and feed the pool. Collected
        # once so the three dispatch branches below cannot drift apart.
        dispatch_kwargs: dict[str, Any] = {
            "task": task,
            "n_workers": n_workers,
            "on_job_error": on_job_error,
            "blas_threads": blas_threads,
            "chunksize": chunksize,
            "broadcast": broadcast,
            # The references were placed on the INNER job above, where the
            # caller's own function is - letting the engine add them to the
            # outer job as well would hand `_run_and_save_job` a keyword
            # argument it has no parameter for.
            "inject_broadcast": False,
            "initializer": initializer,
            "initargs": initargs,
            "reuse_pool": reuse_pool,
            "max_tasks_per_child": max_tasks_per_child,
        }

        t0 = time.perf_counter()
        if debug:
            # No pool means no pool initializer, so the two things it would
            # have done once per worker have to happen here instead - this
            # process IS the worker in debug mode, and without the payload
            # installed the BroadcastRefs in each job resolve against nothing.
            if broadcast:
                install_broadcast(broadcast)
            if initializer is not None:
                initializer(*initargs)
            raw_results = process_jobs_(jobs)
            for r in raw_results:
                log_job(os.getpid(), 0.0, r)
        elif live_display:
            # logging_redirect_tqdm is tqdm's own answer to exactly this
            # clash: for its duration, any logging call that would otherwise
            # write straight to the terminal (ours - log_job's per-job
            # DEBUG/WARNING - or a caller's own configured handler) is instead
            # routed through tqdm.write(), which clears the active bars,
            # prints the line cleanly above them, and redraws them - instead
            # of corrupting their redraw or spawning duplicate bar lines.
            with _progress_renderer(len(jobs), task, worker_stats) as render, \
                 logging_redirect_tqdm():
                def on_progress(pid: int, atom_s: float, result: Any) -> None:
                    log_job(pid, atom_s, result)
                    render(pid, atom_s, result)

                raw_results = process_jobs(
                    jobs,
                    on_progress=on_progress,
                    text_progress=False,
                    # The tqdm bar above already shows aggregate progress -
                    # the engine's own milestone logging would just repeat it.
                    milestones=False,
                    **dispatch_kwargs,
                )
        else:
            # `text_progress=None` keeps the existing default (the book's
            # `\r` line whenever interactive and show_progress=False);
            # passing it explicitly - e.g. to force it off so per-job DEBUG
            # logging isn't fighting a `\r`-rewriting line on the same
            # stream - always wins.
            effective_text_progress = (
                text_progress if text_progress is not None
                else (not show_progress) and interactive
            )
            raw_results = process_jobs(
                jobs,
                on_progress=log_job,
                text_progress=effective_text_progress,
                **dispatch_kwargs,
            )
        elapsed_s = time.perf_counter() - t0

        if worker_stats:
            _log_worker_ranking(worker_stats)

        # Restore submission order. `process_jobs` yields in completion order (by
        # design - that is what makes its progress reporting honest), but these
        # results are label-addressed, so a caller zipping them against the
        # param_sets it passed in would silently mismatch. Jobs that never made it
        # off the ground are folded back into their original positions too.
        raw_results = raw_results + unsendable
        label_order = {label: i for i, label in enumerate(labels)}
        raw_results.sort(key=lambda r: label_order.get(r.label, len(label_order)))

        n_ok = sum(1 for r in raw_results if r.status == "ok")
        n_failed = len(raw_results) - n_ok

        with open(manifest_path, "a") as f:
            f.write("results:\n")
            for r in raw_results:
                if r.status == "ok":
                    f.write(f"  {r.label}: ok -> {r.output_path}\n")
                else:
                    f.write(f"  {r.label}: ERROR - {r.error}\n")
            f.write(f"\nn_ok: {n_ok}\n")
            f.write(f"n_failed: {n_failed}\n")
            f.write(f"elapsed_s: {elapsed_s:.3f}\n")

        _log.info(
            "run done run_id=%s task=%s n_ok=%d n_failed=%d elapsed_s=%.2f "
            "throughput_jobs_s=%.2f",
            run_id, task, n_ok, n_failed, elapsed_s,
            (len(raw_results) / elapsed_s) if elapsed_s > 0 else 0.0,
        )
        # Real _summary_log.info(...) calls - reaches run.log and any handler
        # a prod deployment configures either way. On an interactive terminal
        # with nothing configured, `_terminal_summary_handler` (entered above)
        # is what makes them actually appear on screen - there is no separate
        # print() standing in for a log line here.
        for label, value in (
            ("Logs stored here    ", log_path),
            ("Output stored here  ", output_path),
            ("Manifest stored here", manifest_path),
        ):
            _summary_log.info("%s - %s", label, value)

        return RunSummary(
            run_id=run_id,
            manifest_path=str(manifest_path),
            output_dir=str(output_path),
            log_dir=str(log_path),
            n_jobs=len(jobs),
            n_ok=n_ok,
            n_failed=n_failed,
            elapsed_s=elapsed_s,
            results=raw_results,
            worker_stats=worker_stats,
        )
