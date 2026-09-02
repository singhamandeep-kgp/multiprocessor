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

from mpengine.banner import print_banner
from mpengine.engine import expand_call, process_jobs, process_jobs_

N_WORKERS = os.cpu_count() or 4

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
    return {p.name: load_fn(p) for p in sorted(path.iterdir()) if p.is_file()}


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
    """The actual Pool/sequential target. Unwraps its own bookkeeping fields,
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

    output_path = output_dir / label
    try:
        save_fn(result, output_path)
    except Exception as exc:
        logger.error("save failed %s: %s\n%s", label, exc, traceback.format_exc())
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
    random.shuffle(available_names)

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


def _print_worker_ranking(stats: dict[int, WorkerStats]) -> None:
    """Rank workers fastest-to-slowest by seconds per atom.

    Note the caveat printed alongside: with uneven atom sizes, a low
    s/atom can mean small atoms rather than a genuinely faster worker.
    """
    if not stats:
        return

    ranked = sorted(stats.values(), key=lambda s: s.avg_atom_s)
    print("\nworkers, fastest to slowest (by avg seconds per atom):")
    for i, s in enumerate(ranked):
        tag = ""
        if len(ranked) > 1:
            tag = "  <- fastest" if i == 0 else ("  <- slowest" if i == len(ranked) - 1 else "")
        print(
            f"  {s.name:<12} (PID {s.pid:>6})  {s.n_atoms:>3} atoms  "
            f"avg {s.avg_atom_s:6.2f}s/atom  {s.atoms_per_s:6.2f} atoms/s{tag}"
        )
    print("  (uneven atom sizes: a low s/atom can mean small atoms, not a faster worker)")


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

    task = task or func.__name__
    run_id = f"{task}_{dt.datetime.now():%Y%m%d_%H%M%S}"

    output_path = Path(output_dir) / run_id
    log_path = Path(log_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    Path(manifest_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_dir) / f"{run_id}.txt"

    labels = labels or [f"job_{i:04d}" for i in range(len(param_sets))]
    if len(labels) != len(param_sets):
        raise ValueError(f"got {len(labels)} labels for {len(param_sets)} param sets")

    jobs = [
        {
            "func": _run_and_save_job,
            "job": {
                "label": label,
                "output_dir": output_path,
                "log_dir": log_path,
                "save_fn": save_fn,
                "inner_job": {"func": func, **params},
            },
        }
        for label, params in zip(labels, param_sets)
    ]

    with open(manifest_path, "w") as f:
        f.write(f"run_id: {run_id}\n")
        f.write(f"launched: {dt.datetime.now().isoformat(sep=' ', timespec='seconds')}\n")
        f.write(f"func: {func.__name__}\n")
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

    worker_stats: dict[int, WorkerStats] = {}

    print_banner(task, n_workers, debug)

    t0 = time.perf_counter()
    if debug:
        raw_results = process_jobs_(jobs)
    elif show_progress:
        with _progress_renderer(len(jobs), task, worker_stats) as on_progress:
            raw_results = process_jobs(jobs, task=task, n_workers=n_workers, on_progress=on_progress)
    else:
        raw_results = process_jobs(jobs, task=task, n_workers=n_workers)
    elapsed_s = time.perf_counter() - t0

    if show_progress and not debug:
        _print_worker_ranking(worker_stats)

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

    print(f"Logs stored here     - {log_path}")
    print(f"Output stored here   - {output_path}")
    print(f"Manifest stored here - {manifest_path}")

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
