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
import logging
import os
import pickle
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mpengine.engine import expand_call, process_jobs, process_jobs_

N_WORKERS = min(os.cpu_count() or 4, 8)

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
    n_jobs: int
    n_ok: int
    n_failed: int
    elapsed_s: float
    results: list[JobResult] = field(default_factory=list)


def save_pickle(obj: Any, path: Path) -> None:
    """Default save_fn. A caller-supplied save_fn must be module-level
    (importable by reference) - like any other Pool target, it crosses the
    Windows spawn boundary and a lambda or closure will fail to pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


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
    n_threads: int = N_WORKERS,
    debug: bool = False,
) -> RunSummary:
    """Run `func(**params)` for every dict in `param_sets`, organized.

    Destinations: pass `base_dir` and the three output locations are derived
    as `<base_dir>/outputs`, `<base_dir>/logs` and `<base_dir>/manifests`; or
    pass `output_dir`/`log_dir`/`manifest_dir` explicitly to place them
    independently. An explicit path always wins over the derived one, so you
    can give a `base_dir` and still redirect just the logs elsewhere.

    `func` must be module-level/importable (the same Windows-spawn constraint
    `engine.py` already establishes) - a lambda or a function defined inside
    another function will fail to pickle when `debug=False`.

    Every destination directory gets a `<run_id>` subfolder (`task` + a
    timestamp), so successive runs never collide and a worker's log file
    (named by its pid) can never be confused with a previous run's.
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
        f.write(f"n_threads: {n_threads}\n")
        f.write(f"debug: {debug}\n")
        f.write(f"n_jobs: {len(jobs)}\n")
        f.write(f"output_dir: {output_path}\n")
        f.write(f"log_dir: {log_path}\n")
        f.write("\nparameters:\n")
        for label, params in zip(labels, param_sets):
            f.write(f"  {label}: {params}\n")
        f.write("\n")

    t0 = time.perf_counter()
    raw_results = process_jobs_(jobs) if debug else process_jobs(jobs, task=task, n_threads=n_threads)
    elapsed_s = time.perf_counter() - t0

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

    return RunSummary(
        run_id=run_id,
        manifest_path=str(manifest_path),
        n_jobs=len(jobs),
        n_ok=n_ok,
        n_failed=n_failed,
        elapsed_s=elapsed_s,
        results=raw_results,
    )
