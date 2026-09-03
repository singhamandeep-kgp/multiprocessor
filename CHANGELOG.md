# Changelog

All notable changes to `mpengine` are documented here.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the
version is below 1.0, breaking changes may land in a minor release — each one
is called out under **Breaking** with the migration needed.

## [0.3.0] — 2026-09-03

A correctness and observability release. Ten defects found in an audit of the
library were fixed, and all run reporting moved to the `logging` module.

### Breaking

- **Run reporting no longer prints to stdout.** The "stored here" paths and the
  worker ranking now go through `logging`. A library writing unconditionally to
  stdout breaks piping, notebooks and log aggregation for anything embedding it.
  *Migration:* call `logging.basicConfig(level=logging.INFO)` to see them again.
  The ASCII banner still prints, deliberately.
- **Labels are now validated.** A label naming an output file may no longer
  contain a path separator or `..`, be absolute, be empty, or be duplicated —
  all of these previously wrote somewhere other than the run directory, or
  silently overwrote another job's result. Such calls now raise `ValueError`.
  *Migration:* sanitise labels before passing them; `job_0000`-style names are
  unaffected.
- **`run_id` format gained microseconds** (`task_YYYYmmdd_HHMMSS_ffffff`).
  *Migration:* anything parsing `run_id` needs updating.
- **`run()` returns results in submission order**, not completion order, since
  they are label-addressed. *Migration:* if you relied on completion ordering,
  sort by your own timing data instead.
- **The live progress display is suppressed when stderr is not a terminal.**
  Piped or redirected runs get periodic INFO milestones instead of tqdm redraws.
  *Migration:* none for interactive use; pass `text_progress=True` to
  `process_jobs` to force the text line.

### Fixed

- **A dead worker no longer hangs the run forever.** Dispatch moved from
  `mp.Pool` + `imap_unordered` to `ProcessPoolExecutor` + `as_completed`.
  `mp.Pool` cannot detect a worker killed mid-job (OOM, or a segfault in native
  code such as BLAS): it only resolves a task slot when a result reaches the
  output queue, and a dead process posts nothing — so a run blocked forever with
  no exception and no log line, and the advertised per-job isolation could not
  fire because the process that would raise was gone. Now raises
  `BrokenProcessPool` annotated with how far the run got.
- **`nested_parts` no longer emits empty molecules.** Rounding each float
  boundary independently let adjacent boundaries collapse onto the same integer,
  dispatching jobs with no work in them — this affected 68.4% of
  `(num_atoms, num_threads)` pairs below 120, including `(10, 8)`. Replaced with
  largest-remainder width allocation. Atom conservation and the equal-work
  property are unchanged (N=7500/M=8 still gives a 1.002 max/min work ratio).
- **One unpicklable job no longer sinks the whole batch.** Jobs are serialized
  individually; a payload that cannot be cloudpickled becomes a single failed
  `JobResult` instead of raising before any job runs.
- **Same-second runs of the same task no longer collide.** They previously
  shared directories silently, and the second run's manifest (opened `w`)
  truncated the first's outright.
- **Saves are atomic** (temp file + `os.replace`), so a `save_fn` failing
  partway can no longer leave a partial file for `load_run_outputs` to choke on.
- **Task-name inference tolerates `functools.partial` and callable objects**,
  which have no `__name__` — ironic given the library's closure support.
- **The progress display no longer perturbs the global RNG.** It used
  `random.shuffle` on the module-global state to pick worker codenames, which
  would derail a caller's reproducible sequence — a real hazard for Monte Carlo.
- **An empty job list returns an empty result** instead of raising `IndexError`
  (`process_jobs`) or `ValueError` (`run`).
- **Validation now precedes all side effects**, so a rejected call no longer
  leaves orphaned run directories and a half-written manifest behind.

### Added

- Full lifecycle logging on two loggers, `mpengine.engine` and
  `mpengine.orchestrator`, with a `NullHandler` installed by default: run
  start/done with throughput, resolved directories, dispatch start/done, a
  warning when `n_workers` is clamped to the job count, progress milestones,
  per-job outcomes (DEBUG on success, WARNING on failure), and ERROR on worker
  death. Per-job failures reaching the parent log is new — they previously
  reached only that worker's own file.
- A per-run `<log_dir>/<run_id>/run.log` capturing the parent's whole view of
  one run, written regardless of the caller's logging configuration.
- `process_jobs` gained `on_job_error` and `text_progress` parameters.

## [0.2.0] — 2026-09-02

### Breaking

- **`n_threads` renamed to `n_workers`** on both `run()` and `process_jobs()`.
  The engine dispatches OS processes, not threads, and CPython's GIL makes
  threads useless for the CPU-bound work it targets, so the old name described
  the wrong mechanism. *Migration:* rename the keyword argument at call sites.

### Changed

- **Worker defaults are no longer capped.** `N_WORKERS` was `min(cpu_count, 8)`
  and `process_jobs` defaulted to a hardcoded `24`; both now follow
  `os.cpu_count()`, so machines with more than 8 cores actually use them.
- Pool size clamps *down* to the job count — previously a 3-job run on a 16-core
  machine spawned 16 processes and paid full spawn cost for 13 that did nothing.

### Added

- An ASCII spider banner on every `run()`, with the art shipped as package data.

## [0.1.2] — 2026-09-01

- Removed the AFML attribution paragraph from the README (and so from the PyPI
  project page).

## [0.1.1] — 2026-09-01

- Corrected the PyPI project page: 0.1.0 shipped with the pre-publish README,
  which told users to install from a local path.

## [0.1.0] — 2026-09-01

Initial release: the job-dict dispatch engine (`expand_call`, `process_jobs`,
`process_jobs_`), the `run()` orchestration layer with manifests, per-worker
logs, on-disk results and per-job failure isolation, `lin_parts`/`nested_parts`
partitioning, closure and lambda support via `cloudpickle`, and a live
per-worker progress display.

[0.3.0]: https://github.com/singhamandeep-kgp/multiprocessor/releases/tag/v0.3.0
[0.2.0]: https://github.com/singhamandeep-kgp/multiprocessor/releases/tag/v0.2.0
[0.1.2]: https://github.com/singhamandeep-kgp/multiprocessor/releases/tag/v0.1.2
[0.1.1]: https://github.com/singhamandeep-kgp/multiprocessor/releases/tag/v0.1.1
[0.1.0]: https://github.com/singhamandeep-kgp/multiprocessor/releases/tag/v0.1.0
