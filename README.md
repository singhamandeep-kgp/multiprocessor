# mpengine

[![CI](https://github.com/singhamandeep-kgp/multiprocessor/actions/workflows/ci.yml/badge.svg)](https://github.com/singhamandeep-kgp/multiprocessor/actions/workflows/ci.yml)

*(This repo is named `multiprocessor`; the package it ships is `mpengine` — see below.)*

A small, general-purpose multiprocessing engine. Give it any callable and a list
of parameter sets; it runs them across processes and leaves behind a record of
what happened.

Every run produces three things:

| what | where |
|---|---|
| a manifest of exactly what was launched, and how it ended | `<base>/manifests/<run_id>.txt` |
| each job's result, saved to disk | `<base>/outputs/<run_id>/<label>` |
| one log file per **worker process** | `<base>/logs/<run_id>/worker_<pid>.log` |
| a log of the run as a whole | `<base>/logs/<run_id>/run.log` |

A single job failing does **not** abort the run — it is recorded as a failed
job and the rest continue.

## Install

```bash
pip install mpengine
```

Requires Python 3.10+. `numpy`, `cloudpickle` and `tqdm` come with it —
pip installs them for you.

<details>
<summary>Installing from source instead</summary>

```bash
pip install git+https://github.com/singhamandeep-kgp/multiprocessor.git   # latest from GitHub
pip install -e path/to/multiprocessor                                     # editable, for developing mpengine itself
```
</details>

## Use

```python
from mpengine import run

def my_task(x, y):
    return x * y

if __name__ == "__main__":          # required - see below
    summary = run(
        my_task,
        [{"x": 2, "y": 3}, {"x": 4, "y": 5}],
        base_dir="runs",
    )

    print(summary.n_ok, summary.n_failed)
    for r in summary.results:
        print(r.label, r.status, r.output_path or r.error)
```

### The `if __name__ == "__main__":` guard

On Windows and macOS, Python starts worker processes with `spawn`, which
re-imports your module in every worker. Without the guard, that re-import runs
your `run(...)` call again in each worker, which spawns more workers, and so on
until the process dies. Put anything that *calls* `run()` inside the guard;
your task functions themselves stay at module level, as normal.

### Placing the outputs

`base_dir` derives all three locations. To place them independently, pass any of
`output_dir`, `log_dir`, `manifest_dir` — an explicit path always wins, so you
can give a `base_dir` and still redirect just the logs.

### Reading the results back

Results are written to disk, one file per job (pickled by default). To load a
finished run back into memory as `{label: result}`:

```python
from mpengine import load_run_outputs

outputs = load_run_outputs(summary.output_dir)   # or the path run() printed
print(outputs["job_0000"])
```

Only successful jobs appear — a failed job never wrote a file, so its label is
absent; check `summary.results` to see which failed and why. If you wrote the
run with a custom `save_fn`, pass the matching reader as
`load_run_outputs(..., load_fn=your_loader)`.

### Other options

| argument | meaning |
|---|---|
| `save_fn` | how each result is written. Default `save_pickle`; supply your own (e.g. parquet) |
| `labels` | names for each job; defaults to `job_0000`, `job_0001`, … |
| `task` | names the run; defaults to `func.__name__` |
| `n_workers` | worker *process* count (not threads — see below); defaults to `os.cpu_count()`, clamped down to the number of jobs if there are fewer jobs than that |
| `debug` | run sequentially in-process — real tracebacks you can attach a debugger to, no pool |
| `show_progress` | live terminal display: one overall bar for the whole run, plus a live rate number per worker process. Ignored when `debug=True` |

Develop with `debug=True`, then flip it off. Chasing a bug through a process
pool means reading a traceback re-raised from a worker that has already exited,
and it never tells you which job dict was at fault.

### Performance

The defaults are already tuned; these are the knobs for when they aren't
enough. Measured on 8 cores with OpenBLAS against joblib 1.5.3 — reproduce
with `python benchmarks/bench.py`, which keeps each "before" path runnable
rather than quoting a remembered number.

| argument | meaning | measured |
|---|---|---|
| `blas_threads` | threads each worker's native BLAS may use for one numpy call. `'auto'` = `cpu_count // workers`; an int to set it; `None` to disable | 24 SVD jobs **8.26s → 1.20s** |
| `chunksize` | jobs per submission. `'auto'` batches large runs and collapses to 1 on small ones | 20,000 tiny jobs **5.55s → 1.01s** |
| `broadcast` | a `dict` of values shipped once per *worker* instead of once per job, delivered to your function as keyword arguments | 200 jobs + 80 MB panel **7.97s → 0.55s** |
| `reuse_pool` | keep the pool (and its broadcast payload) alive between calls instead of rebuilding it. Off by default | 10 dispatches **9.86s → 0.02s** |
| `initializer`, `initargs` | run once per worker before its first job — and unlike the stdlib's, may be a closure | — |
| `max_tasks_per_child` | recycle a worker every N jobs, so a slow leak can't end the run (Python 3.11+) | — |

**BLAS threads** is the one that catches people out. numpy hands matrix work to
a native library that is itself multi-threaded, and each worker process loads
its own copy believing it owns the machine — so eight workers each spawn eight
BLAS threads and 64 threads fight over 8 cores. Capping them means parallelism
comes from mpengine, one job per core, instead. This is on by default.

**Broadcast** is for the panel or fitted model every job needs:

```python
summary = run(score, param_sets, base_dir="runs",
              broadcast={"panel": panel}, reuse_pool=True)
```

`score` is then called as `score(**params, panel=panel)`. Worth being precise
about when it pays: batching alone already reduces a closure-captured payload
to one copy per batch, so on a single cold call with a payload under ~10 MB,
broadcast costs slightly more than it saves. Above that, or paired with
`reuse_pool=True` where the payload is delivered once for the life of the pool,
it is a different order of magnitude — 0.55s on the 80 MB case, level with
joblib's memmapped 0.58s. Call `shutdown_pools()` when you're done with a
reused pool, or let `atexit` do it.

### Logging

The library logs its whole lifecycle through the standard `logging` module and
writes nothing to stdout except the banner, so embedding it never pollutes a
host application's output. It installs only a `NullHandler` — to see anything,
configure logging yourself:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Two loggers, so either half can be tuned or silenced independently:

| logger | emits |
|---|---|
| `mpengine.orchestrator` | run start/done, resolved dirs, per-job outcomes, worker ranking |
| `mpengine.engine` | dispatch start/done, worker-count clamping, progress milestones, worker death |

Levels: **INFO** for run lifecycle, **DEBUG** for per-job success (a 10,000-job
sweep would otherwise be 10,000 INFO lines), **WARNING** for a failed job or a
clamped worker count, **ERROR** for a dead worker.

Every run also writes `<log_dir>/<run_id>/run.log` containing the parent's
whole view of that run — dispatch, milestones, every job outcome, the summary —
alongside the existing per-worker log files. That happens regardless of how you
configure logging, so a run is always a self-contained durable record.

**Progress display is terminal-aware.** Interactively you get the tqdm bars and
per-worker lines as before. When stderr is not a terminal — piped, redirected to
a log file, running under CI or a scheduler — those are suppressed (they render
as unreadable escape-sequence noise in a file) and periodic completion
milestones are logged at INFO instead.

### Closures and lambdas

`func`, and any custom `save_fn`, can be a closure or a lambda — not just a
module-level function. Jobs are serialized with `cloudpickle` before crossing
the process boundary, which (unlike stdlib `pickle`) can serialize a function
by *value* (its bytecode plus whatever it captured), not just by reference.

The one caveat: whatever a closure captures travels with **every job** that
uses it — a closure capturing a large array re-serializes that array per job.
That's a cost trade-off to be aware of, not a limitation on what's allowed.

## Going lower-level

`run()` is the convenient layer. The primitives underneath are exported too, if
you want to drive the pool yourself:

```python
from mpengine import expand_call, process_jobs, process_jobs_

jobs = [{"func": my_task, "x": 1, "y": 2}, {"func": other_task, "n": 5}]
results = process_jobs(jobs)     # n_workers defaults to os.cpu_count(); or process_jobs_ to stay sequential
```

A job is just a dict carrying its own callback plus that callback's kwargs, so a
single call can dispatch entirely different functions with different signatures
and return types.

Partitioning helpers are available for splitting work into chunks:

```python
from mpengine import lin_parts, nested_parts, parts_to_molecules
```

`lin_parts` gives equal-count chunks. `nested_parts` gives equal-*work* chunks
for triangular workloads — where item `i` costs `O(i)`, such as an
expanding-window computation — which keeps workers from idling while one
overloaded worker finishes.

## Tests

```bash
pip install -e ".[test]"
pytest                    # everything, ~30s
pytest -m "not slow"      # skip the process-pool tests, under a second
```

576 tests. The ones that spawn real pools are marked `slow` and dominate the
runtime, so the marker exists to keep a fast inner loop available - but they
run by default, because the behaviour they cover (a dead worker surfacing
instead of hanging, per-job failure attribution inside a batch, BLAS budgets
read back from inside a worker) is exactly the behaviour worth guarding.

CI runs the suite on Linux, Windows and macOS across Python 3.10, 3.12 and
3.14. The platform spread matters here: Windows and macOS start workers with
`spawn`, Linux does not, and mpengine caps BLAS threads by a different
mechanism in each case.

## Changelog

See [CHANGELOG.md](CHANGELOG.md). Versions below 1.0 may carry breaking changes
in a minor release; each is listed there with the migration needed.

## What else is in the repo

The `src/learning/` package holds demo and exercise scripts used while
developing the engine. It depends on a separate private project and is
deliberately **not** packaged — installing `mpengine` never pulls it in.
