# mpengine

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

## Changelog

See [CHANGELOG.md](CHANGELOG.md). Versions below 1.0 may carry breaking changes
in a minor release; each is listed there with the migration needed.

## What else is in the repo

The `src/learning/` package holds demo and exercise scripts used while
developing the engine. It depends on a separate private project and is
deliberately **not** packaged — installing `mpengine` never pulls it in.
