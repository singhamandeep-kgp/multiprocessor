# mpengine

A small, general-purpose multiprocessing engine. Give it any callable and a list
of parameter sets; it runs them across processes and leaves behind a record of
what happened.

Every run produces three things:

| what | where |
|---|---|
| a manifest of exactly what was launched, and how it ended | `<base>/manifests/<run_id>.txt` |
| each job's result, saved to disk | `<base>/outputs/<run_id>/<label>` |
| one log file per **worker process** | `<base>/logs/<run_id>/worker_<pid>.log` |

A single job failing does **not** abort the run — it is recorded as a failed
job and the rest continue.

## Install

```bash
pip install -e path/to/mpengine        # editable, for local development
pip install path/to/mpengine           # or a plain install
```

Only `numpy` is required.

## Use

```python
from mpengine import run

def my_task(x, y):        # must be module-level - see the constraint below
    return x * y

summary = run(
    my_task,
    [{"x": 2, "y": 3}, {"x": 4, "y": 5}],
    base_dir="runs",
)

print(summary.n_ok, summary.n_failed)
for r in summary.results:
    print(r.label, r.status, r.output_path or r.error)
```

### Placing the outputs

`base_dir` derives all three locations. To place them independently, pass any of
`output_dir`, `log_dir`, `manifest_dir` — an explicit path always wins, so you
can give a `base_dir` and still redirect just the logs.

### Other options

| argument | meaning |
|---|---|
| `save_fn` | how each result is written. Default `save_pickle`; supply your own (e.g. parquet) — it must be module-level |
| `labels` | names for each job; defaults to `job_0000`, `job_0001`, … |
| `task` | names the run; defaults to `func.__name__` |
| `n_threads` | worker count; defaults to `min(cpu_count, 8)` |
| `debug` | run sequentially in-process — real tracebacks you can attach a debugger to, no pool |

Develop with `debug=True`, then flip it off. Chasing a bug through a process
pool means reading a `RemoteTraceback` from a worker that has already exited,
and it never tells you which job dict was at fault.

### The one hard constraint

`func`, and any custom `save_fn`, must be **module-level functions**. They are
pickled by reference to cross the process boundary, so a lambda, a closure, or a
function defined inside another function will fail. This is a property of
`spawn`, which Windows always uses.

## Going lower-level

`run()` is the convenient layer. The primitives underneath are exported too, if
you want to drive the pool yourself:

```python
from mpengine import expand_call, process_jobs, process_jobs_

jobs = [{"func": my_task, "x": 1, "y": 2}, {"func": other_task, "n": 5}]
results = process_jobs(jobs, n_threads=8)     # or process_jobs_ to stay sequential
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

## Provenance

The dispatch core follows López de Prado, *Advances in Financial Machine
Learning*, Chapter 20; docstrings name the specific snippets. Nothing in
`mpengine` is finance-specific.

The `src/learning/` package in this repo holds the exercises the engine grew out
of. It depends on a separate private project and is deliberately **not**
packaged — installing `mpengine` elsewhere never pulls it in.
