# mpengine — project context

Paste this whole file into a new Claude Code session to resume work here with full context.

## Goal

**The primary deliverable is `mpengine`: a general-purpose, portable multiprocessing
engine** (see `README.md`). It began as a study project working through López de
Prado's *Advances in Financial Machine Learning*, Chapter 20, and that lineage still
shows in the code's docstrings — but the engine is now the point, and the exercises
are secondary/historical.

The original learning mode — study a sub-concept, then build a small runnable script
proving it with **real, measured timings** rather than claims from the book — still
governs how new work is verified.

**Scope boundary — read this first:** the `Equity_StatArb` project is used purely as
a *data source*: real US-equity price data sitting in a Delta Lake, read through its
`DataAPI`. We do **not** use or reproduce any of its statistical-arbitrage domain
logic — no pairs trading, no cointegration, no hedge-ratio OLS regressions. Every
exercise's actual *computation* must stay domain-neutral (returns, moving averages,
simple aggregate stats over price series). Only the *data plumbing* — reading Delta
tables — comes from Equity_StatArb. (This was an explicit correction mid-project:
an early ex02 draft built a pairwise-cointegration screening experiment and was
rejected for being "stat-arb," redesigned around a simple moving average instead.)

## Source material

- `Multiprocessing.md` — condensed study notes on
  Ch.20 concepts, algorithms (`linParts`/`nestedParts`/`expandCall`/`processJobs`/
  `mpPandasObj`/`processJobsRedux`), and a numbered list of ~13 exercises.
- `Chapter20_Multiprocessing_and_Vectorization.pdf`
  — the actual book chapter.

## Repo architecture

Its own small Python project, kept separate from `Equity_StatArb` (which stays
production-clean — no exercise scratch code lives inside it).

**Restructured in Aug 2026** from the old exercise-centric layout (package `mpvec`,
with `common/` + `ch20/` subpackages) into two clearly separated top-level packages:

```
<project root>/
  README.md                   <- mpengine usage docs, written for outside consumers
  CONTEXT.md                  <- this file
  Multiprocessing.md
  Chapter20_Multiprocessing_and_Vectorization.pdf
  pyproject.toml              # name "mpengine"; packages.find include = ["mpengine*"]
  .venv/                      # own venv (Python 3.14.6, matches Equity_StatArb's)
  src/
    mpengine/                 # THE DELIVERABLE - portable, no statarb coupling, numpy-only dep
      __init__.py             # public API: run, RunSummary, JobResult, save_pickle,
                              #   expand_call, process_jobs, process_jobs_, report_progress,
                              #   equal_chunks, lin_parts, nested_parts, parts_to_molecules
      engine.py               # book primitives (Snippets 20.8/20.9/20.10)
      orchestrator.py         # run(): manifests, per-worker logs, on-disk outputs, failure isolation
      partition.py            # equal_chunks, lin_parts (20.5), nested_parts (20.6), parts_to_molecules
    learning/                 # secondary/historical - deliberately NOT packaged
      __init__.py
      data.py                 # load_close_series(...) via statarb.data.api.DataAPI
      timing.py               # time_it(), time_runs(), print_table()
      ex01_vectorization_returns.py
      ex02_single_vs_multi.py
      ex03_partitioning.py
      ex04_engine.py
      demo_runner.py          # smoke test for mpengine.run
```

**Why the split matters:** `pyproject.toml` sets `include = ["mpengine*"]`, so only
`mpengine` is packaged. `learning` stays importable *locally* because the editable
install puts all of `src/` on `sys.path` — but it never ships. Verified empirically:
a built wheel contains only the four `mpengine` modules, and a throwaway consumer
project with its own venv installed just that wheel, ran `from mpengine import run`
successfully, and confirmed `statarb`/`learning`/`mpvec` were all absent.

`partition.py` deliberately lives in `mpengine`, not `learning` — `lin_parts`/
`nested_parts` are core "how do I chunk work" functionality (the book uses them
*inside* `mpPandasObj`), not exercise scaffolding.

No `uv` installed on this machine — used stdlib `venv` + `pip` instead.

**One-time setup** (don't redo unless `.venv` is deleted):
```
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ../Equity_StatArb   # gives `import statarb` (pulls in polars/deltalake/pyarrow/numpy/pandas/statsmodels)
./.venv/Scripts/python.exe -m pip install -e .                   # installs mpengine itself, editable
```
The editable installs are plain `.pth` files in `.venv/Lib/site-packages/` holding
absolute paths to each project's `src/`. Consequence: **renaming this project folder
breaks only mpengine's `.pth`** (statarb's points at `Equity_StatArb/src` and is
unaffected) — the fix is just re-running `pip install -e .`, not rebuilding the venv.

**How every exercise reads data:** `from statarb.data.api import DataAPI` — the
*only* sanctioned read path per that module's own docstring. `DataAPI()` with no
args auto-resolves to the real curated warehouse at `~/statarb-data`
(`C:\Users\AmandeepSingh\statarb-data`, ~890MB, real Delta tables:
`curated/{prices,dim_security,corporate_actions,calendar}`). Key schema facts:
`sid` (int32 surrogate key), date column is **`d`** (not `date`), `px_close` is
float32 on disk (becomes float64 if `adjusted != "raw"`).

**How to run any exercise:**
```
./.venv/Scripts/python.exe -m learning.ex01_vectorization_returns
./.venv/Scripts/python.exe -m learning.ex02_single_vs_multi
./.venv/Scripts/python.exe -m learning.ex03_partitioning
./.venv/Scripts/python.exe -m learning.ex04_engine
./.venv/Scripts/python.exe -m learning.demo_runner
```
(Not with `Equity_StatArb`'s own venv python — that one has `statarb` but not
`mpengine` installed. Two separate venvs; VS Code's interpreter picker doesn't
auto-switch per subfolder, so the Run button can silently target the wrong one —
select the right interpreter via `Ctrl+Shift+P` → "Python: Select Interpreter".)

## Exercises built so far

### ex01 — vectorization (§20.2)

Daily simple returns (`ret[i] = price[i]/price[i-1] - 1`) over AAPL's full raw
close-price history (~8,241 rows), computed 4 ways and timed with `timeit`:
1. scalar loop (explicit indexing)
2. numpy vectorized matrix algebra (`prices[1:]/prices[:-1] - 1`)
3. compiled iterator (`map` + `zip` + lambda)
4. compiled generator (generator expression over `zip`)

**Result:** numpy vectorized ~300-500x faster than the scalar loop (this number is
noisy run-to-run — its absolute time is only microseconds, so tiny system jitter is
a huge *relative* swing; see below). `map`/generator landed only ~1.1-1.3x over the
scalar loop — much less than the "2-5x" hypothesis — because all three (loop / map /
generator) still touch the numpy array **one element at a time**, paying numpy's
per-element "boxing" cost every time; only the vectorized version avoids touching
elements individually at all. **The real lesson: the loop *mechanism* barely
matters once you're indexing a numpy array element-by-element — what matters is
whether you do that at all.**

Why the numpy speedup bounces between runs (e.g. 398x vs 478x): it's measurement
noise landing disproportionately on the tiny numpy absolute time (a few
microseconds) — the same-size hiccup that's invisible against a 1ms scalar loop
becomes huge against a 2-3 microsecond numpy call, since `speedup = slow/fast` and
`fast` is the noisy small denominator.

### ex02 — §20.3: single-thread vs. multithreading vs. multiprocessing (+ vectorization)

Atom = one security's full raw close-price history → its 20-day simple moving
average (SMA). ~300 securities pulled from the curated Delta lake in **one** query
(not 300 separate ones), split into per-security numpy arrays, then computed 4 ways
with the same atom-vs-execution-mechanism separation as ex01:

- **A. single-thread** — plain loop, no executor, naive (unvectorized, per-day
  Python `sum()` over a slice) atom
- **B. multi-threading** — `ThreadPoolExecutor` over molecules (equal chunks of
  securities), *same* naive atom fn as A
- **C. multiprocessing** — `ProcessPoolExecutor` over molecules, *same* naive atom
  fn as A/B
- **D. multiprocessing + vectorized** — `ProcessPoolExecutor` over molecules,
  vectorized atom (`np.cumsum` trick)

**Result** (8 workers, window=20): A=5,239ms, B=5,184ms (**~1.0x** — GIL confirmed,
threads bought nothing for CPU-bound work), C=1,505ms (**3.5x** — real parallelism,
but short of the theoretical 8x because Windows `multiprocessing` always uses
`spawn`, so every worker pays real process-boot + module-reimport cost), D=1,031ms
(**5.1x** — better than C but not dramatically more, because once the atom itself
is vectorized/fast, the *fixed* per-process spawn overhead becomes the dominant
cost — **vectorizing can't shrink a cost that's paid once per process regardless
of how little work that process does**).

Windows-specific detail: molecule-runner functions (`run_molecule_naive`,
`run_molecule_vectorized`) had to be plain **module-level** functions (not
nested/lambdas), because Windows `multiprocessing` always uses `spawn`, which needs
worker target functions to be importable by reference.

New shared helper this exercise introduced: `mpengine/partition.py::
equal_chunks(seq, n)` — a simple equal-count atom→molecule splitter, standing in
for the book's own `linParts` (the actual quadratic-derivation algorithm is
deliberately deferred to its own future exercise rather than built inline here).

### ex03 — §20.4: `linParts` vs `nestedParts` (exercises 3+4+5)

Implemented the book's actual quadratic-derivation algorithms from scratch in
`mpengine/partition.py`: `lin_parts(num_atoms, num_threads)` (Snippet 20.5,
equal row-*count* boundaries) and `nested_parts(num_atoms, num_threads,
upper_triang=False)` (Snippet 20.6, equal row-*work* boundaries via the
iteratively-solved quadratic `(1/2)(r_m+r_{m-1}+1)(r_m-r_{m-1}) =
N(N+1)/(2M)`), plus a shared `parts_to_molecules(atoms, parts)` slicer.

Atom: row `i` (`i = 1..7500`) over AAPL's raw close-price history — the max
simple return over every lookback window `k=1..i` ending at day `i`,
deliberately unvectorized so row `i` genuinely costs `O(i)` (same triangular
shape as the book's SADF/barrier-touch motivating cases). The same atom
function is split into 8 molecules two ways and run via `ProcessPoolExecutor`
(only the molecule boundaries differ — unlike ex02, the atom fn itself never
changes across variants):
- **A. single-thread** baseline
- **B. `lin_parts`** — equal row-count molecules
- **C. `nested_parts`** — equal-work molecules (`upper_triang=False`, since
  cost increases with row index here)

**Result:** the diagnostic (pure arithmetic on the parts arrays, no workload
run) predicted `lin_parts` cost ratio 14.96x (min molecule 440k, max 6.59M)
vs. `nested_parts` 1.002x (all ~3.51-3.52M) — confirmed empirically by the
*observed* per-molecule wall-clock times: `lin_parts` spread 0.12s-1.14s
(workers finish early and idle) vs. `nested_parts` tight at ~0.79-0.82s
(workers finish together). Timed result: A=4,140ms, B=2,811ms (1.5x),
C=2,472ms (1.7x) — **nested_parts beat lin_parts by only 1.14x**, well below
the ~1.9x predicted from cost ratios alone, because Windows process-spawn
overhead (paid per molecule regardless of that molecule's size) compresses
the theoretical partitioning win, same lesson ex02 found for C vs. D.

### ex04 — §20.5: the multiprocessing engine (exercises 6+7)

The book's framing: *"It would be a mistake to write a parallelization wrapper
for each multiprocessed function"* — so ex02/ex03's bespoke molecule-runners get
replaced by a generic engine in a new shared module `mpengine/engine.py`:
`expand_call` (Snippet 20.10), `process_jobs_` (20.8), `report_progress` +
`process_jobs` (20.9). A job is just a dict — one `'func'` entry plus that
function's kwargs — and `expand_call` turns the dict into a call, which is what
lets the engine dispatch anything without knowing what's inside.

Per an explicit decision, this uses **raw `multiprocessing.Pool` +
`imap_unordered`**, mirroring the book's Snippet 20.9, rather than ex02/ex03's
`concurrent.futures` (whose `.map()` yields in submission order — `report_progress`
needs unordered as-completed streaming to report honestly).

This exercise is pure stdlib and needs no price data at all — the atom is a toy
`sum_of_squares(start, end)`, correctly domain-neutral since the subject is the
plumbing, not the workload.

**Three deviations from the book, each deliberate:**
- `expand_call` **copies** `kargs` before popping `'func'`. The book's literal
  `kargs.pop('func')` mutates the caller's dict, so a job list can only be run
  once — a second pass dies with `KeyError('func')`. ex04 runs the *same* list
  through both paths to compare them, so the copy is load-bearing.
- `process_jobs` uses explicit `close()`/`join()` in a `try/finally`, **not**
  `with mp.Pool(...) as pool:` — `Pool.__exit__` calls `terminate()` (an abrupt
  kill), not the book's graceful shutdown.
- `report_progress` keeps the book's *minute* units even though a fast demo
  always shows "0.00 minutes"; the snippet is sized for real multi-minute runs.

**Results:** (b) submission order `[30M, 1M, 20M, 2M, 10M, 3M]` came back in
completion order `[1M, 2M, 3M, 10M, 20M, 30M]` — perfectly inverted, cheap jobs
overtaking expensive ones queued ahead of them (the same uneven-cost problem
ex03 partitioned around, surfacing here as out-of-order output). (c) with one
job deliberately missing its `end` kwarg: the sequential path failed immediately
at that job's exact position with a 3-frame in-process traceback ending at
`expand_call` — debugger-attachable. The pool path reported **20% done before
failing** (a good result had already streamed), then surfaced a
`multiprocessing.pool.RemoteTraceback` from a now-dead worker, chained through
`pool.py:873 raise value` — and **nothing in it identifies which job dict was at
fault**. That gap is the whole reason `process_jobs_` exists.

Implementation note: stdout is buffered but `report_progress`/`traceback` write
to unbuffered stderr, so `ex04` flushes at section boundaries to keep the two
streams interleaved in order when piped.

**Follow-up (d):** the user flagged that `build_job` originally hardcoded
`sum_of_squares` as its callback, so despite `mpengine/engine.py` itself already
being fully generic (`expand_call` never references any specific function),
the exercise never actually *demonstrated* the book's "regardless of their
arguments and output structure" claim — every demo dispatched the same
function. Fixed by generalizing `build_job(func, **kwargs)` and adding a
second callback, `char_histogram(text: str, top_n: int) -> dict[str, int]`
— deliberately unlike `sum_of_squares` in both argument types (str/int vs.
int/int) and return type (dict vs. int). A new demo (d) dispatches both
through one `process_jobs` call; both `char_histogram` jobs finished before
either `sum_of_squares` job at the chosen sizes, so the run visibly proved
out-of-order dispatch of two unrelated signatures through the identical code
path, with `dict` and `int` results sitting side by side in the output — and
`mpengine/engine.py` needed zero changes to support it. Also switched every
callback to return a labelled `(label, value)` pair rather than a bare value:
demo (b)'s original approach reverse-mapped results back to their jobs by
exploiting that `sum_of_squares` is monotonic in `end`, which only works with
one shared function and breaks outright once callbacks differ — with
`imap_unordered`, results arrive detached from the jobs that produced them, so
identity has to travel inside the result itself.

## Project pivot: `mpengine` becomes the deliverable

After ex04, the project's shape changed on purpose. Rather than continuing
strictly numbered book exercises, `mpengine` is now meant to be a real,
reusable multiprocessing toolkit — the remaining book concepts (mpPandasObj,
mpBatches, output reduction, bound-method pickling) fold in as *features* of
that toolkit wherever they're the natural fit, rather than staying separate
numbered demo scripts. `learning/exNN_*.py` scripts stop being the primary
deliverable; new work lands directly in `common/` with a runnable
(non-numbered) demo alongside it.

### `mpengine/orchestrator.py` — production orchestration on top of `engine.py`

The user's concrete ask: one entry point where you pick a function, give it
parameter sets, and get back an organized, resilient run — a manifest
recording exactly what was launched, every job's output saved to disk, and
per-worker-process log files so a failure is attributable to a specific
process. `engine.py` itself is untouched — `run()` wraps it, reusing
`expand_call`/`process_jobs_`/`process_jobs` rather than reimplementing them.

Key design choices:
- **One log file per worker process** (keyed by `os.getpid()`, lazily opened
  on that worker's first job, reused for every job after) — not per job, since
  workers are reused across many jobs. Confirmed by an actual run: worker
  `12220` handled 3 jobs including the deliberately-broken one, and its single
  log file shows `starting job_0000` → `completed job_0000` → `starting
  job_0002` → `failed job_0002: ... [full traceback]` → `starting job_0004` →
  `completed job_0004` — one continuous, attributable record.
- **Failure isolation, deliberately the opposite of ex04.** ex04's
  `process_jobs` lets an exception abort the whole run (that was the point of
  its Heisenbug demo). `runner.py`'s own per-job wrapper
  (`_run_and_save_job`) catches exceptions instead, records a failed
  `JobResult`, and the run continues — confirmed: a 5-job demo with one job
  missing a required kwarg still returned `n_ok=4, n_failed=1`, not an
  aborted run.
- **Caller-specified output format** (`save_fn`, default `save_pickle`) rather
  than auto-detecting by type — same Windows-spawn constraint as any Pool
  target: must be module-level/importable.
- **Per-run isolation via a `run_id`** (`{task}_{timestamp}`): every
  destination (`output_dir`, `log_dir`, `manifest_dir`) gets a `<run_id>`
  subfolder, so successive runs never collide and a worker's pid-named log
  file can't be confused with a previous run's.
- **`debug=True`** routes through `process_jobs_` (sequential, in-process) for
  clean tracebacks while developing a new function, same distinction ex04
  established — confirmed this still creates exactly one log file (the main
  process's own pid), since `_run_and_save_job` runs regardless of debug mode.

Demo: `learning/demo_runner.py` (not exercise-numbered) — 5 toy jobs, 1
deliberately broken, printing the actual manifest contents and listing the
actual output/log files `run()` produced, per this project's "measured, not
asserted" ethos. Its own scratch output (`src/learning/_demo_runner_scratch/`) is
gitignored and deleted after each verification run, not committed.

**Book concepts not yet folded in** (still open, no longer tied to exercise
numbers): pickling bound methods (`copy_reg`), `mpBatches`-style
oversubscription, on-the-fly output reduction (`processJobsRedux`), and the
one-touch double-barrier integration example (Snippets 20.3-20.4, now
unblocked since the PDF text was captured in conversation).

## Environment notes (workflow, not project knowledge)

- VS Code is opened with `C:\Users\AmandeepSingh\Personal` as the workspace root.
  Workspace-level settings (`files.exclude` etc.) must live at
  `C:\Users\AmandeepSingh\Personal\.vscode\settings.json` — a nested `.vscode`
  folder inside a subfolder (e.g. inside `Multiprocessing_and_Vectorisation`
  itself) is inert and ignored by VS Code.
- A personal Claude Code skill exists at
  `~/.claude/skills/hide-vscode-clutter/SKILL.md` (hides `*.egg-info`,
  `__pycache__`, `.venv`, `.pytest_cache`, `__init__.py` from the Explorer tree) —
  already applied, persists automatically; no need to reapply in a new session.
- Two separate venvs exist (`Equity_StatArb/.venv` and
  this project's `.venv`) — VS Code's selected Python
  interpreter does not auto-switch between them per subfolder.
