# Chapter 20 — Multiprocessing and Vectorization
Study notes: concepts → algorithms → exercises

---

## (a) Core concepts, in learning order

1. **Vectorization first.** Before reaching for multiple processes, check whether nested loops can be replaced with array/iterator operations. Cheapest form of parallelism — no processes, no pickling, no overhead.

2. **Why Python can't just use threads.** The GIL lets only one thread per core hold write access at a time, so Python threads don't give real parallelism for CPU-bound work. Processes don't share memory, so they *can* run truly in parallel — at the cost of needing to explicitly pass data in and results out.

3. **Atoms vs. molecules.** Before you can parallelize anything, you need a vocabulary for splitting work:
   - *Atom* = one indivisible unit of work.
   - *Molecule* = a group of atoms processed sequentially, single-threaded, inside one process.
   - Parallelism happens at the molecule level — you're choosing how to bucket atoms into molecules, then handing each molecule to a core.

4. **Partitioning strategy depends on the workload shape.**
   - Equal-cost atoms → linear partition (equal-sized molecules).
   - Triangular-cost atoms (nested loops where inner loop length depends on outer index) → nested partition (unequal molecule *sizes*, equal molecule *work*).

5. **A generic multiprocessing engine, not one-off wrappers.** The goal is a reusable pipeline: partition atoms → build job dicts → dispatch to a pool → unwrap and call the target function → stitch results back together. Once this pipeline exists, any function can be parallelized without bespoke plumbing.

6. **The dispatch mechanics you have to handle yourself.**
   - Passing an arbitrary function + kwargs through a pool (`expandCall`'s dict-to-call trick).
   - Bound methods aren't picklable by default — you need the `copy_reg` workaround.
   - Progress reporting, since async jobs complete out of order.

7. **Output reduction as a second, independent reason to parallelize.** Beyond wall-clock speed, batching + on-the-fly reduction lets you process datasets too large to hold fully in memory (accumulate a running result instead of collecting every molecule's output in a list).

8. **mpBatches — decoupling molecule count from core count.** Splitting into *more* molecules than you have cores (and letting the pool pull from the queue as cores free up) smooths out uneven per-molecule workloads.

---

## (b) Algorithms / pseudocode worth reimplementing

### 1. Vectorized Cartesian product
```
# atoms: dict of lists, e.g. {'a': [...], 'b': [...], 'c': [...]}
from itertools import product
jobs = (dict(zip(dict0.keys(), combo)) for combo in product(*dict0.values()))
```
Worth internalizing because it generalizes to N dimensions with zero extra code — the un-vectorized version needs one more nested loop per dimension.

### 2. `linParts` — equal-count partition
```
def linParts(numAtoms, numThreads):
    parts = linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
    return ceil(parts).astype(int)
```
Produces N+1 boundary indices that slice a flat atom list into N equal (±1) chunks.

### 3. `nestedParts` — equal-work partition over a triangular task set
```
def nestedParts(numAtoms, numThreads, upperTriang=False):
    parts = [0]
    numThreads_ = min(numThreads, numAtoms)
    for _ in range(numThreads_):
        prev = parts[-1]
        part = 1 + 4 * (prev**2 + prev + numAtoms*(numAtoms+1) / numThreads_)
        part = (-1 + sqrt(part)) / 2
        parts.append(part)
    parts = round(parts).astype(int)
    if upperTriang:                     # front-load the heavy rows
        parts = cumsum(diff(parts)[::-1])
        parts = [0] + parts
    return parts
```
Key idea to reimplement from scratch (don't just copy): derive `r_m` yourself from
`(1/2)(r_m + r_{m-1} + 1)(r_m - r_{m-1}) = (1/2M)N(N+1)` — solving the quadratic is the actual exercise; the closed form is easy to transcribe but easy to misapply if you don't know where it came from.

### 4. `expandCall` — the job-dict-to-function-call trick
```
def expandCall(kargs):
    func = kargs.pop('func')
    return func(**kargs)
```
This is the hinge of the whole engine: a job is just a dict; this function is what turns "a dict" into "a call."

### 5. `processJobs_` — sequential fallback (debugging mode)
```
def processJobs_(jobs):
    return [expandCall(job) for job in jobs]
```

### 6. `processJobs` — async pool dispatch + progress reporting
```
def processJobs(jobs, task=None, numThreads=24):
    if task is None:
        task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    outputs = pool.imap_unordered(expandCall, jobs)
    out, t0 = [], time.time()
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        reportProgress(i, len(jobs), t0, task)
    pool.close(); pool.join()   # prevents memory leaks
    return out
```

### 7. `mpPandasObj` — the full pipeline glued together
```
def mpPandasObj(func, pdObj, numThreads=24, mpBatches=1, linMols=True, **kargs):
    name, atoms = pdObj
    parts = linParts(len(atoms), numThreads*mpBatches) if linMols \
            else nestedParts(len(atoms), numThreads*mpBatches)
    jobs = [{name: atoms[parts[i-1]:parts[i]], 'func': func, **kargs}
            for i in range(1, len(parts))]
    out = processJobs_(jobs) if numThreads == 1 else processJobs(jobs, numThreads=numThreads)
    return stitch(out)   # concat/sort DataFrames or Series, or return list as-is
```

### 8. Pickling bound methods (boilerplate, but worth knowing *why*)
```
def _pickle_method(method):
    return _unpickle_method, (method.im_func.__name__, method.im_self, method.im_class)

def _unpickle_method(func_name, obj, cls):
    for c in cls.mro():
        if func_name in c.__dict__:
            return c.__dict__[func_name].__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
```

### 9. On-the-fly output reduction — `processJobsRedux`
```
def processJobsRedux(jobs, task=None, numThreads=24, redux=None, reduxArgs={}, reduxInPlace=False):
    pool = mp.Pool(processes=numThreads)
    out = None
    for i, out_ in enumerate(pool.imap_unordered(expandCall, jobs), 1):
        if out is None:
            out = out_
        elif reduxInPlace:
            redux(out, out_, **reduxArgs)
        else:
            out = redux(out, out_, **reduxArgs)
        reportProgress(i, len(jobs), ...)
    pool.close(); pool.join()
    return out
```
The exercise-worthy insight: `redux` + `reduxInPlace` is a general accumulator pattern (like `functools.reduce`, but streaming and parallel).

---

## (c) Exercises to prove you understand each concept

**Vectorization**
1. Write both the loop and vectorized version of an N-dimensional Cartesian product where N is a runtime parameter (not hardcoded). Time both for N = 3, 6, 10.

**GIL / threads vs. processes**
2. Write a CPU-bound function (e.g. sum of squares over a large range). Run it with `threading` across 4 threads and with `multiprocessing` across 4 processes. Measure wall-clock time for both and explain the gap from what you measured (not from memory).

**Atoms & molecules / partitioning**
3. Implement `linParts` from scratch without looking at the snippet. Verify it produces N (or fewer) roughly-equal-sized index ranges for a list of 137 atoms and 8 threads.
4. Implement `nestedParts` from scratch, including re-deriving the quadratic solution for `r_m` on paper first. Test it on a lower-triangular workload of N=50 rows split across 6 threads, and confirm (empirically, by summing `i` per partition) that total work per partition is roughly equal.
5. Generate the equivalent of Figure 20.2 yourself: plot per-task cost (1..N) and per-partition total cost, for both `linParts` and `nestedParts`, to see the imbalance linear partitioning creates.

**Multiprocessing engine internals**
6. Implement `expandCall` and `processJobs_` and use them (with `numThreads` effectively 1) to run a toy function across a hand-built list of job dicts. Confirm you understand why this mode exists (debugging traceback clarity vs. multiprocessing).
7. Implement `processJobs` with real `multiprocessing.Pool` + `imap_unordered`, including a working `reportProgress`. Deliberately introduce a bug in the callback and observe how much harder it is to debug than the `numThreads=1` path.
8. Reproduce the pickling boilerplate and demonstrate the failure it fixes: write a class with a bound method, submit it via `Pool.map` without the `copy_reg` registration, watch it fail, then add the registration and watch it succeed.
9. Build your own minimal `mpPandasObj` (function name it something else) that accepts a callback, an atom list, `numThreads`, and `mpBatches`, and returns a stitched result. Test it on a function that returns a `pandas.Series` per molecule.

**mpBatches**
10. Take a workload with one artificially expensive molecule and the rest cheap. Run it with `mpBatches=1` and `mpBatches=10` on the same `numThreads`, and measure the time difference. Explain the result in terms of core idling.

**Output reduction / memory**
11. Implement `processJobsRedux` with `redux=list.append` and `reduxInPlace=True`, and separately with `redux=pd.DataFrame.add` and `reduxInPlace=False`. Confirm both produce correct final output on a toy dataset.
12. Simulate the memory-management motivation from Section 20.6: generate several CSV "chunks" too large to all fit comfortably in memory at once (or fake this with a small memory budget), compute a partial matrix product per chunk, and reduce on the fly with `pd.DataFrame.add` rather than collecting all chunk outputs in a list first. Confirm peak memory stays bounded as you increase the number of chunks.

**Integration**
13. Re-implement the one-touch double-barrier example (Snippets 20.3–20.4) end-to-end using your own `mpPandasObj`-equivalent instead of a hand-rolled `mp.Pool` call, and confirm the result matches the single-threaded version.
