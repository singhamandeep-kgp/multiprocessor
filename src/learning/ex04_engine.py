"""Ch.20 SS20.5 - the multiprocessing engine (exercises 6+7).

Every exercise so far hand-rolled its own pool plumbing: ex02 and ex03 each
built bespoke molecule-runners wired directly to a ProcessPoolExecutor. The
book's point in SS20.5 is that this is the wrong shape - you want one library
that parallelizes *unknown* functions, regardless of their arguments or output.

So a job becomes just a dict: one 'func' entry plus that function's kwargs.
`expand_call` (Snippet 20.10) turns such a dict into a call, and that one trick
is what lets `process_jobs_`/`process_jobs` dispatch anything at all without
knowing what's inside. See `mpengine/engine.py`.

Four demonstrations, on deliberately domain-neutral toy callbacks (no price
data here - this exercise is about the plumbing, not the workload):

  a. exercise 6 - `process_jobs_`, the sequential debug path
  b. exercise 7 - `process_jobs`, real mp.Pool + imap_unordered, showing
     completion order diverge from submission order
  c. exercise 7 - the same buggy job list down both paths, comparing how
     debuggable each failure is (the book's "Heisenbug" footnote)
  d. two structurally different callbacks - different argument types, different
     return types - dispatched through one `process_jobs` call, which is the
     "regardless of their arguments and output structure" claim made concrete

Run:
    python -m learning.ex04_engine
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Callable

from mpengine.engine import process_jobs, process_jobs_

N_WORKERS = os.cpu_count() or 4

# calibrated so costs span ~30x (0.03s .. 0.87s), submitted heaviest-first so
# completion order visibly scrambles relative to submission order
JOB_ENDS = [30_000_000, 1_000_000, 20_000_000, 2_000_000, 10_000_000, 3_000_000]
SMALL_JOB_ENDS = [200_000, 400_000, 600_000, 800_000]


# -- two toy callbacks (module-level: Windows `spawn` needs these importable) -
#
# Every callback returns a (label, value) pair rather than a bare value. That's
# not decoration: `imap_unordered` hands back results detached from the jobs
# that produced them, so with unordered completion the only reliable way to know
# what a result belongs to is to carry its identity inside the result itself.


def sum_of_squares(start: int, end: int) -> tuple[str, int]:
    """Deliberately unvectorized CPU-bound toy: sum of i*i over [start, end).
    Takes two ints, returns an int."""
    total = 0
    for i in range(start, end):
        total += i * i
    return f"sum_of_squares(end={end:,})", total


def char_histogram(text: str, top_n: int) -> tuple[str, dict[str, int]]:
    """Structurally unlike sum_of_squares in every way that matters to the
    engine: takes a str and an int, returns a dict rather than a scalar. The
    engine treats both identically because it never inspects either."""
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    top = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n])
    return f"char_histogram(len={len(text):,}, top_n={top_n})", top


def sum_of_squares_reference(start: int, end: int) -> int:
    """Closed form, for the correctness check only."""

    def s(n: int) -> int:  # sum of i*i for i in [0, n)
        return (n - 1) * n * (2 * n - 1) // 6 if n > 0 else 0

    return s(end) - s(start)


def make_text(seed: int, length: int) -> str:
    """Deterministic pseudo-text, so char_histogram has something to chew on
    without reading files or pulling in a dependency."""
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    value = seed
    chars = []
    for _ in range(length):
        value = (value * 1103515245 + 12345) % 2147483648
        chars.append(alphabet[value % len(alphabet)])
    return "".join(chars)


def build_job(func: Callable[..., Any], **kwargs: Any) -> dict:
    """A job is just a dict: the callback plus the kwargs it needs.

    The engine never inspects either - `expand_call` pops 'func' and splats the
    rest as kwargs, so any callable with any signature works unchanged.
    """
    return {"func": func, **kwargs}


# -- the four demonstrations ----------------------------------------------------


def demo_sequential() -> None:
    print("-- (a) exercise 6: process_jobs_ (sequential debug path) " + "-" * 20)
    jobs = [build_job(sum_of_squares, start=0, end=end) for end in SMALL_JOB_ENDS]
    print(f"submitted {len(jobs)} jobs, ends={SMALL_JOB_ENDS}")

    out = process_jobs_(jobs)
    for label, value in out:
        print(f"  {label} = {value:,}")

    max_diff = max(abs(v - sum_of_squares_reference(0, e)) for e, (_, v) in zip(SMALL_JOB_ENDS, out))
    print(f"correctness check vs closed form - max abs diff: {max_diff}")
    print(
        "No pool, no pickling, no spawn: every job ran in-process, one at a time,\n"
        "and results came back in submission order.\n"
    )


def demo_unordered() -> None:
    print("-- (b) exercise 7: process_jobs (mp.Pool + imap_unordered) " + "-" * 18)
    jobs = [build_job(sum_of_squares, start=0, end=end) for end in JOB_ENDS]
    employed = min(N_WORKERS, len(jobs))
    print(f"employing {employed} worker process(es) for {len(jobs)} job(s) (N_WORKERS={N_WORKERS})")
    print(f"submission order (end values): {JOB_ENDS}", flush=True)

    out = process_jobs(jobs, n_workers=N_WORKERS)
    sys.stderr.flush()

    print(f"completion order (labels):     {[label for label, _ in out]}")
    print(
        "Results arrive as each job finishes, not in the order submitted - the cheap\n"
        "jobs overtake the expensive ones queued ahead of them. Same uneven-cost\n"
        "problem ex03 partitioned around, surfacing here as out-of-order output.\n"
    )


def demo_debuggability() -> None:
    print("-- (c) exercise 7: the same bug, down both paths " + "-" * 28)
    # one job is missing its required `end` kwarg -> TypeError inside the callback
    jobs = [build_job(sum_of_squares, start=0, end=e) for e in SMALL_JOB_ENDS[:2]]
    jobs.append(build_job(sum_of_squares, start=0))
    jobs.extend(build_job(sum_of_squares, start=0, end=e) for e in SMALL_JOB_ENDS[2:])
    print(f"{len(jobs)} jobs, the one at index 2 deliberately missing its 'end' kwarg\n")

    # stdout is buffered but the tracebacks below go to stderr unbuffered, so
    # flush at each boundary to keep the two streams interleaved in order
    print(">>> via process_jobs_ (sequential):", flush=True)
    try:
        process_jobs_(jobs)
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
    print(flush=True)

    print(">>> via process_jobs (pool):", flush=True)
    try:
        out = process_jobs(jobs, n_workers=N_WORKERS)
        print(f"  (unexpectedly succeeded: {out})")
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
    print(flush=True)


def demo_mixed_callbacks() -> None:
    print("-- (d) two different callbacks, one engine " + "-" * 33)
    jobs = [
        build_job(sum_of_squares, start=0, end=2_000_000),
        build_job(char_histogram, text=make_text(seed=1, length=400_000), top_n=3),
        build_job(sum_of_squares, start=0, end=5_000_000),
        build_job(char_histogram, text=make_text(seed=2, length=900_000), top_n=5),
    ]
    print(f"{len(jobs)} jobs: 2x sum_of_squares(int, int)->int, 2x char_histogram(str, int)->dict")
    print("one job list, one process_jobs call, two unrelated signatures\n", flush=True)

    out = process_jobs(jobs, task="mixed", n_workers=N_WORKERS)
    sys.stderr.flush()

    for label, value in out:
        print(f"  {label}\n      -> {value}")
    print(
        "\nThe engine dispatched both callbacks through the identical code path. It\n"
        "never inspected either signature or return type - `expand_call` just pops\n"
        "'func' and splats whatever kwargs remain. That is the whole claim from the\n"
        "top of SS20.5, and nothing in engine.py needed to change to support it.\n"
    )


def main() -> None:
    demo_sequential()
    demo_unordered()
    demo_debuggability()
    demo_mixed_callbacks()

    print(
        "Expected: the sequential path fails immediately at the bad job's exact\n"
        "position, with an ordinary in-process traceback you could attach a debugger\n"
        "to. The pool path streams some successful results first, then surfaces a\n"
        "RemoteTraceback re-raised from a worker that has already died - nothing in\n"
        "it identifies which job dict was at fault. That gap is why process_jobs_\n"
        "exists at all. Read the printed tracebacks rather than assuming this.\n"
        "\n"
        "Note the labelled (label, value) returns throughout: with imap_unordered,\n"
        "results come back detached from the jobs that produced them, so identity\n"
        "has to travel inside the result itself."
    )


if __name__ == "__main__":
    main()
