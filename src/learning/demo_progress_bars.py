"""Demo: `mpengine.run(show_progress=True)` - live terminal feedback.

Not exercise-numbered, following `demo_runner.py`'s pattern. This proves
`show_progress` end-to-end: one overall bar tracking total jobs done, plus a
live rate number per worker process as it completes jobs.

The visual, multi-row, continuously-redrawing effect is only meaningful in a
real terminal - run this directly yourself:

    python -m learning.demo_progress_bars

(Piped/captured output, e.g. through an automated tool, only shows the
bars' final rendered state, not the live redraw.)

Jobs deliberately take a variable, human-visible amount of time (a plain
Python loop, not vectorized - the point here is to *watch* the bars, not to
measure computation), so you can actually see the overall bar and each
worker's rate move before the run finishes.
"""

from __future__ import annotations

from pathlib import Path

from mpengine import run

SCRATCH = Path(__file__).parent / "_demo_progress_bars_scratch"


def slow_square_sum(n: int) -> int:
    return sum(i * i for i in range(n))


def main() -> None:
    # a spread of sizes so different jobs take visibly different amounts of
    # time, and different workers finish jobs at different rates
    param_sets = [{"n": n} for n in [2_000_000, 4_000_000, 1_000_000, 6_000_000, 3_000_000, 5_000_000, 1_500_000, 7_000_000]]

    summary = run(slow_square_sum, param_sets, base_dir=SCRATCH, show_progress=True)

    print(f"\nn_jobs={summary.n_jobs} n_ok={summary.n_ok} n_failed={summary.n_failed}")
    print(f"elapsed_s={summary.elapsed_s:.3f}")
    assert summary.n_ok == len(param_sets) and summary.n_failed == 0


if __name__ == "__main__":
    main()
