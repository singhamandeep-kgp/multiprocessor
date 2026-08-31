"""Demo / regression proof: closures now survive `mpengine.run`'s real pool.

Not exercise-numbered, following `demo_runner.py`'s pattern of measuring
rather than asserting - it prints actual n_ok/n_failed rather than just
claiming the fix works.

Before the cloudpickle fix in `engine.process_jobs`, a closure - a function
defined inside another function, capturing that function's locals - crashed
with a PicklingError the moment `debug=False` tried to submit it to the real
process pool. `debug=True` never hit this, since it never crosses a process
boundary, which is why the two modes used to disagree on the exact same
function. This demo runs the same closure both ways and confirms they now
agree.

Run:
    python -m learning.demo_closures
"""

from __future__ import annotations

from pathlib import Path

from mpengine import run

SCRATCH = Path(__file__).parent / "_demo_closures_scratch"


def make_multiplier(factor: int):
    """A factory returning a closure - `multiply` is not module-level and
    captures `factor` from this enclosing scope, exactly the case that used
    to fail."""

    def multiply(x: int) -> int:
        return x * factor

    return multiply


def main() -> None:
    closure = make_multiplier(10)
    param_sets = [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}]

    summary_debug = run(closure, param_sets, base_dir=SCRATCH, task="closure_debug", debug=True)
    print(f"debug=True  n_ok={summary_debug.n_ok} n_failed={summary_debug.n_failed}")

    summary_pool = run(closure, param_sets, base_dir=SCRATCH, task="closure_pool", debug=False)
    print(f"debug=False n_ok={summary_pool.n_ok} n_failed={summary_pool.n_failed}")

    assert summary_debug.n_ok == len(param_sets) and summary_debug.n_failed == 0
    assert summary_pool.n_ok == len(param_sets) and summary_pool.n_failed == 0
    print("\nclosure dispatched successfully through both debug=True and debug=False.")


if __name__ == "__main__":
    main()
