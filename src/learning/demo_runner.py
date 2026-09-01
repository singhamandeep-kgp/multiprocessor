"""Demo / smoke test for `mpengine.run`.

Not exercise-numbered: this proves the production orchestration layer works
end-to-end on a toy function, in this project's usual style of measuring
rather than asserting - it prints the actual manifest contents and lists the
actual files `run()` produced, rather than just claiming it works.

Five jobs, one deliberately broken (missing a required kwarg), to prove
failure isolation: the run must return 4 ok + 1 failed, not abort.

Run:
    python -m learning.demo_runner
"""

from __future__ import annotations

from pathlib import Path

from mpengine import run

SCRATCH = Path(__file__).parent / "_demo_runner_scratch"


# -- the toy function (module-level: Windows `spawn` needs this importable) --


def sum_of_squares(start: int, end: int) -> int:
    return sum(i * i for i in range(start, end))


def main() -> None:
    param_sets = [
        {"start": 0, "end": 100_000},
        {"start": 0, "end": 200_000},
        {"start": 0},  # deliberately missing 'end' -> TypeError inside the worker
        {"start": 0, "end": 300_000},
        {"start": 0, "end": 400_000},
    ]

    # base_dir derives <base>/outputs, <base>/logs and <base>/manifests; pass
    # output_dir/log_dir/manifest_dir instead to place them independently
    summary = run(sum_of_squares, param_sets, base_dir=SCRATCH)

    print(f"run_id: {summary.run_id}")
    print(f"n_jobs={summary.n_jobs} n_ok={summary.n_ok} n_failed={summary.n_failed}")
    print(f"elapsed_s={summary.elapsed_s:.3f}\n")

    for r in summary.results:
        if r.status == "ok":
            print(f"  {r.label}: ok -> {r.output_path}")
        else:
            print(f"  {r.label}: ERROR - {r.error}")

    print(f"\n--- manifest ({summary.manifest_path}) ---")
    print(Path(summary.manifest_path).read_text())

    output_dir = Path(summary.output_dir)
    log_dir = Path(summary.log_dir)
    print(f"--- output files in {output_dir} ---")
    for p in sorted(output_dir.iterdir()):
        print(f"  {p.name}")

    print(f"\n--- log files in {log_dir} ---")
    for p in sorted(log_dir.iterdir()):
        print(f"  {p.name}")

    assert summary.n_ok == len(param_sets) - 1, "expected exactly one job to fail"
    assert summary.n_failed == 1, "expected exactly one job to fail"
    print("\nfailure isolation confirmed: 4 jobs succeeded despite 1 job failing.")


if __name__ == "__main__":
    main()
