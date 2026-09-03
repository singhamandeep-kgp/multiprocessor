"""run(): manifests, on-disk outputs, per-worker logs, failure isolation.

This is the layer mpengine is actually differentiated on - joblib gives you a
function call, this gives you a run you can audit afterwards - so the
assertions here are mostly about what survives on disk, not about return
values.
"""

from __future__ import annotations

import functools
import os
import pickle
from pathlib import Path

import pytest

from mpengine.orchestrator import (
    _safe_task_name,
    _validate_labels,
    load_run_outputs,
    run,
)
from tests import workers


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_run_produces_a_complete_run_record(base_dir):
    summary = run(workers.square, [{"x": i} for i in range(5)],
                  base_dir=base_dir, task="happy")

    assert summary.n_jobs == 5 and summary.n_ok == 5 and summary.n_failed == 0
    assert summary.elapsed_s > 0
    assert summary.run_id.startswith("happy_")

    manifest = Path(summary.manifest_path)
    assert manifest.is_file()
    text = manifest.read_text()
    assert "n_ok: 5" in text and "n_failed: 0" in text
    assert "func: square" in text
    for i in range(5):
        assert f"job_{i:04d}" in text

    outputs = sorted(p.name for p in Path(summary.output_dir).iterdir())
    assert outputs == [f"job_{i:04d}" for i in range(5)]
    assert (Path(summary.log_dir) / "run.log").is_file()


@pytest.mark.slow
def test_results_come_back_in_submission_order(base_dir):
    """process_jobs yields in completion order by design, but these results
    are label-addressed - a caller zipping them against the param_sets it
    passed in would otherwise silently mismatch."""
    param_sets = [{"x": i, "seconds": 0.12 - i * 0.02} for i in range(5)]
    summary = run(workers.slow, param_sets, base_dir=base_dir, task="order")
    assert [r.label for r in summary.results] == [f"job_{i:04d}" for i in range(5)]


@pytest.mark.slow
def test_outputs_read_back_by_label(base_dir):
    summary = run(workers.square, [{"x": i} for i in range(4)],
                  base_dir=base_dir, task="readback")
    loaded = load_run_outputs(summary.output_dir)
    assert loaded == {f"job_{i:04d}": i * i for i in range(4)}


@pytest.mark.slow
def test_custom_labels_name_the_output_files(base_dir):
    summary = run(workers.square, [{"x": 2}, {"x": 3}],
                  base_dir=base_dir, task="labelled", labels=["two", "three"])
    assert load_run_outputs(summary.output_dir) == {"two": 4, "three": 9}


@pytest.mark.slow
def test_explicit_directories_override_the_derived_ones(tmp_path):
    logs = tmp_path / "elsewhere_logs"
    summary = run(workers.square, [{"x": 1}], base_dir=tmp_path / "runs",
                  log_dir=logs, task="split")
    assert Path(summary.log_dir).is_relative_to(logs)
    assert Path(summary.output_dir).is_relative_to(tmp_path / "runs" / "outputs")


# --------------------------------------------------------------------------
# Failure isolation - the actual point of the layer
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_one_failing_job_does_not_lose_the_others(base_dir):
    summary = run(workers.boom_on, [{"x": i, "victim": 2} for i in range(5)],
                  base_dir=base_dir, task="isolate")
    assert summary.n_ok == 4 and summary.n_failed == 1

    failed = [r for r in summary.results if r.status == "error"]
    assert [r.label for r in failed] == ["job_0002"]
    assert "job 2 exploded" in failed[0].error
    assert failed[0].output_path is None

    # The failure is in the manifest, and the survivors are still on disk.
    assert "job_0002: ERROR" in Path(summary.manifest_path).read_text()
    assert len(load_run_outputs(summary.output_dir)) == 4


@pytest.mark.slow
def test_a_failing_save_fn_leaves_nothing_partial_behind(base_dir):
    """Saves are written to a sibling temp path and renamed, so a save_fn that
    dies partway cannot leave a corpse at the real output path - which a later
    load_run_outputs would choke on, taking down the read-back of an otherwise
    healthy run."""
    summary = run(workers.square, [{"x": 1}, {"x": 2}],
                  base_dir=base_dir, task="badsave", save_fn=workers.save_boom)
    assert summary.n_failed == 2
    assert all("save failed" in r.error for r in summary.results)
    assert list(Path(summary.output_dir).iterdir()) == [], "no partial files"
    assert load_run_outputs(summary.output_dir) == {}


@pytest.mark.slow
def test_load_run_outputs_skips_partial_files(base_dir):
    summary = run(workers.square, [{"x": 5}], base_dir=base_dir, task="partial")
    # Simulate a write that was interrupted between temp file and rename.
    (Path(summary.output_dir) / ".job_0001.partial").write_bytes(b"truncated")
    assert load_run_outputs(summary.output_dir) == {"job_0000": 25}


def test_load_run_outputs_rejects_a_non_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        load_run_outputs(tmp_path / "does_not_exist")


# --------------------------------------------------------------------------
# Label validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label,reason",
    [
        ("../escaped", "contains a path separator"),
        ("a/b", "contains a path separator"),
        ("a\\b", "contains a path separator"),
        ("C:/Windows/Temp/x", "absolute path"),
        ("", "empty"),
        ("   ", "empty"),
    ],
)
def test_unsafe_labels_are_rejected(label, reason):
    """Each label is used directly as a filename, and pathlib's `/` is
    unforgiving: an absolute label discards the base directory entirely, `..`
    walks out of the run tree, and an empty one collapses onto the run
    directory so save_fn is handed a directory to write."""
    with pytest.raises(ValueError, match="not usable"):
        _validate_labels([label])


def test_duplicate_labels_are_rejected():
    """Two jobs sharing a label write to one path, both report ok with the
    same output_path, and whichever finishes last silently wins."""
    with pytest.raises(ValueError, match="must be unique"):
        _validate_labels(["a", "b", "a"])


def test_ordinary_labels_pass():
    _validate_labels(["job_0000", "run.2024-01-01", "alpha_beta"])


def test_label_count_must_match_param_sets(base_dir):
    with pytest.raises(ValueError, match="2 labels for 3 param sets"):
        run(workers.square, [{"x": i} for i in range(3)],
            base_dir=base_dir, labels=["a", "b"])


# --------------------------------------------------------------------------
# Validation precedes side effects
# --------------------------------------------------------------------------
def test_a_rejected_call_creates_no_directories(base_dir):
    """A rejected call used to leave orphaned run directories and a
    half-written manifest behind."""
    with pytest.raises(ValueError):
        run(workers.square, [{"x": 1}], base_dir=base_dir, labels=["../escape"])
    assert not base_dir.exists()


def test_missing_destinations_are_reported_together(tmp_path):
    with pytest.raises(ValueError, match="output_dir, log_dir, manifest_dir"):
        run(workers.square, [{"x": 1}])


def test_partial_destinations_name_only_what_is_missing(tmp_path):
    with pytest.raises(ValueError, match="missing: log_dir, manifest_dir"):
        run(workers.square, [{"x": 1}], output_dir=tmp_path / "o")


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_an_empty_param_set_list_is_an_empty_run(base_dir):
    """Previously a ValueError, from inferring a task name off an empty list."""
    summary = run(workers.square, [], base_dir=base_dir, task="empty")
    assert summary.n_jobs == 0 and summary.n_ok == 0 and summary.n_failed == 0
    assert summary.results == []


@pytest.mark.slow
def test_back_to_back_runs_never_share_a_directory(base_dir):
    """At second resolution two runs of the same task produced an identical
    run_id, and because the directories are made with exist_ok=True they were
    silently shared - the second run's manifest, opened 'w', truncated the
    first outright."""
    first = run(workers.square, [{"x": 1}], base_dir=base_dir, task="samesecond")
    second = run(workers.square, [{"x": 2}], base_dir=base_dir, task="samesecond")

    assert first.run_id != second.run_id
    assert first.output_dir != second.output_dir
    assert Path(first.manifest_path).read_text().strip(), "first manifest was truncated"
    assert load_run_outputs(first.output_dir) == {"job_0000": 1}
    assert load_run_outputs(second.output_dir) == {"job_0000": 4}


@pytest.mark.slow
def test_debug_mode_runs_in_the_calling_process(base_dir):
    summary = run(workers.current_pid, [{"_x": i} for i in range(3)],
                  base_dir=base_dir, task="dbg", debug=True)
    assert summary.n_ok == 3
    assert set(load_run_outputs(summary.output_dir).values()) == {os.getpid()}


@pytest.mark.slow
def test_a_partial_or_callable_object_can_be_the_target(base_dir):
    """Neither has __name__, and both are exactly what the library advertises
    support for - so task-name inference must not fall over on them."""
    summary = run(functools.partial(workers.add, y=10), [{"x": 1}], base_dir=base_dir)
    assert summary.n_ok == 1
    assert load_run_outputs(summary.output_dir) == {"job_0000": 11}

    summary = run(workers.CallableObject(), [{"x": 5}], base_dir=base_dir)
    assert summary.n_ok == 1


@pytest.mark.slow
def test_a_lambda_target_does_not_produce_an_invalid_path(base_dir):
    """A lambda's __name__ is literally '<lambda>', and `<` and `>` are not
    legal in Windows filenames - so an unnamed target used to crash run()'s
    own directory creation with WinError 123 before a single job ran."""
    summary = run(lambda x: x * 3, [{"x": 4}], base_dir=base_dir)
    assert summary.n_ok == 1
    assert load_run_outputs(summary.output_dir) == {"job_0000": 12}


@pytest.mark.slow
def test_a_custom_save_fn_may_be_a_closure(base_dir):
    """Like any other job field, save_fn crosses the process boundary via
    cloudpickle rather than stdlib pickle."""
    suffix = b"|checked"

    def save_with_marker(obj, path):
        path.write_bytes(pickle.dumps(obj) + suffix)

    summary = run(workers.square, [{"x": 6}], base_dir=base_dir,
                  task="closuresave", save_fn=save_with_marker)
    assert summary.n_ok == 1
    assert (Path(summary.output_dir) / "job_0000").read_bytes().endswith(suffix)


@pytest.mark.slow
def test_each_worker_gets_its_own_log_file(base_dir):
    summary = run(workers.slow, [{"x": i, "seconds": 0.05} for i in range(8)],
                  base_dir=base_dir, task="workerlogs", n_workers=2, chunksize=1)
    logs = list(Path(summary.log_dir).glob("worker_*.log"))
    assert logs, "a failure must be attributable to a specific process"
    assert any("completed" in p.read_text() for p in logs)


# --------------------------------------------------------------------------
# Task-name sanitising
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<lambda>", "lambda"),      # the case that actually crashed run()
        ("my_task", "my_task"),      # ordinary names are untouched
        ("CallableObject", "CallableObject"),
        ("a/b", "a_b"),
        (r"a\b", "a_b"),
        ("x:y|z", "x_y_z"),
        ("tab\there", "tab_here"),   # control characters are illegal too

        ("  spaced  ", "spaced"),
        ("...", "job"),              # nothing usable left
        ("", "job"),
        ("con", "con_task"),         # a Windows device name
        ("LPT1", "LPT1_task"),
    ],
)
def test_task_names_are_made_safe_for_a_directory(raw, expected):
    assert _safe_task_name(raw) == expected


@pytest.mark.slow
def test_the_manifest_keeps_the_original_function_name(base_dir):
    """Sanitising is for the directory, not for the record - a lambda still
    reads as `<lambda>` in the manifest even though its run directory cannot
    be called that."""
    summary = run(lambda x: x + 1, [{"x": 1}], base_dir=base_dir)
    text = Path(summary.manifest_path).read_text()
    assert "func: <lambda>" in text
    assert "task: lambda" in text
    assert "lambda_" in summary.run_id and "<" not in summary.run_id
