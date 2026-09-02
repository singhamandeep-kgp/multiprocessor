"""The run banner - purely cosmetic, deliberately kept out of everything else.

`orchestrator.py` imports exactly one name from here and calls it once. All the
ASCII art, formatting and printing lives in this module, so the engine and the
orchestration layer stay free of decoration.

The spider is the mascot: a multiprocessing engine, and a creature famous for
having eight legs. The art itself lives in `spider.txt` rather than as a string
literal here - it is data, not code, the same call made for `worker_names.json`.
"""

from __future__ import annotations

import sys
from importlib import resources


def _load_spider() -> str:
    """Read the art from packaged data.

    `read_text` applies universal-newline translation, so the file's stored
    CRLF endings come back as `\\n` and render correctly on any OS without the
    stored bytes ever being modified - which matters, because the art is kept
    byte-for-byte as supplied.
    """
    return resources.files("mpengine").joinpath("spider.txt").read_text(encoding="utf-8")


def print_banner(task: str, n_workers: int, debug: bool, file=sys.stdout) -> None:
    """Print the spider, then one line saying what is being launched.

    Goes to stdout by default, matching the other run-level reporting `run()`
    does (the stored-here paths, the worker ranking). The stderr stream stays
    reserved for the live-redraw progress display, which is a different kind of
    output and would interleave badly with anything else written there.
    """
    print(_load_spider(), file=file)
    if debug:
        print(f"  mpengine - '{task}' sequentially (debug mode)\n", file=file)
    else:
        print(f"  mpengine - '{task}' on {n_workers} workers\n", file=file)
