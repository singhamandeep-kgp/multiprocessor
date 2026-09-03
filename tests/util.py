"""Small helpers shared across the test modules."""

from __future__ import annotations

import logging


def log_text(caplog) -> str:
    """All captured log records as one formatted string.

    `record.message` only exists once a formatter has run, so a caplog
    assertion has to call `getMessage()` itself to see the interpolated
    lifecycle lines mpengine emits with %-style arguments.
    """
    return "\n".join(r.getMessage() for r in caplog.records)


def records_at(caplog, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == level]
