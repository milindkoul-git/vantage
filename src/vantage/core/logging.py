"""Structured logging on the standard library.

Two formatters over one logging call style:

``console``
    Human-readable, for interactive development.
``json``
    One JSON object per line, for the log aggregation that a deployed
    multi-camera installation will need.

Structured fields travel in a single reserved ``extra`` key so that call sites
read identically regardless of formatter::

    log = get_logger(__name__)
    log.info("source opened", extra=fields(source_id="cam0", fps=30.0))

No third-party logging dependency: stdlib ``logging`` handles the fan-out,
levels and thread safety already, and one reserved key is all the structure
this platform needs.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

FIELDS_KEY = "vantage_fields"

_RESERVED = frozenset(
    [
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    ]
)


def fields(**kwargs: Any) -> dict[str, Any]:
    """Wrap structured fields for a logging call's ``extra=`` argument."""
    return {FIELDS_KEY: kwargs}


class ConsoleFormatter(logging.Formatter):
    """``HH:MM:SS LEVEL logger: message | key=value`` with fields appended."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = getattr(record, FIELDS_KEY, None)
        if extra:
            rendered = " ".join(f"{k}={_compact(v)}" for k, v in extra.items())
            base = f"{base} | {rendered}"
        return base


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with structured fields inlined at top level."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "thread": record.threadName,
        }
        extra = getattr(record, FIELDS_KEY, None)
        if extra:
            payload.update(extra)
        # Anything attached via extra= outside the reserved key is still emitted
        # rather than silently dropped.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != FIELDS_KEY and not key.startswith("_"):
                payload.setdefault(key, value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _compact(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    text = str(value)
    return f'"{text}"' if " " in text else text


def configure_logging(level: str = "INFO", fmt: str = "console", stream: Any = None) -> None:
    """Install the root handler. Idempotent - safe to call from tests and CLI."""
    formatter: logging.Formatter
    if fmt == "json":
        formatter = JsonFormatter()
    elif fmt == "console":
        formatter = ConsoleFormatter()
    else:
        raise ValueError(f"unknown log format {fmt!r}; expected 'console' or 'json'")

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger("vantage")
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())
    # The platform's logs are its own; don't duplicate them into a host app's root.
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger inside the ``vantage`` namespace."""
    if not name.startswith("vantage"):
        name = f"vantage.{name}"
    return logging.getLogger(name)
