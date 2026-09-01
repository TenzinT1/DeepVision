"""Structured logging for DeepVision.

A small, dependency-free structured logger every module imports. It emits either
human-readable lines (default) or JSON (set ``DEEPVISION_LOG_JSON=1``), and lets
ingestion attach ``paper_id`` / ``job_id`` / ``stage`` context so pipeline logs
can be correlated and streamed back to the ingestion modal UI.

This module is fully implemented (shared utility, not builder-owned).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, MutableMapping

__all__ = ["get_logger", "configure_logging", "LoggerAdapter"]

_CONFIGURED = False
_DEFAULT_LEVEL = os.getenv("DEEPVISION_LOG_LEVEL", "INFO").upper()
_JSON = os.getenv("DEEPVISION_LOG_JSON", "").strip() in {"1", "true", "yes"}

# Reserved LogRecord attributes; anything else attached to a record is treated
# as structured context and rendered alongside the message.
_RESERVED = set(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime", "taskName"}


class _HumanFormatter(logging.Formatter):
    """Render ``time level logger: message [k=v ...]``."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v for k, v in record.__dict__.items() if k not in _RESERVED
        }
        if extras:
            kv = " ".join(f"{k}={v}" for k, v in extras.items())
            return f"{base} [{kv}]"
        return base


class _JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object (one line)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    """Install the DeepVision root handler exactly once.

    Idempotent: safe to call from any entrypoint (API startup, worker, tests).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    if _JSON:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            _HumanFormatter(
                fmt="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root = logging.getLogger("deepvision")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level or _DEFAULT_LEVEL)
    root.propagate = False
    _CONFIGURED = True


#: Keyword arguments the stdlib logging call sites accept directly; anything
#: else passed to a logging method is treated as structured context and folded
#: into ``extra`` (the whole point of this adapter — callers use the
#: ``log.warning("msg", key=value)`` idiom throughout the codebase).
_LOGGING_KWARGS = frozenset({"exc_info", "stack_info", "stacklevel", "extra"})


class LoggerAdapter(logging.LoggerAdapter):
    """A logging adapter that merges bound context into every record."""

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        extra = dict(self.extra or {})
        extra.update(kwargs.pop("extra", None) or {})
        # Fold any non-standard keyword args (e.g. ``paper_id=...``,
        # ``error=...``) into structured context so ``log.warning("msg",
        # key=value)`` works without raising ``TypeError`` in the stdlib logger.
        for key in [k for k in kwargs if k not in _LOGGING_KWARGS]:
            extra[key] = kwargs.pop(key)
        kwargs["extra"] = extra
        return msg, kwargs

    def bind(self, **context: Any) -> "LoggerAdapter":
        """Return a new adapter with additional bound ``key=value`` context."""
        merged = dict(self.extra or {})
        merged.update(context)
        return LoggerAdapter(self.logger, merged)


def get_logger(name: str, **context: Any) -> LoggerAdapter:
    """Return a structured logger under the ``deepvision`` namespace.

    Args:
        name: module name, typically ``__name__``.
        **context: optional bound fields (e.g. ``paper_id=..., job_id=...``)
            that will be attached to every record from the returned logger.
    """
    configure_logging()
    logger_name = name if name.startswith("deepvision") else f"deepvision.{name}"
    return LoggerAdapter(logging.getLogger(logger_name), dict(context))
