from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(*, level: str, json_output: bool, logger_name: str) -> None:
    """Configure stdlib + structlog for consistent structured logs."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=log_level, format="%(message)s")

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger(logger_name)
    logger.info("logging_configured", level=level.upper(), json_output=json_output)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
