"""
Shared structured logging setup using structlog.

Every service calls configure_logging() once in its lifespan startup,
then get_logger(__name__) wherever it needs a logger. JSON output in
production, colourised console output in DEBUG mode.

Usage in a service:
    from shared.logging import configure_logging, get_logger

    configure_logging(debug=settings.DEBUG)
    logger = get_logger(__name__)
"""
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(*, debug: bool = False) -> None:
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer = structlog.dev.ConsoleRenderer() if debug else structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if debug else logging.INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)