"""
Shared structured logging setup using structlog.

Every service calls configure_logging() once in its lifespan startup.
JSON in production, colourised console in DEBUG mode.
"""
# services/ingestion/core/logging.py

from shared.shared.logging import (
    configure_logging,
    get_logger,
)

__all__ = ["configure_logging", "get_logger"]

