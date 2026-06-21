"""
services/preprocessing/core/logging.py

Three-line re-export, identical pattern to ingestion's core/logging.py.
Keeps service code importing from `services.preprocessing.core.logging`
rather than reaching into `shared` directly.
"""
from shared.logging import configure_logging, get_logger

__all__ = ["configure_logging", "get_logger"]
