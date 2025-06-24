"""logging_config.py
Centralized logging configuration for the project.
This module must be imported before any other module creates loggers
so that the configuration applies globally.

The config is intentionally lightweight (std. out, time-stamped, single-line)
but can be swapped with JSON or external log aggregators easily.
"""
from __future__ import annotations

import logging
import os
from logging.config import dictConfig
from typing import Any, Dict

DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": LOG_FORMAT,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {"handlers": ["console"], "level": DEFAULT_LEVEL},
}

def setup_logging() -> None:
    """Apply the dictConfig so that it affects the entire interpreter."""
    dictConfig(LOGGING_CONFIG)


# Initialize immediately so importing this module has side-effects
setup_logging() 