"""Logging configuration for xpyd-acc."""

from __future__ import annotations

import logging
import sys

_PACKAGE_LOGGER = "xpyd_acc"


def setup_logging(verbosity: int = 0) -> None:
    """Configure logging based on verbosity level.

    Args:
        verbosity: -1 = quiet (ERROR), 0 = default (WARNING),
                   1 = verbose (INFO), 2+ = debug (DEBUG).
    """
    if verbosity <= -1:
        level = logging.ERROR
    elif verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )

    logger = logging.getLogger(_PACKAGE_LOGGER)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the package namespace."""
    return logging.getLogger(f"{_PACKAGE_LOGGER}.{name}")
