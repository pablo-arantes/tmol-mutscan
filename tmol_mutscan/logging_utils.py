"""Console and file logging shared by every subcommand."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "tmol_mutscan"


def setup_logging(verbosity: int = 0, log_file: Path | None = None) -> logging.Logger:
    """Configure the package logger.

    Args:
        verbosity: 0 for INFO, 1 or more for DEBUG.
        log_file: Optional file that also receives every record at DEBUG.

    Returns:
        The configured package logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbosity else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the package logger."""
    return logging.getLogger(LOGGER_NAME)
