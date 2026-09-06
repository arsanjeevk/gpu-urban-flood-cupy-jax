"""Logging setup shared by the CLI and the Python API.

The package logs everything under the ``gurugram_flood`` logger namespace;
library code never calls ``basicConfig`` so an embedding application keeps
full control. :func:`setup_logging` is a convenience for the CLI and for
scripts that just want sensible console (and optional file) output.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "gurugram_flood"

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(child: str | None = None) -> logging.Logger:
    """Return the package logger, or a child logger like ``gurugram_flood.solver``."""
    return logging.getLogger(f"{LOGGER_NAME}.{child}" if child else LOGGER_NAME)


def setup_logging(level: int | str = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Attach console (and optional file) handlers to the package logger.

    Safe to call more than once: existing handlers installed by this
    function are replaced, not duplicated.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(file_handler)

    return logger
