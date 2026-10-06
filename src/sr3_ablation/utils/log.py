# ADDED: new file, not part of the upstream SR3 codebase.
"""Console and file logging for the ``sr3_ablation`` logger hierarchy."""

from __future__ import annotations

import logging
from pathlib import Path

ROOT_LOGGER = "sr3_ablation"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """(Re)configure the package logger; safe to call once per command."""
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(_FORMAT, datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")
