# ADDED: new file, not part of the upstream SR3 codebase.
"""Console and file logging for the ``sr3_ablation`` logger hierarchy."""  # ADDED

from __future__ import annotations  # ADDED

import logging  # ADDED
from pathlib import Path  # ADDED

ROOT_LOGGER = "sr3_ablation"  # ADDED
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"  # ADDED


def configure_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:  # ADDED
    """(Re)configure the package logger; safe to call once per command."""  # ADDED
    logger = logging.getLogger(ROOT_LOGGER)  # ADDED
    logger.setLevel(level)  # ADDED
    logger.propagate = False  # ADDED
    for handler in list(logger.handlers):  # ADDED
        logger.removeHandler(handler)  # ADDED
        handler.close()  # ADDED
    formatter = logging.Formatter(_FORMAT, datefmt="%H:%M:%S")  # ADDED
    console = logging.StreamHandler()  # ADDED
    console.setFormatter(formatter)  # ADDED
    logger.addHandler(console)  # ADDED
    if log_file is not None:  # ADDED
        log_file.parent.mkdir(parents=True, exist_ok=True)  # ADDED
        file_handler = logging.FileHandler(log_file, encoding="utf-8")  # ADDED
        file_handler.setFormatter(formatter)  # ADDED
        logger.addHandler(file_handler)  # ADDED
    return logger  # ADDED


def get_logger(name: str) -> logging.Logger:  # ADDED
    """Child logger of the package's root logger."""  # ADDED
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")  # ADDED
