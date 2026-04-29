import logging
import sys
from pathlib import Path


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int):
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def init_logger(log_file: Path, level: int = logging.DEBUG) -> logging.Logger:
    """Configure application-wide logging."""

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(MaxLevelFilter(logging.WARNING))  # INFO and WARNING only
    console_handler.setFormatter(console_formatter)

    # Root logger
    rootlogger = logging.getLogger("inat_pipeline")
    rootlogger.setLevel(logging.DEBUG)
    rootlogger.addHandler(file_handler)
    rootlogger.addHandler(console_handler)
    rootlogger.propagate = False  # don't bubble up to the root logger's StreamHandler

    return rootlogger
