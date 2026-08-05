import logging
import sys
from pathlib import Path


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int):
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def init_logger(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure application-wide logging with accumulating file handler."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    pkg_logger = logging.getLogger("pytorch_pipeline")
    pkg_logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if init_logger is called multiple times
    if not pkg_logger.handlers:
        # Accumulating file handler (mode="a")
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        pkg_logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.addFilter(MaxLevelFilter(logging.WARNING))
        console_handler.setFormatter(console_formatter)
        pkg_logger.addHandler(console_handler)

    pkg_logger.propagate = False
    return pkg_logger
