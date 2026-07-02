"""Centralized logging setup for the Smart Agriculture DSS project."""

import logging
from datetime import datetime

from src.config.config_loader import get_project_root, load_config

DATE_FORMAT_FOR_FILENAME = "%Y-%m-%d"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DIR = "logs/system"


def setup_logger(module_name: str) -> logging.Logger:
    """Create and configure a logger for a given module.

    Reads settings from configs/logging.yaml (level, format, output
    directory). Writes logs to ``logs/system/{date}_{module}.log`` and
    optionally echoes to the console.

    Args:
        module_name: Name of the module requesting the logger. Used as
            both the logger name and part of the log filename.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging_config = load_config("logging")
    log_level_name = logging_config.get("level", DEFAULT_LOG_LEVEL)
    log_format = logging_config.get("format", DEFAULT_LOG_FORMAT)
    date_format = logging_config.get("date_format", DEFAULT_DATE_FORMAT)
    log_dir_relative = logging_config.get("log_dir", DEFAULT_LOG_DIR)
    console_output = logging_config.get("console_output", True)

    project_root = get_project_root()
    log_dir = project_root / log_dir_relative
    log_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime(DATE_FORMAT_FOR_FILENAME)
    log_file_path = log_dir / f"{today_str}_{module_name}.log"

    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, log_level_name, logging.INFO))

    if not logger.handlers:
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger