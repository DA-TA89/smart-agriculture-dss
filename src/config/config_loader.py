"""Configuration loading utilities for the Smart Agriculture DSS project."""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_DIR_NAME = "configs"
YAML_EXTENSION = ".yaml"

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    The project root is located by walking up from this file until a
    directory containing the ``configs/`` folder is found.

    Returns:
        Path: Absolute path to the project root.

    Raises:
        FileNotFoundError: If the project root cannot be located.
    """
    current_dir = Path(__file__).resolve().parent
    for candidate in [current_dir, *current_dir.parents]:
        if (candidate / CONFIG_DIR_NAME).is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root containing a 'configs/' folder."
    )


def load_config(config_name: str) -> Dict[str, Any]:
    """Load a YAML configuration file from the configs/ directory.

    Args:
        config_name: Name of the config file without extension (e.g. "data").

    Returns:
        Dict[str, Any]: Parsed configuration content.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = get_project_root() / CONFIG_DIR_NAME / f"{config_name}{YAML_EXTENSION}"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file_handle:
        config_data = yaml.safe_load(file_handle) or {}
    logger.debug("Loaded config '%s' from %s", config_name, config_path)
    return config_data