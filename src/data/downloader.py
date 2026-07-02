"""Dataset download module - retrieves the PlantVillage dataset from Kaggle.

kagglehub is only imported lazily, inside download_plantvillage_dataset(),
and only when no dataset is already present under data/raw. This means
users who manually download and extract the dataset from the Kaggle
website never need to install kagglehub or configure Kaggle API
credentials at all.
"""

import logging
import shutil
from pathlib import Path

from src.config.config_loader import get_project_root, load_config
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("data_downloader")


def download_plantvillage_dataset() -> Path:
    """Ensure the PlantVillage dataset is available under data/raw.

    Behavior:
        - If data/raw already contains a dataset folder (e.g. manually
          downloaded and extracted from the Kaggle website), that folder
          is returned as-is and nothing is downloaded.
        - Otherwise, falls back to auto-downloading via kagglehub, using
          the handle defined in configs/data.yaml. This requires the
          kagglehub package and Kaggle API credentials (KAGGLE_USERNAME /
          KAGGLE_KEY environment variables, or ~/.kaggle/kaggle.json).

    Returns:
        Path: Absolute path to the dataset folder inside data/raw.

    Raises:
        RuntimeError: If no local dataset is found and kagglehub is
            unavailable or the download fails.
    """
    data_config = load_config("data")["dataset"]
    base_config = load_config("base")["paths"]
    kaggle_handle = data_config["kaggle_handle"]
    raw_dir_relative = base_config["data_raw_dir"]

    project_root = get_project_root()
    raw_dir = project_root / raw_dir_relative
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing_subfolders = [path for path in raw_dir.iterdir() if path.is_dir()]
    if existing_subfolders:
        logger.info(
            "Dataset already present at %s, skipping Kaggle download", existing_subfolders[0]
        )
        return existing_subfolders[0]

    logger.info("No local dataset found under %s, attempting kagglehub download", raw_dir)
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "No dataset found under data/raw and kagglehub is not installed. "
            "Either 'pip install kagglehub==0.3.13' to auto-download, or manually "
            "download and extract the dataset into data/raw/ from "
            f"https://www.kaggle.com/datasets/{kaggle_handle}"
        ) from error

    cache_path_str = kagglehub.dataset_download(kaggle_handle)
    cache_path = Path(cache_path_str)

    if not cache_path.exists():
        raise RuntimeError(f"Kaggle download failed, path does not exist: {cache_path}")

    destination_path = raw_dir / cache_path.name
    logger.info("Copying dataset from cache %s to %s", cache_path, destination_path)
    shutil.copytree(cache_path, destination_path)
    logger.info("Dataset ready at %s", destination_path)

    return destination_path


def resolve_dataset_directory(dataset_dir: Path) -> Path:
    """Resolve the directory that directly contains class subfolders.

    Some Kaggle archives nest multiple image variants (e.g. "color",
    "grayscale", "segmented") under the dataset root. If
    configs/data.yaml sets `dataset.dataset_subfolder`, this appends it
    to the given path; otherwise the path is returned unchanged.

    Args:
        dataset_dir: Path returned by download_plantvillage_dataset().

    Returns:
        Path: Directory that directly contains per-class image folders.

    Raises:
        FileNotFoundError: If the configured subfolder does not exist.
    """
    data_config = load_config("data")["dataset"]
    subfolder = data_config.get("dataset_subfolder")

    if not subfolder:
        return dataset_dir

    resolved_path = dataset_dir / subfolder
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Configured dataset_subfolder '{subfolder}' not found under {dataset_dir}. "
            "Inspect the downloaded folder and update configs/data.yaml."
        )
    logger.info("Using configured dataset subfolder: %s", resolved_path)
    return resolved_path