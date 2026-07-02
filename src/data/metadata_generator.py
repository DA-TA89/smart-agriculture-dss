"""Metadata generation module - builds a structured index of the dataset."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from PIL import Image

from src.config.config_loader import get_project_root, load_config
from src.data.validator import ValidationResult
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("metadata_generator")


def _parse_class_name(
    class_name: str, separator: str, healthy_keyword: str
) -> Tuple[str, str, bool]:
    """Split a PlantVillage class folder name into plant type and disease.

    Args:
        class_name: Raw folder name, e.g. "Tomato___Bacterial_spot".
        separator: Separator token between plant type and disease label.
        healthy_keyword: Keyword identifying a healthy sample.

    Returns:
        Tuple[str, str, bool]: (plant_type, disease_name, is_healthy)
    """
    if separator in class_name:
        plant_type, disease_name = class_name.split(separator, maxsplit=1)
    else:
        plant_type, disease_name = class_name, "unknown"
    is_healthy = healthy_keyword.lower() in disease_name.lower()
    return plant_type, disease_name, is_healthy


def generate_metadata(dataset_dir: Path, validation_result: ValidationResult) -> pd.DataFrame:
    """Build a metadata DataFrame describing every image in the dataset.

    Args:
        dataset_dir: Root directory containing class subfolders of images.
        validation_result: Prior validation findings, used to flag corrupt
            and duplicate images.

    Returns:
        pd.DataFrame: One row per image with descriptive metadata columns.
    """
    dataset_dir = dataset_dir.resolve()
    data_config = load_config("data")["dataset"]
    allowed_extensions = {ext.lower() for ext in data_config["allowed_extensions"]}
    separator = data_config["class_separator"]
    healthy_keyword = data_config["healthy_keyword"]

    corrupt_set = set(validation_result.corrupt_images)
    duplicate_set = set(validation_result.duplicate_images)
    project_root = get_project_root()

    records = []
    image_paths = sorted(
        path for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_extensions
    )

    for image_id, image_path in enumerate(image_paths, start=1):
        class_name = image_path.parent.name
        plant_type, disease_name, is_healthy = _parse_class_name(
            class_name, separator, healthy_keyword
        )
        is_corrupt = str(image_path) in corrupt_set
        is_duplicate = str(image_path) in duplicate_set

        width: Optional[int] = None
        height: Optional[int] = None
        image_format: Optional[str] = None
        if not is_corrupt:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image_format = image.format
            except OSError:
                is_corrupt = True

        records.append({
            "image_id": image_id,
            "filepath": str(image_path.relative_to(project_root)),
            "class_name": class_name,
            "plant_type": plant_type,
            "disease_name": disease_name,
            "is_healthy": is_healthy,
            "width": width,
            "height": height,
            "image_format": image_format,
            "file_size_bytes": image_path.stat().st_size,
            "is_corrupt": is_corrupt,
            "is_duplicate": is_duplicate,
        })

    metadata_df = pd.DataFrame.from_records(records)
    logger.info("Generated metadata for %d images", len(metadata_df))
    return metadata_df


def save_metadata(metadata_df: pd.DataFrame, filename_key: str = "output_filename") -> Path:
    """Persist the metadata DataFrame to data/processed as CSV.

    Args:
        metadata_df: The metadata DataFrame to save.
        filename_key: Key in configs/data.yaml -> metadata section that
            holds the target filename.

    Returns:
        Path: Path to the saved CSV file.
    """
    base_config = load_config("base")["paths"]
    data_config = load_config("data")["metadata"]
    filename = data_config[filename_key]

    project_root = get_project_root()
    processed_dir = project_root / base_config["data_processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / filename

    metadata_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Metadata saved to %s", output_path)
    return output_path