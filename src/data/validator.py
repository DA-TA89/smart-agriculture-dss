"""Data validation module - checks the raw PlantVillage dataset for integrity issues."""

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from PIL import Image, UnidentifiedImageError

from src.config.config_loader import get_project_root, load_config
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("data_validator")

HASH_READ_CHUNK_SIZE = 8192


@dataclass
class ValidationResult:
    """Container for the outcome of a dataset validation run."""

    total_images: int = 0
    corrupt_images: List[str] = field(default_factory=list)
    duplicate_images: List[str] = field(default_factory=list)
    undersized_images: List[str] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)


def _compute_file_hash(file_path: Path, algorithm: str) -> str:
    """Compute a hex digest hash for a file's contents.

    Args:
        file_path: Path to the file to hash.
        algorithm: Hash algorithm name supported by hashlib (e.g. "md5").

    Returns:
        str: Hex digest of the file contents.
    """
    hasher = hashlib.new(algorithm)
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(HASH_READ_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_image_readable(file_path: Path) -> bool:
    """Check whether an image file can be opened and verified by PIL.

    Args:
        file_path: Path to the image file.

    Returns:
        bool: True if the image is readable, False if corrupt.
    """
    try:
        with Image.open(file_path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def validate_dataset(dataset_dir: Path) -> ValidationResult:
    """Validate every image found under a dataset directory.

    Scans recursively for image files, checking file extension, minimum
    file size, corruption, and exact-content duplicates. Class labels are
    inferred from the immediate parent folder name of each image.

    Args:
        dataset_dir: Root directory containing class subfolders of images.

    Returns:
        ValidationResult: Aggregated validation findings.
    """
    dataset_dir = dataset_dir.resolve()
    data_config = load_config("data")["dataset"]
    allowed_extensions = {ext.lower() for ext in data_config["allowed_extensions"]}
    min_file_size_bytes = data_config["min_file_size_bytes"]
    hash_algorithm = data_config["duplicate_hash_algorithm"]

    result = ValidationResult()
    seen_hashes: Dict[str, str] = {}
    class_counts: Dict[str, int] = defaultdict(int)

    image_paths = [
        path for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]
    result.total_images = len(image_paths)
    logger.info("Found %d candidate images under %s", result.total_images, dataset_dir)

    for image_path in image_paths:
        class_name = image_path.parent.name
        class_counts[class_name] += 1

        if image_path.stat().st_size < min_file_size_bytes:
            result.undersized_images.append(str(image_path))
            continue

        if not _is_image_readable(image_path):
            result.corrupt_images.append(str(image_path))
            continue

        file_hash = _compute_file_hash(image_path, hash_algorithm)
        if file_hash in seen_hashes:
            result.duplicate_images.append(str(image_path))
        else:
            seen_hashes[file_hash] = str(image_path)

    result.class_distribution = dict(class_counts)
    logger.info(
        "Validation complete: %d corrupt, %d duplicates, %d undersized",
        len(result.corrupt_images), len(result.duplicate_images), len(result.undersized_images),
    )
    return result


def write_validation_report(result: ValidationResult, dataset_dir: Path) -> Path:
    """Write a Markdown validation report summarizing the dataset findings.

    Args:
        result: The validation result to report on.
        dataset_dir: The dataset directory that was validated (for reference).

    Returns:
        Path: Path to the written report file.
    """
    base_config = load_config("base")["paths"]
    data_config = load_config("data")
    report_filename = data_config["validation"]["report_filename"]
    sample_limit = data_config["validation"]["sample_corrupt_log_limit"]

    project_root = get_project_root()
    report_dir = project_root / base_config["docs_data_dictionary_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / report_filename

    lines = [
        "# Data Validation Report",
        "",
        f"**Dataset directory:** `{dataset_dir}`",
        f"**Total images scanned:** {result.total_images}",
        f"**Corrupt images:** {len(result.corrupt_images)}",
        f"**Duplicate images:** {len(result.duplicate_images)}",
        f"**Undersized images:** {len(result.undersized_images)}",
        "",
        "## Class Distribution",
        "",
        "| Class | Image Count |",
        "|---|---|",
    ]
    for class_name, count in sorted(result.class_distribution.items()):
        lines.append(f"| {class_name} | {count} |")

    lines.append("")
    lines.append("## Sample Corrupt Images")
    lines.extend(f"- {path}" for path in result.corrupt_images[:sample_limit])

    lines.append("")
    lines.append("## Sample Duplicate Images")
    lines.extend(f"- {path}" for path in result.duplicate_images[:sample_limit])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Validation report written to %s", report_path)
    return report_path