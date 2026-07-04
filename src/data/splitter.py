"""Stratified train/validation/test splitting for the PlantVillage metadata.

Reads the cleaned metadata produced in Phase 3 (``data/processed/metadata_clean.csv``)
and splits it into three stratified subsets so class proportions are
preserved across train, validation, and test sets. Per SDS Section 6, the
``data`` module owns dataset-reading and indexing responsibilities; it does
not train models.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("data_splitter")


def build_class_mapping(metadata_df: pd.DataFrame, stratify_column: str) -> Dict[str, int]:
    """Build a deterministic class-name-to-index mapping.

    Classes are sorted alphabetically so the mapping is reproducible across
    runs and machines.

    Args:
        metadata_df: Metadata DataFrame containing the class column.
        stratify_column: Name of the column holding class labels.

    Returns:
        Dictionary mapping each class name to an integer index.
    """
    class_names = sorted(metadata_df[stratify_column].unique().tolist())
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    logger.info("Built class mapping for %d classes", len(class_to_idx))
    return class_to_idx


def stratified_split(
    metadata_df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split metadata into stratified train/validation/test subsets.

    Args:
        metadata_df: Cleaned metadata DataFrame (output of Phase 3).
        config: Parsed content of ``configs/data.yaml``.

    Returns:
        Tuple of ``(train_df, val_df, test_df)``.

    Raises:
        ValueError: If the configured ratios do not sum to 1.0.
    """
    split_cfg = config["split"]
    stratify_column = split_cfg["stratify_column"]
    train_ratio = split_cfg["train_ratio"]
    val_ratio = split_cfg["val_ratio"]
    test_ratio = split_cfg["test_ratio"]
    random_seed = split_cfg["random_seed"]

    ratio_sum = round(train_ratio + val_ratio + test_ratio, 6)
    if ratio_sum != 1.0:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {ratio_sum}"
        )

    train_val_df, test_df = train_test_split(
        metadata_df,
        test_size=test_ratio,
        stratify=metadata_df[stratify_column],
        random_state=random_seed,
    )

    val_size_within_remaining = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_within_remaining,
        stratify=train_val_df[stratify_column],
        random_state=random_seed,
    )

    logger.info(
        "Stratified split done: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def save_class_mapping(class_to_idx: Dict[str, int], output_dir: Path, filename: str) -> Path:
    """Persist the class-to-index mapping as JSON.

    Args:
        class_to_idx: Mapping produced by ``build_class_mapping``.
        output_dir: Directory where the JSON file will be written
            (typically ``data/processed``).
        filename: Name of the JSON file.

    Returns:
        Full path of the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(class_to_idx, file_handle, ensure_ascii=False, indent=2)
    logger.info("Saved class mapping to %s", output_path)
    return output_path


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    config: Dict[str, Any],
) -> Dict[str, Path]:
    """Save the three split DataFrames to CSV files.

    Args:
        train_df: Training subset.
        val_df: Validation subset.
        test_df: Test subset.
        output_dir: Directory where CSV files are written (e.g. ``data/processed``).
        config: Parsed content of ``configs/data.yaml``.

    Returns:
        Dictionary mapping split name (``train``/``val``/``test``) to the
        full path of its CSV file.
    """
    split_cfg = config["split"]
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        "train": split_cfg["train_metadata_filename"],
        "val": split_cfg["val_metadata_filename"],
        "test": split_cfg["test_metadata_filename"],
    }
    dataframes = {"train": train_df, "val": val_df, "test": test_df}

    paths: Dict[str, Path] = {}
    for split_name, filename in filenames.items():
        output_path = output_dir / filename
        dataframes[split_name].to_csv(output_path, index=False)
        paths[split_name] = output_path
        logger.info(
            "Saved %s split (%d rows) to %s", split_name, len(dataframes[split_name]), output_path
        )

    return paths


def generate_split_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stratify_column: str,
    min_samples_per_class_warning: int,
    output_path: Path,
) -> Path:
    """Generate a Markdown report summarizing the split results.

    Verifies every class is represented in all three subsets and flags
    class/split combinations whose count falls below the configured
    warning threshold (relevant given the 36.23x imbalance found in Phase 4).

    Args:
        train_df: Training subset.
        val_df: Validation subset.
        test_df: Test subset.
        stratify_column: Name of the class column.
        min_samples_per_class_warning: Threshold below which a class/split
            combination is flagged in the report.
        output_path: Full path of the Markdown file to write.

    Returns:
        The full path of the written report.
    """
    train_counts = train_df[stratify_column].value_counts()
    val_counts = val_df[stratify_column].value_counts()
    test_counts = test_df[stratify_column].value_counts()

    all_classes = sorted(set(train_counts.index) | set(val_counts.index) | set(test_counts.index))

    lines = [
        "# Data Split Report",
        "",
        f"- Total classes: {len(all_classes)}",
        f"- Train rows: {len(train_df)}",
        f"- Validation rows: {len(val_df)}",
        f"- Test rows: {len(test_df)}",
        "",
        "## Per-class Counts",
        "",
        "| class_name | train | val | test |",
        "|---|---|---|---|",
    ]

    warnings = []
    for class_name in all_classes:
        train_count = int(train_counts.get(class_name, 0))
        val_count = int(val_counts.get(class_name, 0))
        test_count = int(test_counts.get(class_name, 0))
        lines.append(f"| {class_name} | {train_count} | {val_count} | {test_count} |")

        if val_count < min_samples_per_class_warning or test_count < min_samples_per_class_warning:
            warnings.append(
                f"- `{class_name}`: val={val_count}, test={test_count} "
                f"(below warning threshold {min_samples_per_class_warning})"
            )

    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    lines.extend(warnings if warnings else ["None. Every class meets the minimum threshold."])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Saved split report to %s", output_path)
    return output_path