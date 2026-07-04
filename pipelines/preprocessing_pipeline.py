"""Phase 5 orchestrator: stratified split, class mapping, DataLoader sanity check.

Run directly with:
    python pipelines\\preprocessing_pipeline.py

Note: unlike an earlier draft of this script, no call to
``resolve_dataset_directory`` (Phase 3, ``src/data/downloader.py``) is
needed here. Per ``docs/data_dictionary/metadata_schema.md``, the
``filepath`` column already stores each image's path relative to the
project root (including the ``data/raw/.../color/`` prefix), so joining it
with the project root is sufficient.
"""

import logging
import os
import sys

# --- sys.path bootstrap so `src` imports work regardless of the caller's cwd ---
_PROJECT_ROOT_FOR_IMPORTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT_FOR_IMPORTS not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_FOR_IMPORTS)

import pandas as pd

from src.config.config_loader import get_project_root, load_config
from src.data.dataloader import build_dataloaders
from src.data.splitter import (
    build_class_mapping,
    generate_split_report,
    save_class_mapping,
    save_splits,
    stratified_split,
)
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("preprocessing_pipeline")


def run_preprocessing_pipeline() -> None:
    """Run the full Phase 5 preprocessing pipeline end to end."""
    project_root = get_project_root()
    base_config = load_config("base")
    data_config = load_config("data")

    processed_dir = project_root / base_config["paths"]["data_processed_dir"]
    data_dictionary_dir = project_root / base_config["paths"]["docs_data_dictionary_dir"]

    clean_metadata_path = processed_dir / data_config["metadata"]["clean_output_filename"]
    metadata_df = pd.read_csv(clean_metadata_path)
    logger.info("Loaded clean metadata: %d rows", len(metadata_df))

    stratify_column = data_config["split"]["stratify_column"]
    class_to_idx = build_class_mapping(metadata_df, stratify_column)
    save_class_mapping(class_to_idx, processed_dir, data_config["split"]["class_mapping_filename"])

    train_df, val_df, test_df = stratified_split(metadata_df, data_config)
    save_splits(train_df, val_df, test_df, processed_dir, data_config)

    report_path = data_dictionary_dir / data_config["split"]["split_report_filename"]
    generate_split_report(
        train_df,
        val_df,
        test_df,
        stratify_column,
        data_config["split"]["min_samples_per_class_warning"],
        report_path,
    )

    # Sanity check: build dataloaders and log one batch's tensor shape.
    # `filepath` in the split CSVs is already relative to project_root,
    # so project_root itself is the correct base directory here.
    dataloaders = build_dataloaders(data_config, processed_dir, project_root)

    sample_images, sample_labels = next(iter(dataloaders["train"]))
    logger.info(
        "Sanity check batch: images=%s, labels=%s",
        tuple(sample_images.shape),
        tuple(sample_labels.shape),
    )
    logger.info("Phase 5 preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    run_preprocessing_pipeline()