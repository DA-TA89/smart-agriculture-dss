"""ETL pipeline entry point for Phase 3 - Data Engineering.

Run from the project root with the virtual environment activated:
    python pipelines/etl_pipeline.py
"""

import logging
import sys
from pathlib import Path

# Ensure the project root is importable when this script is run directly
# (e.g. `python pipelines\etl_pipeline.py`), since Python only adds the
# script's own folder to sys.path in that case, not the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.cleaner import clean_metadata, save_clean_metadata
from src.data.downloader import download_plantvillage_dataset, resolve_dataset_directory
from src.data.metadata_generator import generate_metadata, save_metadata
from src.data.validator import validate_dataset, write_validation_report
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("etl_pipeline")


def run_etl_pipeline() -> None:
    """Run the full Phase 3 ETL pipeline: download, validate, clean, index.

    Steps:
        1. Download the PlantVillage dataset from Kaggle into data/raw
           (skipped automatically if already present).
        2. Resolve the correct class-containing subfolder.
        3. Validate images for corruption, duplicates, and undersized files.
        4. Write a Markdown validation report to docs/data_dictionary.
        5. Generate a full metadata index and save to data/processed.
        6. Clean the metadata index and save the clean version.
    """
    logger.info("=== ETL Pipeline started ===")

    raw_dataset_dir = download_plantvillage_dataset()
    dataset_dir = resolve_dataset_directory(raw_dataset_dir)

    validation_result = validate_dataset(dataset_dir)
    write_validation_report(validation_result, dataset_dir)

    metadata_df = generate_metadata(dataset_dir, validation_result)
    save_metadata(metadata_df)

    clean_df = clean_metadata(metadata_df)
    save_clean_metadata(clean_df)

    logger.info("=== ETL Pipeline completed successfully ===")


if __name__ == "__main__":
    run_etl_pipeline()