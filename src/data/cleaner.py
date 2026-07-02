"""Data cleaning module - filters invalid records out of the metadata index."""

import logging
from pathlib import Path

import pandas as pd

from src.data.metadata_generator import save_metadata
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("data_cleaner")


def clean_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Remove corrupt and duplicate entries from the metadata DataFrame.

    Note: this does not delete files from data/raw. Raw data remains
    immutable; only the metadata index used for downstream processing
    is filtered.

    Args:
        metadata_df: Full metadata DataFrame including flagged rows.

    Returns:
        pd.DataFrame: Cleaned metadata containing only valid, unique images.
    """
    before_count = len(metadata_df)
    clean_df = metadata_df[
        (~metadata_df["is_corrupt"]) & (~metadata_df["is_duplicate"])
    ].reset_index(drop=True)
    removed_count = before_count - len(clean_df)

    logger.info("Removed %d invalid/duplicate rows out of %d", removed_count, before_count)
    return clean_df


def save_clean_metadata(clean_df: pd.DataFrame) -> Path:
    """Persist the cleaned metadata DataFrame to data/processed as CSV.

    Args:
        clean_df: Cleaned metadata DataFrame.

    Returns:
        Path: Path to the saved CSV file.
    """
    return save_metadata(clean_df, filename_key="clean_output_filename")