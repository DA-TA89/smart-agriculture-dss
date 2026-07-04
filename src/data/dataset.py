"""PyTorch Dataset for the PlantVillage leaf-image classification task.

Reads image paths and class labels from a split metadata CSV (produced by
``src/data/splitter.py``) and returns transformed image tensors ready to be
consumed by a DataLoader.

Per ``docs/data_dictionary/metadata_schema.md``, the ``filepath`` column is
a path *relative to the project root* (it already includes
``data/raw/.../color/...``), so images are located by joining it directly
with the project root — no dataset-subfolder resolution is needed here.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, Tuple

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("plantvillage_dataset")


class PlantVillageDataset(Dataset):
    """Leaf-image dataset backed by a split metadata CSV file."""

    def __init__(
        self,
        metadata_csv_path: Path,
        class_to_idx: Dict[str, int],
        project_root_dir: Path,
        transform: Callable,
        filepath_column: str = "filepath",
        class_column: str = "class_name",
    ) -> None:
        """Initialize the dataset.

        Args:
            metadata_csv_path: Path to a split CSV (train/val/test) that
                contains at least the filepath and class columns.
            class_to_idx: Mapping from class name to integer label.
            project_root_dir: Absolute path to the project root. Relative
                ``filepath`` values are resolved against this directory.
            transform: torchvision transform pipeline applied to each image.
            filepath_column: Name of the column holding the image's
                project-root-relative path.
            class_column: Name of the column holding the class label.
        """
        self.metadata_df: pd.DataFrame = pd.read_csv(metadata_csv_path)
        self.class_to_idx = class_to_idx
        self.project_root_dir = Path(project_root_dir)
        self.transform = transform
        self.filepath_column = filepath_column
        self.class_column = class_column
        logger.info(
            "Loaded dataset from %s (%d rows)", metadata_csv_path, len(self.metadata_df)
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.metadata_df)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """Load and transform a single sample.

        Args:
            index: Positional row index within the split metadata.

        Returns:
            Tuple of ``(transformed image tensor, integer class label)``.
        """
        row = self.metadata_df.iloc[index]
        relative_path = Path(row[self.filepath_column])
        image_path = (
            relative_path if relative_path.is_absolute() else self.project_root_dir / relative_path
        )

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        image_tensor = self.transform(image)
        label = self.class_to_idx[row[self.class_column]]
        return image_tensor, label

    def get_class_sample_counts(self) -> Dict[str, int]:
        """Return the number of samples per class in this split.

        Used by ``src/data/dataloader.py`` to build a
        ``WeightedRandomSampler`` that counteracts class imbalance.

        Returns:
            Dictionary mapping class name to sample count.
        """
        return self.metadata_df[self.class_column].value_counts().to_dict()