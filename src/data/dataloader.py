"""DataLoader factory for the PlantVillage classification task.

Builds PyTorch DataLoaders for the train/validation/test splits produced by
``src/data/splitter.py``, wiring together the Dataset
(``src/data/dataset.py``) and the transform pipelines
(``src/features/transforms.py``). A ``WeightedRandomSampler`` is optionally
applied to the training loader to counteract the ~36x class imbalance
identified in Phase 4.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import PlantVillageDataset
from src.features.transforms import build_eval_transform, build_train_transform
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("dataloader_builder")


def _load_class_mapping(processed_dir: Path, filename: str) -> Dict[str, int]:
    """Load the class-to-index mapping saved by the splitter.

    Args:
        processed_dir: Directory containing the processed metadata artifacts.
        filename: Name of the JSON mapping file.

    Returns:
        Dictionary mapping class name to integer label.
    """
    mapping_path = processed_dir / filename
    with mapping_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _build_weighted_sampler(
    dataset: PlantVillageDataset, class_to_idx: Dict[str, int]
) -> WeightedRandomSampler:
    """Build a sampler that inversely weights frequent classes.

    Args:
        dataset: Training dataset instance.
        class_to_idx: Class-name-to-index mapping.

    Returns:
        Configured ``WeightedRandomSampler`` for the training DataLoader.
    """
    class_counts = dataset.get_class_sample_counts()
    idx_to_count = {class_to_idx[name]: count for name, count in class_counts.items()}

    labels = dataset.metadata_df[dataset.class_column].map(class_to_idx)
    sample_weights = [1.0 / idx_to_count[label] for label in labels]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(
    config: Dict[str, Any], processed_dir: Path, project_root_dir: Path
) -> Dict[str, DataLoader]:
    """Build train/validation/test DataLoaders from split metadata.

    Args:
        config: Parsed content of ``configs/data.yaml``.
        processed_dir: Directory holding split CSVs and the class mapping
            (typically ``data/processed``).
        project_root_dir: Absolute path to the project root. The
            ``filepath`` column in each split CSV is relative to this
            directory (per ``metadata_schema.md``), so no separate
            dataset-subfolder resolution is required here.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"`` mapping to
        ``DataLoader`` instances.
    """
    split_cfg = config["split"]
    metadata_cfg = config["metadata"]
    dataloader_cfg = config["dataloader"]
    augmentation_cfg = config["augmentation"]

    class_to_idx = _load_class_mapping(processed_dir, split_cfg["class_mapping_filename"])

    train_transform = build_train_transform(config)
    eval_transform = build_eval_transform(config)

    filepath_column = metadata_cfg["filepath_column"]
    class_column = split_cfg["stratify_column"]

    split_files = {
        "train": (split_cfg["train_metadata_filename"], train_transform),
        "val": (split_cfg["val_metadata_filename"], eval_transform),
        "test": (split_cfg["test_metadata_filename"], eval_transform),
    }

    datasets: Dict[str, PlantVillageDataset] = {}
    for split_name, (filename, transform) in split_files.items():
        csv_path = processed_dir / filename
        datasets[split_name] = PlantVillageDataset(
            metadata_csv_path=csv_path,
            class_to_idx=class_to_idx,
            project_root_dir=project_root_dir,
            transform=transform,
            filepath_column=filepath_column,
            class_column=class_column,
        )

    train_sampler = None
    if augmentation_cfg["use_weighted_sampler"]:
        train_sampler = _build_weighted_sampler(datasets["train"], class_to_idx)

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=dataloader_cfg["batch_size"],
            sampler=train_sampler,
            shuffle=dataloader_cfg["shuffle_train"] if train_sampler is None else False,
            num_workers=dataloader_cfg["num_workers"],
            pin_memory=dataloader_cfg["pin_memory"],
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=dataloader_cfg["batch_size"],
            shuffle=False,
            num_workers=dataloader_cfg["num_workers"],
            pin_memory=dataloader_cfg["pin_memory"],
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=dataloader_cfg["batch_size"],
            shuffle=False,
            num_workers=dataloader_cfg["num_workers"],
            pin_memory=dataloader_cfg["pin_memory"],
        ),
    }

    logger.info(
        "Built dataloaders: train=%d batches, val=%d batches, test=%d batches",
        len(dataloaders["train"]),
        len(dataloaders["val"]),
        len(dataloaders["test"]),
    )
    return dataloaders