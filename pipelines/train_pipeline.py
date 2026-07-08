"""Training pipeline orchestrator for EfficientNet-B0 and ViT-Tiny classifiers."""

import json
import sys
import time
from pathlib import Path

# --- sys.path bootstrap (bài học từ Phase 3, xem PHASE3_LOG.md muc 3.1) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.amp import GradScaler

from src.config.config_loader import get_project_root, load_config
from src.data.dataloader import build_dataloaders
from src.models.model_factory import build_model, get_checkpoint_prefix
from src.training.checkpoint_manager import save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.optimizer_factory import build_loss_function, build_optimizer
from src.training.trainer import train_one_epoch, validate_one_epoch
from src.utils.logger import setup_logger

logger = setup_logger("train_pipeline")

MODEL_KEYS = ("efficientnet", "vit")


def _resolve_device(preferred_device: str) -> torch.device:
    """Resolves the compute device, falling back to CPU if CUDA is unavailable.

    Args:
        preferred_device: Preferred device string from config (e.g. "cuda").

    Returns:
        A valid torch.device.
    """
    if preferred_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(preferred_device)


def _load_num_classes(base_config: dict, data_config: dict) -> int:
    """Loads the number of classes from the Phase 5 class_to_idx.json mapping.

    Args:
        base_config: Parsed contents of configs/base.yaml.
        data_config: Parsed contents of configs/data.yaml.

    Returns:
        Number of distinct disease classes.
    """
    mapping_path = (
        get_project_root()
        / base_config["paths"]["data_processed_dir"]
        / data_config["split"]["class_mapping_filename"]
    )
    with open(mapping_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    return len(class_to_idx)


def _build_dataloaders(base_config: dict, data_config: dict) -> dict:
    """Builds train/val/test DataLoaders via src.data.dataloader.build_dataloaders.

    Args:
        base_config: Parsed contents of configs/base.yaml.
        data_config: Parsed contents of configs/data.yaml.

    Returns:
        Dict with keys "train", "val", "test" mapping to DataLoader instances.
    """
    project_root = get_project_root()
    processed_dir = project_root / base_config["paths"]["data_processed_dir"]
    return build_dataloaders(data_config, processed_dir, project_root)


def run_training_for_model(
    model_key: str,
    num_classes: int,
    dataloaders: dict,
    training_config: dict,
    device: torch.device,
) -> None:
    """Trains a single model end-to-end and saves the best checkpoint.

    Args:
        model_key: One of "efficientnet" or "vit".
        num_classes: Number of output classes.
        dataloaders: Dict with "train" and "val" DataLoader objects.
        training_config: Parsed contents of configs/training.yaml -> "training".
        device: Device to train on.
    """
    logger.info("=== Starting training for model: %s ===", model_key)
    model = build_model(model_key, num_classes).to(device)
    checkpoint_prefix = get_checkpoint_prefix(model_key)

    optimizer = build_optimizer(
        model.trainable_parameters(),
        optimizer_name=training_config["optimizer"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    criterion = build_loss_function(training_config["loss_function"])

    use_amp = training_config["use_amp"] and device.type == "cuda"
    scaler = GradScaler(device="cuda") if use_amp else None
    early_stopper = EarlyStopping(patience=training_config["early_stopping_patience"], mode="max")

    best_macro_f1 = 0.0
    num_epochs = training_config["num_epochs"]

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(
            model, dataloaders["train"], optimizer, criterion, device,
            scaler, use_amp, training_config["log_interval_batches"],
        )
        val_loss, val_metrics = validate_one_epoch(model, dataloaders["val"], criterion, device)
        elapsed = time.time() - start_time

        logger.info(
            "[%s] Epoch %d/%d - train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f (%.1fs)",
            model_key, epoch, num_epochs, train_loss, val_loss,
            val_metrics["accuracy"], val_metrics["macro_f1"], elapsed,
        )

        is_best = early_stopper.step(val_metrics["macro_f1"])
        if is_best:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(model, training_config["checkpoint_dir"], checkpoint_prefix, best_macro_f1, epoch)
        if early_stopper.should_stop:
            logger.info("Stopping early at epoch %d for model %s.", epoch, model_key)
            break

    logger.info("=== Finished training %s. Best val_macro_f1=%.4f ===", model_key, best_macro_f1)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_train_pipeline() -> None:
    """Orchestrates training for both EfficientNet-B0 and ViT-Tiny models."""
    base_config = load_config("base")
    data_config = load_config("data")
    training_config = load_config("training")["training"]

    device = _resolve_device(training_config["device"])
    logger.info("Using device: %s", device)

    num_classes = _load_num_classes(base_config, data_config)
    logger.info("Detected %d classes from class_to_idx.json", num_classes)

    dataloaders = _build_dataloaders(base_config, data_config)

    for model_key in MODEL_KEYS:
        try:
            run_training_for_model(model_key, num_classes, dataloaders, training_config, device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower():
                logger.error(
                    "CUDA OutOfMemoryError khi train %s. Hãy giảm dataloader.batch_size "
                    "trong configs/data.yaml (hiện tại 16 -> thử 8).", model_key,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise


if __name__ == "__main__":
    run_train_pipeline()