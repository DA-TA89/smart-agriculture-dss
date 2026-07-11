"""Model evaluation utilities: metrics, confusion matrix, inference speed, model size.

Loads a trained checkpoint for a given architecture (per the Phase 6 naming
convention: ``{architecture}_plantvillage_f1_{score}.pth``), runs inference
on a held-out DataLoader, and computes the full set of classification
metrics required by Phase 7 (see Phase_DSS_Agriculture.md, Phase 7
checklist: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion
Matrix, Inference Time, Model Size).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.models.model_factory import build_model, get_checkpoint_prefix
from src.training.checkpoint_manager import DATASET_TAG
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("model_evaluator")


def find_checkpoint_path(models_dir: Path, checkpoint_prefix: str) -> Path:
    """Locates the best checkpoint file for a given architecture.

    Args:
        models_dir: Directory containing .pth checkpoint files (models/).
        checkpoint_prefix: Architecture prefix, e.g. "efficientnet_b0".

    Returns:
        Path to the matched checkpoint file.

    Raises:
        FileNotFoundError: If no matching checkpoint exists.
    """
    pattern = f"{checkpoint_prefix}_{DATASET_TAG}_f1_*.pth"
    matches = sorted(models_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for prefix '{checkpoint_prefix}' in "
            f"{models_dir} (pattern: {pattern})."
        )
    if len(matches) > 1:
        logger.warning(
            "Multiple checkpoints matched prefix '%s': %s. Using the most "
            "recently modified one.",
            checkpoint_prefix,
            [match.name for match in matches],
        )
        matches.sort(key=lambda path: path.stat().st_mtime)
    return matches[-1]


def load_trained_model(
    model_key: str, checkpoint_path: Path, num_classes: int, device: torch.device
) -> nn.Module:
    """Builds a model via the Phase 6 factory and loads trained weights.

    Args:
        model_key: One of model_factory.SUPPORTED_MODEL_KEYS.
        checkpoint_path: Path to the .pth state_dict file.
        num_classes: Number of output classes.
        device: Device to move the model to.

    Returns:
        The model in eval() mode, ready for inference.
    """
    model = build_model(model_key, num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint for '%s' from %s", model_key, checkpoint_path)
    return model


def run_inference(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs full inference over a DataLoader and collects predictions.

    Args:
        model: Trained model in eval() mode.
        dataloader: DataLoader to iterate over (typically the test split).
        device: Device to run inference on.

    Returns:
        Tuple of (y_true, y_pred, y_proba) as numpy arrays. y_proba has
        shape (n_samples, n_classes).
    """
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probas: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probas = torch.softmax(logits, dim=1)
            preds = torch.argmax(probas, dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probas.append(probas.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.concatenate(all_probas, axis=0)
    return y_true, y_pred, y_proba


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    eval_config: Dict[str, Any],
    min_samples_warning: int,
) -> Dict[str, Any]:
    """Computes the full Phase 7 metric set for one model.

    Args:
        y_true: Ground-truth integer labels, shape (n_samples,).
        y_pred: Predicted integer labels, shape (n_samples,).
        y_proba: Predicted class probabilities, shape (n_samples, n_classes).
        class_names: Ordered class names (index i corresponds to label i).
        eval_config: Parsed content of configs/evaluation.yaml.
        min_samples_warning: Support threshold below which a class's
            per-class metrics are flagged as low-confidence (reused from
            configs/data.yaml -> split.min_samples_per_class_warning).

    Returns:
        Dictionary with overall metrics, confusion matrix, per-class
        report, and a list of small-sample class names.
    """
    metrics_cfg = eval_config["metrics"]
    average = metrics_cfg["classification_average"]
    label_indices = list(range(len(class_names)))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    try:
        roc_auc = roc_auc_score(
            y_true,
            y_proba,
            multi_class=metrics_cfg["roc_auc_multi_class"],
            average=metrics_cfg["roc_auc_average"],
            labels=label_indices,
        )
    except ValueError as error:
        logger.warning("ROC-AUC computation failed (%s). Reporting NaN.", error)
        roc_auc = float("nan")

    conf_matrix = confusion_matrix(y_true, y_pred, labels=label_indices)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    small_sample_classes = [
        name for name in class_names if report_dict[name]["support"] < min_samples_warning
    ]

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": conf_matrix,
        "per_class_report": report_dict,
        "small_sample_classes": small_sample_classes,
    }


def measure_inference_time_ms(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_warmup_batches: int,
    num_timed_batches: int,
) -> float:
    """Measures average per-image inference latency.

    Runs a few warm-up batches first (unmeasured), then times a fixed
    number of batches to compute average milliseconds per image.

    Args:
        model: Trained model in eval() mode.
        dataloader: Source of input batches.
        device: Device to run inference on.
        num_warmup_batches: Number of unmeasured warm-up batches.
        num_timed_batches: Number of batches to time.

    Returns:
        Average inference time in milliseconds per image.
    """
    model.eval()
    data_iter = iter(dataloader)

    def _next_batch() -> torch.Tensor:
        nonlocal data_iter
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, _ = next(data_iter)
        return images

    with torch.no_grad():
        for _ in range(num_warmup_batches):
            images = _next_batch().to(device)
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()

        total_images = 0
        start_time = time.perf_counter()
        for _ in range(num_timed_batches):
            images = _next_batch().to(device)
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_images += images.size(0)
        elapsed_seconds = time.perf_counter() - start_time

    return (elapsed_seconds / total_images) * 1000


def get_model_size_mb(checkpoint_path: Path) -> float:
    """Returns the checkpoint file size in megabytes.

    Args:
        checkpoint_path: Path to the .pth file.

    Returns:
        File size in MB.
    """
    return checkpoint_path.stat().st_size / (1024 * 1024)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Counts total and trainable parameters of a model.

    Args:
        model: The model to inspect.

    Returns:
        Dictionary with keys "total_parameters" and "trainable_parameters".
    """
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def evaluate_model(
    model_key: str,
    dataloader: DataLoader,
    class_names: List[str],
    models_dir: Path,
    eval_config: Dict[str, Any],
    min_samples_warning: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Runs the full evaluation flow for a single trained model.

    Args:
        model_key: One of model_factory.SUPPORTED_MODEL_KEYS.
        dataloader: Held-out (test) DataLoader.
        class_names: Ordered class names (index i corresponds to label i).
        models_dir: Directory containing checkpoint files.
        eval_config: Parsed content of configs/evaluation.yaml.
        min_samples_warning: Support threshold for small-sample warnings.
        device: Device to run inference on.

    Returns:
        Dictionary combining metrics, confusion matrix, per-class report,
        inference speed, model size, and parameter counts for this model.
    """
    checkpoint_prefix = get_checkpoint_prefix(model_key)
    checkpoint_path = find_checkpoint_path(models_dir, checkpoint_prefix)

    model = load_trained_model(model_key, checkpoint_path, len(class_names), device)

    y_true, y_pred, y_proba = run_inference(model, dataloader, device)
    metrics = compute_evaluation_metrics(
        y_true, y_pred, y_proba, class_names, eval_config, min_samples_warning
    )

    speed_cfg = eval_config["inference_speed"]
    inference_ms = measure_inference_time_ms(
        model,
        dataloader,
        device,
        speed_cfg["num_warmup_batches"],
        speed_cfg["num_timed_batches"],
    )
    model_size_mb = get_model_size_mb(checkpoint_path)
    param_counts = count_parameters(model)

    result: Dict[str, Any] = {
        "model_key": model_key,
        "checkpoint_prefix": checkpoint_prefix,
        "checkpoint_path": str(checkpoint_path),
        "inference_time_ms_per_image": inference_ms,
        "model_size_mb": model_size_mb,
        **param_counts,
        **metrics,
    }
    logger.info(
        "Evaluated '%s': accuracy=%.4f, macro_f1=%.4f, roc_auc=%.4f, "
        "inference=%.3fms/img, size=%.2fMB",
        model_key,
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["roc_auc"],
        inference_ms,
        model_size_mb,
    )
    return result