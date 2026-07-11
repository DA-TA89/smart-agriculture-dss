"""Phase 7 orchestrator: model evaluation, comparison, and MLflow experiment tracking.

Loads the best checkpoint for each trained architecture (EfficientNet-B0,
ViT-Tiny/16), evaluates them on the held-out test split, generates
per-model and comparison Markdown reports plus confusion matrix plots, and
logs the run to MLflow.

Scope note (see Phase_DSS_Agriculture.md, Phase 7): this pipeline only does
MLflow *Experiment Tracking* (log metrics/params/artifacts). MLflow *Model
Registry* (staging -> production promotion) is Phase 11's responsibility,
not implemented here.
"""

import sys
from pathlib import Path

# Bootstrap: allow running this script directly
# (python pipelines\evaluate_pipeline.py) without a "No module named 'src'"
# error. Same fix as pipelines/etl_pipeline.py — see PHASE3_LOG.md section 3,
# issue #1.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging
from typing import Any, Dict, List

import mlflow
import torch
from torch.utils.data import DataLoader

from src.config.config_loader import get_project_root, load_config
from src.data.dataloader import build_dataloaders
from src.evaluation.evaluator import evaluate_model
from src.evaluation.report_generator import (
    generate_comparison_report,
    generate_evaluation_report,
    plot_confusion_matrix,
)
from src.models.model_factory import SUPPORTED_MODEL_KEYS
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("evaluate_pipeline")


def _resolve_device(training_config: Dict[str, Any]) -> torch.device:
    """Resolves the compute device, falling back to CPU if CUDA is unavailable.

    Args:
        training_config: Parsed content of configs/training.yaml.

    Returns:
        Resolved torch.device.
    """
    requested_device = training_config["training"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def _build_test_dataloader(data_config: Dict[str, Any], project_root: Path) -> DataLoader:
    """Builds the test DataLoader via the Phase 5 dataloader factory.

    Args:
        data_config: Parsed content of configs/data.yaml.
        project_root: Absolute path to the project root.

    Returns:
        The "test" DataLoader from build_dataloaders().
    """
    base_config = load_config("base")
    processed_dir = project_root / base_config["paths"]["data_processed_dir"]
    dataloaders = build_dataloaders(data_config, processed_dir, project_root)
    return dataloaders["test"]


def _log_result_to_mlflow(result: Dict[str, Any]) -> None:
    """Logs one model's evaluation result as an MLflow run.

    Args:
        result: Output dict from evaluator.evaluate_model().
    """
    with mlflow.start_run(run_name=f"eval_{result['model_key']}"):
        mlflow.log_param("model_key", result["model_key"])
        mlflow.log_param("checkpoint_path", result["checkpoint_path"])
        mlflow.log_param("total_parameters", result["total_parameters"])
        mlflow.log_param("trainable_parameters", result["trainable_parameters"])

        mlflow.log_metric("accuracy", result["accuracy"])
        mlflow.log_metric("precision_macro", result["precision"])
        mlflow.log_metric("recall_macro", result["recall"])
        mlflow.log_metric("macro_f1", result["macro_f1"])
        mlflow.log_metric("roc_auc_macro_ovr", result["roc_auc"])
        mlflow.log_metric("inference_time_ms_per_image", result["inference_time_ms_per_image"])
        mlflow.log_metric("model_size_mb", result["model_size_mb"])
    logger.info("Logged '%s' evaluation run to MLflow", result["model_key"])


def run_evaluate_pipeline() -> List[Dict[str, Any]]:
    """Runs the full Phase 7 evaluation pipeline for all trained models.

    Returns:
        List of evaluation result dicts, one per model_key.
    """
    project_root = get_project_root()
    base_config = load_config("base")
    data_config = load_config("data")
    training_config = load_config("training")
    eval_config = load_config("evaluation")

    device = _resolve_device(training_config)
    logger.info("Evaluation device resolved to: %s", device)

    test_dataloader = _build_test_dataloader(data_config, project_root)
    class_to_idx: Dict[str, int] = test_dataloader.dataset.class_to_idx
    class_names = sorted(class_to_idx, key=lambda name: class_to_idx[name])

    models_dir = project_root / base_config["paths"]["models_dir"]
    docs_evaluation_dir = project_root / base_config["paths"]["docs_evaluation_dir"]
    min_samples_warning = data_config["split"]["min_samples_per_class_warning"]

    mlflow_cfg = eval_config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    report_cfg = eval_config["report"]
    results: List[Dict[str, Any]] = []

    for model_key in SUPPORTED_MODEL_KEYS:
        logger.info("Evaluating model: %s", model_key)
        result = evaluate_model(
            model_key=model_key,
            dataloader=test_dataloader,
            class_names=class_names,
            models_dir=models_dir,
            eval_config=eval_config,
            min_samples_warning=min_samples_warning,
            device=device,
        )
        results.append(result)

        # Image outputs follow SDS Section 7 kebab-case convention.
        kebab_key = result["checkpoint_prefix"].replace("_", "-")
        confusion_matrix_path = docs_evaluation_dir / report_cfg[
            "confusion_matrix_filename_template"
        ].format(model_key=kebab_key)
        plot_confusion_matrix(
            result["confusion_matrix"], class_names, confusion_matrix_path, model_key
        )

        report_path = docs_evaluation_dir / report_cfg[
            "evaluation_report_filename_template"
        ].format(model_key=result["checkpoint_prefix"])
        generate_evaluation_report(result, report_path)

        _log_result_to_mlflow(result)

    comparison_path = docs_evaluation_dir / report_cfg["comparison_report_filename"]
    generate_comparison_report(results, comparison_path)

    logger.info(
        "Phase 7 evaluation pipeline complete. Reports saved to %s", docs_evaluation_dir
    )
    return results


if __name__ == "__main__":
    run_evaluate_pipeline()