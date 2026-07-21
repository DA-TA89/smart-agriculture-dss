"""Orchestrator pipeline for Phase 8: image(s) -> prediction -> decision report.

Wires together src.inference (AI Layer) and src.decision_engine (Business
Layer) end-to-end, per the Decision Support Flow in
Architecture_DSS_Agriculture.md Section 10. Intended for CLI testing of
the Phase 8 deliverables. Accepts either a single image file or a
directory (processed as a batch, with a shared model load — see
``load_pipeline_context()``).

Phase 9's REST API should call ``run_decision_pipeline()`` (single image)
or reuse ``load_pipeline_context()`` / ``predict_and_decide()`` directly
(e.g. to keep the model loaded once across requests) instead of shelling
out to this script.
"""

import sys
from pathlib import Path

# --- sys.path bootstrap (same pattern as etl_pipeline.py / train_pipeline.py) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import random
from collections import Counter
from typing import Any, Dict, List, Optional

import torch

from src.config.config_loader import get_project_root, load_config
from src.decision_engine.decision_engine import run_decision_engine
from src.decision_engine.decision_report_builder import format_decision_report_markdown
from src.decision_engine.knowledge_base_loader import (
    load_knowledge_base,
    validate_against_class_names,
)
from src.inference.predictor import load_class_names, load_inference_model, predict_image
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("decision_pipeline")


def load_pipeline_context(model_key: str) -> Dict[str, Any]:
    """Loads everything needed to run predictions, exactly once.

    Reuses this single loaded model/knowledge base across as many images as
    needed — avoids reloading the checkpoint per image, which matters on
    memory-constrained GPUs (see PHASE6_LOG.md, GPU MX550 2GB VRAM).

    Args:
        model_key: One of model_factory.SUPPORTED_MODEL_KEYS
            ("efficientnet" or "vit").

    Returns:
        Dictionary bundling the loaded model, class names, device, and all
        parsed configs needed by predict_and_decide().
    """
    project_root = get_project_root()
    base_config = load_config("base")
    data_config = load_config("data")
    training_config = load_config("training")
    inference_config = load_config("inference")
    decision_config = load_config("decision_engine")

    device = torch.device(
        training_config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    processed_dir = project_root / base_config["paths"]["data_processed_dir"]
    models_dir = project_root / base_config["paths"]["models_dir"]
    class_to_idx_path = processed_dir / data_config["split"]["class_mapping_filename"]

    class_names = load_class_names(class_to_idx_path)
    model = load_inference_model(model_key, models_dir, len(class_names), device)

    kb_path = project_root / decision_config["knowledge_base"]["file_path"]
    knowledge_base = load_knowledge_base(kb_path)
    validate_against_class_names(knowledge_base, class_names)

    logger.info("Pipeline context loaded (model_key=%s, device=%s)", model_key, device)
    return {
        "project_root": project_root,
        "device": device,
        "class_names": class_names,
        "model": model,
        "data_config": data_config,
        "inference_config": inference_config,
        "decision_config": decision_config,
        "knowledge_base": knowledge_base,
    }


def predict_and_decide(image_path: Path, context: Dict[str, Any]) -> Dict[str, Any]:
    """Runs prediction + decision engine for one image using a preloaded context.

    Args:
        image_path: Path to a single leaf image file.
        context: Output of load_pipeline_context().

    Returns:
        The decision report dictionary, plus "top_k_predictions".
    """
    prediction = predict_image(
        image_path,
        context["model"],
        context["class_names"],
        context["data_config"],
        context["device"],
        top_k=context["inference_config"]["inference"]["top_k"],
    )
    report = run_decision_engine(
        image_reference=str(image_path),
        predicted_class=prediction["predicted_class"],
        confidence_score=prediction["confidence_score"],
        knowledge_base=context["knowledge_base"],
        decision_config=context["decision_config"],
    )
    report["top_k_predictions"] = prediction["top_k_predictions"]
    return report


def run_decision_pipeline(image_path: str, model_key: str) -> Dict[str, Any]:
    """Public single-image entry point (unchanged signature for Phase 9 reuse).

    Args:
        image_path: Path to a leaf image file (absolute or relative to cwd).
        model_key: One of model_factory.SUPPORTED_MODEL_KEYS.

    Returns:
        The final decision report dictionary.
    """
    context = load_pipeline_context(model_key)
    return predict_and_decide(Path(image_path), context)


def collect_image_paths(input_path: Path, data_config: Dict[str, Any]) -> List[Path]:
    """Resolves input_path into a list of image file paths.

    Args:
        input_path: A single image file, or a directory (searched
            recursively for images, so it can point directly at e.g.
            ``data/raw/plantvillage dataset/color/``).
        data_config: Parsed content of configs/data.yaml (reuses
            ``dataset.allowed_extensions`` — no separate extension list).

    Returns:
        Sorted list of matching image file paths.

    Raises:
        FileNotFoundError: If input_path does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")
    if input_path.is_file():
        return [input_path]

    allowed_extensions = {ext.lower() for ext in data_config["dataset"]["allowed_extensions"]}
    found = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in allowed_extensions)
    logger.info("Found %d image(s) under directory %s", len(found), input_path)
    return found


def select_images_to_process(
    all_images: List[Path],
    random_count: Optional[int],
    limit: Optional[int],
    seed: Optional[int],
) -> List[Path]:
    """Applies --random / --limit selection to a list of candidate images.

    This function is a pure selector — it never silently truncates the
    result on its own. When neither --random nor --limit is given, the
    caller must decide (via confirm_large_batch()) whether processing the
    full, unfiltered list is acceptable; that decision belongs to the CLI
    entry point, not to this function (CLI tools should not decide this
    for the user implicitly).

    Args:
        all_images: Full list of images found by collect_image_paths().
        random_count: Value of --random, or None.
        limit: Value of --limit, or None.
        seed: Optional RNG seed for --random (reproducible sampling).

    Returns:
        - random_count random images (sorted) if --random was given.
        - The first `limit` images if --limit was given.
        - Otherwise, the full all_images list, unchanged.
    """
    if random_count is not None:
        rng = random.Random(seed) if seed is not None else random
        chosen = rng.sample(all_images, k=min(random_count, len(all_images)))
        return sorted(chosen)

    if limit is not None:
        return all_images[:limit]

    return all_images


def confirm_large_batch(image_count: int, threshold: int) -> bool:
    """Asks the user to confirm processing a large, unfiltered batch.

    Only called when the user gave neither --random nor --limit and the
    directory contains more than `threshold` images. The CLI never decides
    this silently (explicit > implicit) — see PHASE8_LOG.md for the
    project-owner decision behind this behaviour.

    Args:
        image_count: Number of images found under the input directory.
        threshold: configs/inference.yaml -> inference.batch_confirm_threshold.

    Returns:
        True if the user confirmed with "y", False otherwise (including a
        non-interactive stdin, which is treated as "no" for safety).
    """
    print(
        f"[!] Tìm thấy {image_count} ảnh, chưa chỉ định --limit hoặc --random. "
        f"Xử lý toàn bộ {image_count} ảnh có thể mất nhiều thời gian."
    )
    try:
        answer = input("Tiếp tục xử lý toàn bộ? [y/N]: ").strip().lower()
    except EOFError:
        answer = ""
    return answer == "y"


def _truncate_middle(text: str, max_len: int) -> str:
    """Truncates text to max_len characters, keeping both ends visible.

    PlantVillage filenames start with a meaningless random UUID but end
    with a meaningful, human-readable suffix (e.g.
    "...___RS_Late.B 4946.JPG") — truncating only the end would hide the
    part a person actually reads to tell images apart.

    Args:
        text: The string to truncate.
        max_len: Maximum length of the returned string.

    Returns:
        text unchanged if it already fits, otherwise a "start...end"
        string of exactly max_len characters.
    """
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    keep = max_len - 3
    head_len = keep - keep // 2
    tail_len = keep // 2
    return f"{text[:head_len]}...{text[-tail_len:]}"


def _pad(text: str, width: int) -> str:
    """Left-aligns text to exactly `width` characters, truncating if needed.

    Plain f"{text:<width}" only pads short strings — it does NOT cut long
    ones, which is what caused the summary table's columns to drift out of
    alignment when PlantVillage filenames exceeded the column width. This
    helper guarantees every cell is exactly `width` characters wide.

    Args:
        text: Cell content.
        width: Target column width.

    Returns:
        text truncated (via _truncate_middle) and/or padded to `width`.
    """
    return f"{_truncate_middle(text, width):<{width}}"


IMAGE_COLUMN_WIDTH = 35
CONFIDENCE_COLUMN_WIDTH = 10


def _compute_table_column_widths(
    class_names: List[str], decision_config: Dict[str, Any]
) -> Dict[str, int]:
    """Computes summary-table column widths from real config/data, not magic numbers.

    class/priority/verification columns are sized to fit the LONGEST actual
    value that can appear (all 38 class names, all configured priority and
    verification labels) so nothing in those columns ever needs truncating —
    only the free-form image filename column does.

    Args:
        class_names: Full list of class names (e.g. context["class_names"]).
        decision_config: Parsed content of configs/decision_engine.yaml.

    Returns:
        Dictionary of column widths keyed by "class", "priority", "verification".
    """
    class_width = max(len(name) for name in class_names)
    priority_width = max(len(label) for label in decision_config["priority"]["order"])
    verification_width = max(
        len(decision_config["verification"]["verified_label"]),
        len(decision_config["verification"]["needs_verification_label"]),
    )
    return {"class": class_width, "priority": priority_width, "verification": verification_width}


def _print_summary_table(
    results: List[Dict[str, Any]], failed: List[str], context: Dict[str, Any]
) -> None:
    """Prints a compact, correctly-aligned summary table for a batch run.

    Also prints a breakdown of how many results fall into each priority
    level and each verification status, so a batch spanning many different
    predicted classes (typical with --random) is still easy to scan at a
    glance instead of requiring the reader to eyeball every row.

    Args:
        results: List of successful decision reports.
        failed: List of image paths (as strings) that failed to process.
        context: Output of load_pipeline_context() — used to size columns
            from real class_names/decision_config, not hardcoded widths.
    """
    widths = _compute_table_column_widths(context["class_names"], context["decision_config"])

    print(f"\n=== Tổng hợp {len(results)} ảnh ===")
    header = (
        f"{_pad('Ảnh', IMAGE_COLUMN_WIDTH)}  "
        f"{_pad('Predicted class', widths['class'])}  "
        f"{_pad('Confidence', CONFIDENCE_COLUMN_WIDTH)}  "
        f"{_pad('Priority', widths['priority'])}  "
        f"{_pad('Verification', widths['verification'])}"
    )
    print(header)
    print("-" * len(header))
    for report in results:
        image_name = Path(report["image_reference"]).name
        prediction = report["ai_prediction"]
        confidence_str = f"{prediction['confidence_score']:.2%}"
        row = (
            f"{_pad(image_name, IMAGE_COLUMN_WIDTH)}  "
            f"{_pad(prediction['predicted_class'], widths['class'])}  "
            f"{_pad(confidence_str, CONFIDENCE_COLUMN_WIDTH)}  "
            f"{_pad(report['priority'], widths['priority'])}  "
            f"{_pad(report['verification_status'], widths['verification'])}"
        )
        print(row)

    if results:
        decision_config = context["decision_config"]
        priority_counts = Counter(r["priority"] for r in results)
        verification_counts = Counter(r["verification_status"] for r in results)

        priority_summary = " | ".join(
            f"{label}={priority_counts.get(label, 0)}"
            for label in decision_config["priority"]["order"]
        )
        verification_labels = [
            decision_config["verification"]["verified_label"],
            decision_config["verification"]["needs_verification_label"],
        ]
        verification_summary = " | ".join(
            f"{label}={verification_counts.get(label, 0)}" for label in verification_labels
        )
        print(f"\nPhân bố Priority:     {priority_summary}")
        print(f"Phân bố Verification: {verification_summary}")

    if failed:
        print(f"\n[!] {len(failed)} ảnh lỗi, đã bỏ qua:")
        for failed_path in failed:
            print(f"  - {failed_path}")


def _save_report_markdown(
    report: Dict[str, Any],
    image_path: Path,
    project_root: Path,
    decision_config: Dict[str, Any],
) -> Path:
    """Saves one decision report as Markdown under docs/decision_engine/.

    The filename includes both the source image name and the predicted
    class, to avoid collisions when a batch run predicts the same class
    for multiple images.

    Args:
        report: Output of predict_and_decide().
        image_path: The source image path (used for the filename).
        project_root: Absolute path to the project root.
        decision_config: Parsed content of configs/decision_engine.yaml.

    Returns:
        Path to the saved Markdown file.
    """
    report_cfg = decision_config["report"]
    save_dir = project_root / report_cfg["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = report_cfg["filename_template"].format(
        image_stem=image_path.stem, class_name=report["ai_prediction"]["predicted_class"]
    )
    output_path = save_dir / filename
    output_path.write_text(format_decision_report_markdown(report), encoding="utf-8")
    logger.info("Decision report saved to %s", output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    """Parses CLI arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Run the Phase 8 decision pipeline on one image or a folder of images."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to a single image file, OR a directory (searched recursively).",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        choices=["efficientnet", "vit"],
        help="Model to use. Defaults to configs/inference.yaml -> default_model_key.",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Also save each report as Markdown under docs/decision_engine/sample_reports/.",
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--random",
        type=int,
        default=None,
        metavar="N",
        help="If image_path is a directory, randomly pick N images instead of all/limited ones.",
    )
    sample_group.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="If image_path is a directory, only process the first N images found.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for --random (reproducible sampling).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    inference_default_key = load_config("inference")["inference"]["default_model_key"]
    selected_model_key = args.model_key or inference_default_key

    data_config_cli = load_config("data")
    inference_config_cli = load_config("inference")

    candidate_images = collect_image_paths(Path(args.image_path), data_config_cli)

    if args.random is None and args.limit is None:
        confirm_threshold = inference_config_cli["inference"]["batch_confirm_threshold"]
        if len(candidate_images) > confirm_threshold:
            if not confirm_large_batch(len(candidate_images), confirm_threshold):
                print(
                    "Đã hủy. Dùng --limit N hoặc --random N để chỉ định số lượng ảnh cần xử lý."
                )
                sys.exit(0)

    images_to_process = select_images_to_process(
        candidate_images, args.random, args.limit, args.seed
    )

    if not images_to_process:
        print("Không tìm thấy ảnh nào để xử lý.")
        sys.exit(1)

    context = load_pipeline_context(selected_model_key)
    project_root_cli = get_project_root()
    decision_config_cli = load_config("decision_engine")

    successful_reports: List[Dict[str, Any]] = []
    failed_images: List[str] = []

    for image_path in images_to_process:
        try:
            decision_report = predict_and_decide(image_path, context)
        except Exception as error:  # noqa: BLE001 - one bad image must not kill the batch
            logger.warning("Failed to process %s: %s", image_path, error)
            failed_images.append(str(image_path))
            continue

        successful_reports.append(decision_report)

        # 1 ảnh: in đầy đủ Decision Report. Nhiều ảnh: chỉ in bảng tổng hợp bên dưới để
        # log không quá dài — report đầy đủ từng ảnh vẫn lấy được qua --save-report.
        # Future improvement (ngoài phạm vi hiện tại): thêm --verbose để in đầy đủ report
        # từng ảnh ngay cả khi chạy batch.
        if len(images_to_process) == 1:
            print(format_decision_report_markdown(decision_report))

        if args.save_report:
            saved_path = _save_report_markdown(
                decision_report, image_path, project_root_cli, decision_config_cli
            )
            print(f"[Đã lưu report tại: {saved_path}]")

    if len(images_to_process) > 1:
        _print_summary_table(successful_reports, failed_images, context)