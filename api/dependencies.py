"""FastAPI dependencies: access to preloaded model inference contexts.

Both supported models ("efficientnet"/"vit") are loaded exactly once at
API startup (see api.main's lifespan, which calls
pipelines.decision_pipeline.load_pipeline_context() per
configs/inference.yaml -> supported_model_keys) and cached on
``app.state.model_contexts``. Reused here per request instead of
reloading a checkpoint per call - the same principle as the Phase 8 CLI's
single model load per batch, but even more important for a REST API given
its typically higher request volume (see PHASE8_LOG.md, Phase 9 handoff
notes, and PHASE2_LOG.md re: 2GB VRAM GPU).

Database session access (``get_db``) lives in src.database.session, not
here - reused directly by routers via ``Depends(get_db)``.
"""

import logging
from typing import Any, Dict

from fastapi import HTTPException, Request, status

from src.config.config_loader import load_config
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("api_dependencies")


def get_default_model_key() -> str:
    """Reads the default model_key from configs/inference.yaml (not hard-coded)."""
    return load_config("inference")["inference"]["default_model_key"]


def get_model_context(request: Request, model_key: str) -> Dict[str, Any]:
    """Fetches the preloaded pipeline context for a given model_key.

    Args:
        request: The current FastAPI request (used to reach app.state).
        model_key: One of the keys loaded at startup (see api.main lifespan).

    Returns:
        The preloaded context dict (model, class_names, configs, device...)
        - same shape as decision_pipeline.load_pipeline_context()'s output.

    Raises:
        HTTPException: 400 if model_key was not preloaded at startup
            (should only happen if configs/inference.yaml ->
            supported_model_keys was changed without restarting the API).
    """
    contexts: Dict[str, Dict[str, Any]] = request.app.state.model_contexts
    if model_key not in contexts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported or unloaded model_key '{model_key}'. "
                f"Loaded model_keys: {list(contexts.keys())}"
            ),
        )
    return contexts[model_key]
