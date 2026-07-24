"""FastAPI application entry point - Phase 9: Smart Agriculture DSS REST API.

Application layer only (SDS Section 3): receives HTTP requests, validates
input, delegates to src.services (Business layer). Never touches AI-layer
internals (src.inference/src.decision_engine) or the database directly -
that would violate Low Coupling (SDS Section 2).

Run with (from project root, venv activated):
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Swagger UI: http://localhost:8000/docs
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict

# --- sys.path bootstrap (same pattern as etl_pipeline.py / train_pipeline.py /
#     decision_pipeline.py) - cần thiết vì uvicorn import module này bằng
#     đường dẫn "api.main", không tự thêm project root vào sys.path. ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import health, predictions
from pipelines.decision_pipeline import load_pipeline_context
from src.config.config_loader import load_config
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("api_main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Loads every supported model exactly once at API startup.

    Both "efficientnet" and "vit" are preloaded (not lazy-loaded per
    request) so a per-request model_key choice never triggers a checkpoint
    reload - both checkpoints together are small (~37MB combined, see
    PHASE7_LOG.md model sizes: 15.77MB + 21.16MB) and fit comfortably even
    on the 2GB VRAM GPU this project targets (PHASE2_LOG.md).
    """
    inference_config = load_config("inference")
    supported_model_keys = inference_config["inference"]["supported_model_keys"]

    contexts: Dict[str, dict] = {}
    for model_key in supported_model_keys:
        logger.info("Loading pipeline context for model_key=%s ...", model_key)
        contexts[model_key] = load_pipeline_context(model_key)
    app.state.model_contexts = contexts

    logger.info("All model contexts loaded: %s", list(contexts.keys()))
    yield
    logger.info("API shutting down.")


_api_config = load_config("api")["api"]

app = FastAPI(
    title=_api_config["title"],
    description=_api_config["description"],
    version=_api_config["version"],
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_api_config["cors_allow_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predictions.router)
