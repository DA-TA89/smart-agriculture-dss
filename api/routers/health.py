"""Liveness endpoint - not part of the Phase 9 checklist itself, but
standard practice for any deployable API (useful for Phase 11 Docker
healthchecks later).
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness check")
def health_check() -> dict:
    """Returns a static OK payload if the API process is alive."""
    return {"status": "ok"}
