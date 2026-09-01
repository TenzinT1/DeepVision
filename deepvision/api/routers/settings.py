"""Settings router — GET /settings, PUT /settings.

Owned by: SETTINGS/CORE domain. Reads and writes the singleton
:class:`AppSettings`, returning derived flags (needs_keys, key presence, vision
availability) so the Settings screen renders without recomputation.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from deepvision.api import settings_store
from deepvision.api.schemas import (
    DETAIL_LEVELS,
    MODEL_CATALOG,
    SettingsResponse,
    UpdateSettingsRequest,
)
from deepvision.models import AppSettings
from deepvision.utils import get_logger

router = APIRouter(tags=["settings"])
log = get_logger(__name__)


def _to_response(settings: AppSettings) -> SettingsResponse:
    """Build the derived, redacted view shared by GET and PUT.

    ``model_catalog`` / ``detail_levels`` are returned verbatim from
    schemas.py (not duplicated here) — imported explicitly rather than
    relying solely on ``SettingsResponse``'s default_factory, per the
    contract note that the router should be updated even though the
    default_factory already makes this work.
    """
    status = settings_store.vision_status(settings)
    return SettingsResponse(
        settings=settings_store.redact_settings(settings),
        needs_keys=settings.needs_keys,
        keys_present=settings_store.keys_present(settings),
        vision_available=status.available,
        vision_status_text=status.text,
        model_catalog=MODEL_CATALOG.model_copy(deep=True),
        detail_levels=list(DETAIL_LEVELS),
    )


@router.get("/settings", response_model=SettingsResponse)
def get_settings() -> SettingsResponse:
    """Return current settings (API key values redacted) plus derived flags."""
    settings = settings_store.load_settings()
    return _to_response(settings)


@router.put("/settings", response_model=SettingsResponse)
def update_settings(request: UpdateSettingsRequest) -> SettingsResponse:
    """Persist new settings and return the refreshed derived view.

    ``request.settings`` is already a validated :class:`AppSettings` (provider
    modes, chunk size/overlap bounds, etc. are enforced by the model itself, so
    FastAPI/pydantic returns 422 before this body runs for structurally invalid
    input). Any remaining validation error while merging/persisting is reported
    as 422 rather than a raw 500.
    """
    try:
        saved = settings_store.save_settings(request.settings)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    log.info(
        "settings updated",
        extra={
            "llm_mode": saved.llm_mode.value,
            "vision_mode": saved.vision_mode.value,
            "embedding_mode": saved.embedding_mode.value,
        },
    )
    return _to_response(saved)
