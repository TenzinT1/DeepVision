"""Health router — GET /health.

Architect-owned stub: signature + decorator fixed; the health-check body is
builder-owned. This one returns a minimal live response so the app boots.
"""

from __future__ import annotations

from fastapi import APIRouter
from sqlmodel import Session, select

from deepvision import __version__
from deepvision.api.schemas import HealthResponse
from deepvision.api.settings_store import load_settings, vision_status
from deepvision.db import get_engine
from deepvision.db.schema import SettingsRow
from deepvision.utils import get_logger

router = APIRouter(tags=["health"])
log = get_logger(__name__)


def _db_ok() -> bool:
    """True if the SQLite engine is reachable and queryable."""
    try:
        with Session(get_engine()) as session:
            session.exec(select(SettingsRow.id).limit(1)).first()
        return True
    except Exception as exc:  # pragma: no cover - defensive
        log.error("db health check failed", extra={"error": str(exc)})
        return False


def _llm_online(settings) -> bool:
    """True if the configured LLM backend responds to a health probe.

    Defensive: provider construction/health() can raise (missing binary,
    ``NotImplementedError`` before the study layer lands, network error, ...) — any
    failure here just means "not online", never a 500.
    """
    try:
        from deepvision.providers.factory import build_llm

        return bool(build_llm(settings, strict=True).health())
    except Exception as exc:
        log.warning("llm health probe failed", extra={"error": str(exc)})
        return False


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe: app version, DB reachability, and local-model status."""
    db_ok = _db_ok()
    try:
        settings = load_settings() if db_ok else None
    except Exception as exc:  # pragma: no cover - defensive
        log.error("failed to load settings for health check", extra={"error": str(exc)})
        settings = None

    if settings is None:
        return HealthResponse(
            status="ok" if db_ok else "degraded",
            version=__version__,
            llm_online=False,
            vision_available=False,
            db_ok=db_ok,
        )

    vstatus = vision_status(settings)
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        version=__version__,
        llm_online=_llm_online(settings),
        vision_available=vstatus.available,
        db_ok=db_ok,
    )
