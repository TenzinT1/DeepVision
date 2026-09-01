"""Settings persistence layer (the api layer — Core / Settings / API glue).

Loads and saves the singleton :class:`SettingsRow` (id == 1) and translates it
to/from :class:`AppSettings` — the shape the provider factory
(:mod:`deepvision.providers.factory`) and the rest of the app consume. Also
provides the derived, safe "GET" view (keys redacted, presence booleans, vision
availability) used by the settings router.

This module is pure persistence + view logic — no FastAPI imports — so it can
be reused from :mod:`deepvision.api.deps`, the settings router, and (if needed)
scripts/tests without spinning up the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlmodel import Session

from deepvision.config import Config, get_config
from deepvision.db import get_engine
from deepvision.db.schema import SettingsRow
from deepvision.models import AppSettings, ProviderKeys, ProviderMode
from deepvision.utils import get_logger

__all__ = [
    "SETTINGS_ROW_ID",
    "VisionStatus",
    "default_settings",
    "load_settings",
    "save_settings",
    "redact_settings",
    "keys_present",
    "vision_status",
]

log = get_logger(__name__)

#: SettingsRow is a singleton; id is always 1.
SETTINGS_ROW_ID = 1


def default_settings(config: Optional[Config] = None) -> AppSettings:
    """Build the local-first default :class:`AppSettings`, seeded from Config.

    ``Config.default_{llm,vision,embedding}_mode`` are plain strings (env-driven);
    coerce them into :class:`ProviderMode` so an invalid env value fails loudly
    at startup rather than silently mis-seeding the DB.
    """
    cfg = config or get_config()
    return AppSettings(
        llm_mode=ProviderMode(cfg.default_llm_mode),
        vision_mode=ProviderMode(cfg.default_vision_mode),
        embedding_mode=ProviderMode(cfg.default_embedding_mode),
    )


def load_settings(session: Optional[Session] = None) -> AppSettings:
    """Load the persisted singleton settings row, seeding defaults on first run.

    Args:
        session: optional existing session to reuse (kept open by the caller);
            a short-lived one is opened and closed when omitted.
    """
    owns_session = session is None
    sess = session or Session(get_engine())
    try:
        row = sess.get(SettingsRow, SETTINGS_ROW_ID)
        if row is None:
            settings = default_settings()
            row = SettingsRow(id=SETTINGS_ROW_ID, data=settings.model_dump(mode="json"))
            sess.add(row)
            sess.commit()
            log.info("seeded default settings row")
            return settings
        try:
            return AppSettings.model_validate(row.data)
        except Exception as exc:  # corrupt / outdated shape -> fail soft
            log.error(
                "failed to parse persisted settings row; falling back to defaults",
                extra={"error": str(exc)},
            )
            return default_settings()
    finally:
        if owns_session:
            sess.close()


def _merge_key(current: Optional[str], incoming: Optional[str]) -> Optional[str]:
    """Merge one API key field for a PUT.

    ``GET /settings`` always redacts key values to ``None`` (see
    :func:`redact_settings`), so a client that fetches then re-submits the
    settings object must not be able to silently wipe a previously-saved key.
    Policy: ``None`` means "unchanged, keep current"; ``""`` means "explicitly
    clear"; any other string replaces the stored value.
    """
    if incoming is None:
        return current
    if incoming == "":
        return None
    return incoming


def save_settings(incoming: AppSettings, session: Optional[Session] = None) -> AppSettings:
    """Persist ``incoming`` settings, merging API keys against the current row.

    Also logs (does not fail) when the embedding provider mode/model changes,
    since that invalidates the existing ChromaDB collection dimension and
    requires re-ingest (contract §7.5) — there is no field on
    ``SettingsResponse`` to carry that flag to the client, so it is surfaced via
    a warning log for now; see the domain report for this friction.
    """
    owns_session = session is None
    sess = session or Session(get_engine())
    try:
        current = load_settings(sess)
        merged_keys = ProviderKeys(
            llm_api_key=_merge_key(current.keys.llm_api_key, incoming.keys.llm_api_key),
            vision_api_key=_merge_key(
                current.keys.vision_api_key, incoming.keys.vision_api_key
            ),
            embedding_api_key=_merge_key(
                current.keys.embedding_api_key, incoming.keys.embedding_api_key
            ),
        )
        merged = incoming.model_copy(update={"keys": merged_keys})

        if (
            current.embedding_mode != merged.embedding_mode
            or current.embedding_model != merged.embedding_model
        ):
            log.warning(
                "embedding provider changed; existing vector store is now stale "
                "(dimension mismatch) — ingested papers need to be re-ingested",
                extra={
                    "old_mode": current.embedding_mode.value,
                    "new_mode": merged.embedding_mode.value,
                    "old_model": current.embedding_model,
                    "new_model": merged.embedding_model,
                },
            )

        row = sess.get(SettingsRow, SETTINGS_ROW_ID)
        payload = merged.model_dump(mode="json")
        if row is None:
            row = SettingsRow(id=SETTINGS_ROW_ID, data=payload)
        else:
            row.data = payload
            row.updated_at = datetime.utcnow()
        sess.add(row)
        sess.commit()
        return merged
    finally:
        if owns_session:
            sess.close()


def redact_settings(settings: AppSettings) -> AppSettings:
    """Return a copy of ``settings`` with API key values nulled out."""
    return settings.model_copy(update={"keys": ProviderKeys()})


def keys_present(settings: AppSettings) -> dict[str, bool]:
    """Which api keys are currently set (booleans only, values never exposed)."""
    return {
        "llm": bool(settings.keys.llm_api_key),
        "vision": bool(settings.keys.vision_api_key),
        "embedding": bool(settings.keys.embedding_api_key),
    }


@dataclass
class VisionStatus:
    """Derived view of :meth:`VisionProvider.availability` for the settings pill."""

    available: bool
    text: str


def vision_status(settings: AppSettings) -> VisionStatus:
    """Query the vision provider's availability for the Settings status pill.

    Defensive by design: never raises. The whole point of this check is to
    safely surface "local model missing/unusable" to the UI, so any failure to
    construct the provider or run the probe (missing weights, missing binary,
    the study layer body not implemented yet, ...) is treated as "unavailable" with an
    explanatory message rather than a 500.
    """
    try:
        from deepvision.providers.factory import build_vision

        provider = build_vision(settings, strict=True)
        info = provider.availability()
        text = info.detail or (
            f"{'Available' if info.available else 'Unavailable'}: {info.model}"
        )
        if info.size_hint and info.available and info.size_hint not in text:
            text = f"{text} ({info.size_hint})"
        return VisionStatus(available=info.available, text=text)
    except Exception as exc:
        log.warning("vision availability check failed", extra={"error": str(exc)})
        return VisionStatus(
            available=False,
            text=f"Vision backend unavailable: {exc}",
        )
