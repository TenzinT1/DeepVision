"""FastAPI dependency signatures.

Shared dependencies injected into route handlers: a DB session, the current
persisted :class:`AppSettings`, and the provider trio built from those settings.
``get_session`` wraps the DB session factory; the settings/provider
dependencies fall back to safe defaults so routes stay importable and testable
without a configured provider.
"""

from __future__ import annotations

from typing import Iterator

from sqlmodel import Session

from deepvision.api.settings_store import load_settings
from deepvision.config import Config, get_config
from deepvision.db import get_engine
from deepvision.models.settings import AppSettings
from deepvision.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    VisionProvider,
)
from deepvision.providers.factory import (
    build_embeddings,
    build_llm,
    build_vision,
)

__all__ = [
    "get_config_dep",
    "get_session",
    "get_settings",
    "get_llm",
    "get_vision",
    "get_embeddings",
]


def get_config_dep() -> Config:
    """Dependency: the process :class:`Config`."""
    return get_config()


def get_session() -> Iterator[Session]:
    """Dependency: a DB session (closed after the request).

    Yields a session bound to the shared engine.
    """
    session = Session(get_engine())
    try:
        yield session
    finally:
        session.close()


def get_settings() -> AppSettings:
    """Dependency: the current persisted :class:`AppSettings`.

    Loads the singleton ``SettingsRow`` (id == 1) via the settings persistence
    layer, seeding local-first defaults on first run. Callable with no
    arguments (opens and closes its own short-lived session) so route bodies
    can call it directly even though the frozen router signatures don't thread
    a ``Session`` through ``Depends``.
    """
    return load_settings()


def get_llm(settings: AppSettings | None = None) -> LLMProvider:
    """Dependency: the LLM provider for the current settings."""
    return build_llm(settings or get_settings(), strict=get_config().strict_providers)


def get_vision(settings: AppSettings | None = None) -> VisionProvider:
    """Dependency: the Vision provider for the current settings."""
    return build_vision(
        settings or get_settings(), strict=get_config().strict_providers
    )


def get_embeddings(settings: AppSettings | None = None) -> EmbeddingProvider:
    """Dependency: the Embedding provider for the current settings."""
    return build_embeddings(
        settings or get_settings(), strict=get_config().strict_providers
    )
