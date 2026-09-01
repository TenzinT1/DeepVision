"""Provider factory — build concrete providers from settings.

These functions read an :class:`~deepvision.models.settings.AppSettings` (and the
process :class:`~deepvision.config.Config`) and return the appropriate concrete
provider. They are fully implemented so every other layer
gets its providers the same way.

Local-first: when a stage is ``local`` and the real local backend is not yet
implemented, the factory still returns a working *default* provider (Echo / Null
/ Hash) unless ``strict=True`` is passed, so the whole system imports and runs
end-to-end during the build waves. Set ``strict=True`` in production wiring once
the local adapters are implemented.
"""

from __future__ import annotations

from deepvision.config import Config, get_config
from deepvision.models.settings import AppSettings, ProviderMode
from deepvision.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    VisionProvider,
)
from deepvision.providers.embeddings import (
    APIEmbeddings,
    HashEmbeddings,
    LocalEmbeddings,
)
from deepvision.providers.llm import APILLM, EchoLLM, LocalLLM
from deepvision.providers.vision import APIVision, LocalVision, NullVision

__all__ = ["build_llm", "build_vision", "build_embeddings"]


def _catalog_vendor(model_id: str, stage: str) -> str:
    """Resolve ``model_id``'s vendor by looking it up in the model catalog.

    Returns ``'local' | 'anthropic' | 'openai'`` for the given ``stage``
    (``'llm'`` or ``'vision'``). Deliberately looks the id up in
    ``MODEL_CATALOG`` rather than sniffing the string (e.g.
    future non-``claude``-prefixed catalog entry keeps routing correctly.

    The import is lazy (inside the function, not at module scope) because
    ``deepvision.api.deps`` imports this module at load time; importing
    ``deepvision.api.schemas`` back at module scope here would form an
    import cycle (``deepvision.api`` package init -> ``deps`` -> this module
    -> ``deepvision.api.schemas`` -> requires ``deepvision.api`` to already
    be registered in ``sys.modules``, but its ``__init__`` hasn't finished).
    Deferring to call time sidesteps it: by the time any provider is actually
    built, every module involved has finished importing.
    """
    from deepvision.api.schemas import MODEL_CATALOG

    for choice in getattr(MODEL_CATALOG, stage):
        if choice.id == model_id:
            return choice.vendor
    # Not a catalog entry (stale/custom model id) — default to openai rather
    # than guessing vendor from the id string.
    return "openai"


def _api_base_url(cfg: Config, vendor: str) -> str:
    """Anthropic models must hit ``anthropic_base_url``, not the OpenAI one."""
    return cfg.anthropic_base_url if vendor == "anthropic" else cfg.openai_base_url


def build_llm(
    settings: AppSettings, *, config: Config | None = None, strict: bool = False
) -> LLMProvider:
    """Construct the LLM provider for ``settings``.

    ``local`` -> :class:`LocalLLM` (or :class:`EchoLLM` when ``strict`` is False).
    ``api`` -> :class:`APILLM`, with base URL and wire protocol resolved from
    the selected model's catalog vendor (Anthropic vs OpenAI).
    """
    cfg = config or get_config()
    if settings.llm_mode is ProviderMode.API:
        model = settings.llm_model or cfg.api_llm_model
        vendor = _catalog_vendor(model, "llm")
        provider = vendor if vendor in ("anthropic", "openai") else "openai"
        return APILLM(
            model=model,
            api_key=settings.keys.llm_api_key,
            base_url=_api_base_url(cfg, vendor),
            provider=provider,
        )
    model = settings.llm_model or cfg.local_llm_model
    if strict:
        # Timeout comes from config (DEEPVISION_LOCAL_LLM_TIMEOUT) so a slow
        # machine can be tuned without editing the adapter; LocalLLM keeps its
        # own DEFAULT_TIMEOUT for direct construction.
        return LocalLLM(
            model=model, host=cfg.ollama_host, timeout=cfg.local_llm_timeout
        )
    return EchoLLM(model=model)


def build_vision(
    settings: AppSettings, *, config: Config | None = None, strict: bool = False
) -> VisionProvider:
    """Construct the Vision provider for ``settings``.

    ``local`` -> :class:`LocalVision` (or :class:`NullVision` when not strict).
    ``api`` -> :class:`APIVision`, with base URL and wire protocol resolved
    from the selected model's catalog vendor (Anthropic vs OpenAI).
    """
    cfg = config or get_config()
    if settings.vision_mode is ProviderMode.API:
        model = settings.vision_model or cfg.api_vision_model
        vendor = _catalog_vendor(model, "vision")
        provider = vendor if vendor in ("anthropic", "openai") else "openai"
        return APIVision(
            model=model,
            api_key=settings.keys.vision_api_key,
            base_url=_api_base_url(cfg, vendor),
            provider=provider,
        )
    model = settings.vision_model or cfg.local_vision_model
    if strict:
        return LocalVision(model=model)
    return NullVision(model=model)


def build_embeddings(
    settings: AppSettings, *, config: Config | None = None, strict: bool = False
) -> EmbeddingProvider:
    """Construct the Embedding provider for ``settings``.

    ``local`` -> :class:`LocalEmbeddings` (or :class:`HashEmbeddings` when not
    strict). ``api`` -> :class:`APIEmbeddings`. No vendor dispatch needed here
    — Anthropic has no embeddings API, so this stage is always local or OpenAI.
    """
    cfg = config or get_config()
    if settings.embedding_mode is ProviderMode.API:
        return APIEmbeddings(
            model=settings.embedding_model or cfg.api_embedding_model,
            api_key=settings.keys.embedding_api_key,
            base_url=cfg.openai_base_url,
        )
    model = settings.embedding_model or cfg.local_embedding_model
    if strict:
        return LocalEmbeddings(model=model, dim=cfg.embedding_dim)
    return HashEmbeddings(model=model, dim=cfg.embedding_dim)
