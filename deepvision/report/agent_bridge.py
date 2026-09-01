"""Agent/RAG bridge — resolve concrete cross-domain implementations.

the report layer orchestrates the agents layer's agents (``ResearchAgent``,
``SummarizerAgent``, ``CitationAgent``, ``MediaAgent``, ``SynthesisAgent``,
their *abstract* interfaces; unlike the provider layer there is no
``factory.py`` equivalent for agents or RAG, so there is no single canonical
name to import for "the" concrete orchestrator/agent/retriever.

Rather than hardcode a class name (and
that would silently stop working the moment they rename it), this module
finds *any* concrete (non-abstract) subclass of a given ABC at call time,
after eagerly importing the owning package so subclasses defined there get
registered with Python's ABC machinery (``__subclasses__``). This lets
the report layer's routes work end-to-end the moment the rag layer/C land a concrete
class, with zero coupling to its name or module path.

If no concrete subclass exists yet (e.g. mid-build, before the rag layer/C land,
or the build intentionally ships without one), callers are expected to fall
back to a degraded-but-valid response instead of crashing — consistent with
this project's "never crash the pipeline" rule for unavailable model
backends.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Optional, Type, TypeVar

from deepvision.utils import get_logger

if TYPE_CHECKING:
    from deepvision.models.settings import AppSettings
    from deepvision.rag.retrieval import ChunkRetriever

__all__ = ["resolve_concrete", "instantiate", "build_retriever"]

log = get_logger(__name__)

T = TypeVar("T")


def build_retriever(settings: "AppSettings") -> "ChunkRetriever":
    """Construct a fully-wired the rag layer ``ChunkRetriever`` from ``settings``.

    ``ChunkRetriever`` needs an :class:`EmbeddingProvider` and a
    :class:`VectorStore` — it has no no-arg constructor, so the generic
    ``resolve_concrete`` + ``instantiate`` path cannot build it. This helper
    supplies both: the embedding provider from the factory (its ``dim`` selects
    the matching Chroma collection) and the persistent vector store opened at
    that dim, keeping retrieval aligned with how ingestion stored the vectors.
    """
    from deepvision.config import get_config
    from deepvision.providers.factory import build_embeddings
    from deepvision.rag.retrieval import ChunkRetriever
    from deepvision.rag.vector_store import open_vector_store

    embeddings = build_embeddings(settings, strict=get_config().strict_providers)
    store = open_vector_store(embeddings.dim)
    return ChunkRetriever(embeddings, store)


def _concrete_subclasses(base: Type[T]) -> list[Type[T]]:
    """Depth-first walk of ``base.__subclasses__()`` collecting non-abstract classes."""
    seen: set[type] = set()
    stack: list[type] = list(base.__subclasses__())
    found: list[Type[T]] = []
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        if not getattr(cls, "__abstractmethods__", None):
            found.append(cls)  # type: ignore[arg-type]
    return found


def resolve_concrete(base: Type[T], *, package: Optional[str] = None) -> Optional[Type[T]]:
    """Return a concrete subclass of ``base``, or ``None`` if none is registered.

    Args:
        base: the abstract base class to resolve (e.g. ``AgentOrchestrator``).
        package: dotted module path to import first so that any concrete
            subclass defined there is registered (e.g. ``"deepvision.agents"``).
    """
    if package:
        try:
            importlib.import_module(package)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "failed to import package while resolving agent implementation",
                extra={"package": package, "base": base.__name__, "error": str(exc)},
            )
    found = _concrete_subclasses(base)
    if not found:
        return None
    if len(found) > 1:
        log.debug(
            "multiple concrete subclasses found; using the first",
            extra={"base": base.__name__, "candidates": [c.__name__ for c in found]},
        )
    return found[0]


def instantiate(cls: Type[T], *args: object, **kwargs: object) -> T:
    """Instantiate ``cls`` with ``args``/``kwargs``, retrying bare on ``TypeError``.

    a couple of documented cases (``Agent.__init__(self, llm)``). This tries
    the "natural" call first (e.g. passing an ``LLMProvider``) and falls back
    to a no-argument constructor so callers don't need to special-case every
    agent's ``__init__``.
    """
    try:
        return cls(*args, **kwargs)
    except TypeError:
        return cls()  # type: ignore[call-arg]
