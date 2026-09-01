"""Provider interfaces (ABCs) for the three swappable model stages.

DeepVision runs local-first but every model stage is pluggable: LLM, Vision, and
Embeddings each sit behind an abstract base class so the local and API
implementations are interchangeable. The provider factory
(:mod:`deepvision.providers.factory`) constructs the right concrete class from
:class:`~deepvision.models.settings.AppSettings`.

These ABCs are the contract. Concrete implementations
(``llm.py``, ``vision.py``, ``embeddings.py``) fill in the bodies.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

__all__ = [
    "Message",
    "VisionAvailability",
    "LLMProvider",
    "VisionProvider",
    "EmbeddingProvider",
]


@dataclass
class Message:
    """A single chat-style message passed to an :class:`LLMProvider`."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class VisionAvailability:
    """Result of :meth:`VisionProvider.availability`.

    Backs the local-vision status pill on the Settings screen.
    """

    available: bool
    model: str
    detail: str = ""
    size_hint: Optional[str] = None  # e.g. "5.4 GB"


class LLMProvider(abc.ABC):
    """Abstract text-generation provider (local Ollama or an API endpoint)."""

    #: Human/model identifier, e.g. "llama3.1:8b" or "gpt-4o-mini".
    model: str

    @abc.abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Return a single completion for ``messages``.

        Args:
            messages: ordered chat messages (system/user/assistant).
            temperature: sampling temperature.
            max_tokens: optional output cap.

        Returns:
            The assistant's full text response.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield incremental text deltas for a streamed completion.

        Used by the Chat screen's token-by-token rendering.
        """
        raise NotImplementedError

    def health(self) -> bool:
        """Return True if the backend is reachable/loaded. Override as needed."""
        return True


class VisionProvider(abc.ABC):
    """Abstract image-understanding provider (Qwen2-VL / Florence-2 or API)."""

    model: str

    @abc.abstractmethod
    def describe_image(
        self,
        image_path: str,
        *,
        prompt: Optional[str] = None,
        caption_hint: Optional[str] = None,
    ) -> str:
        """Return a textual description/analysis of the image at ``image_path``.

        Args:
            image_path: absolute path to a figure/table crop.
            prompt: optional instruction (defaults to a generic caption prompt).
            caption_hint: nearby caption text to condition on, if available.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def availability(self) -> VisionAvailability:
        """Report whether the (local) vision model is installed and usable.

        Drives the Settings screen status pill. API providers report available
        when a key is configured.
        """
        raise NotImplementedError


class EmbeddingProvider(abc.ABC):
    """Abstract text-embedding provider (sentence-transformers or API)."""

    model: str

    #: Output vector dimension (must match the Chroma collection).
    dim: int

    @abc.abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input (order preserved)."""
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Default: delegate to :meth:`embed_texts`."""
        return self.embed_texts([text])[0]

    def embed_image(self, image_path: str) -> Optional[list[float]]:
        """Optionally embed an image crop for visual retrieval.

        Returns ``None`` when the provider has no image-embedding capability
        (the default). Only relevant when the advanced image-embedding toggle is
        on.
        """
        return None

    def embed_iter(self, texts: Iterable[str]) -> Iterator[list[float]]:
        """Convenience streaming wrapper over :meth:`embed_texts`."""
        for vec in self.embed_texts(list(texts)):
            yield vec
