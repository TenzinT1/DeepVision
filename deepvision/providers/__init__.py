"""Provider layer: interfaces, concrete adapters, and the factory.

Local-first. LLM / Vision / Embedding each sit behind an ABC (``base.py``) with
local + API adapters and a functional default (Echo / Null / Hash) so imports
never require a model. Build providers via :func:`build_llm`, :func:`build_vision`,
:func:`build_embeddings`.
"""

from deepvision.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    Message,
    VisionAvailability,
    VisionProvider,
)
from deepvision.providers.embeddings import (
    APIEmbeddings,
    HashEmbeddings,
    LocalEmbeddings,
)
from deepvision.providers.factory import (
    build_embeddings,
    build_llm,
    build_vision,
)
from deepvision.providers.llm import APILLM, EchoLLM, LocalLLM
from deepvision.providers.vision import APIVision, LocalVision, NullVision

__all__ = [
    "Message",
    "VisionAvailability",
    "LLMProvider",
    "VisionProvider",
    "EmbeddingProvider",
    "EchoLLM",
    "LocalLLM",
    "APILLM",
    "NullVision",
    "LocalVision",
    "APIVision",
    "HashEmbeddings",
    "LocalEmbeddings",
    "APIEmbeddings",
    "build_llm",
    "build_vision",
    "build_embeddings",
]
