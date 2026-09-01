"""Embedding provider implementations.

Local (sentence-transformers, default ``bge-large``) and API embedding adapters
as signature stubs, plus a deterministic :class:`HashEmbeddings` default so the
package imports and the vector layer can be exercised without a model.

Real adapters are builder-owned (bodies raise ``NotImplementedError``);
``HashEmbeddings`` is the offline stub (deterministic, no network).
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from typing import Optional, Sequence

from deepvision.providers.base import EmbeddingProvider
from deepvision.utils.logger import get_logger

__all__ = ["HashEmbeddings", "LocalEmbeddings", "APIEmbeddings"]

_log = get_logger(__name__)


class HashEmbeddings(EmbeddingProvider):
    """A deterministic, dependency-free embedding for imports/tests.

    Produces a fixed-dim, L2-normalized vector by hashing token buckets. Not
    semantically meaningful — purely a functional placeholder so retrieval code
    paths run offline.
    """

    def __init__(self, model: str = "hash-embed", dim: int = 1024) -> None:
        self.model = model
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class LocalEmbeddings(EmbeddingProvider):
    """Local embeddings via sentence-transformers.

    ``sentence-transformers`` (and its ``torch`` dependency) is imported lazily
    inside the methods that need it, so this module — and this class — import
    fine with none of it installed. When it (or the model download) is
    unavailable, calls log a warning and fall back to deterministic zero
    vectors of the expected dimension rather than raising, so the ingestion
    pipeline never crashes; downstream retrieval quality will simply be poor
    until the dependency/model is installed.

    :meth:`embed_image` lazily loads a small CLIP model (``clip-ViT-B-32``,
    also via sentence-transformers) for the optional image-embedding toggle.
    Note its output dimension (512) differs from the default text model's
    (1024) — see the integration report for how downstream layers should
    handle mixed-dimension vectors.
    """

    def __init__(
        self, model: str = "BAAI/bge-large-en-v1.5", *, dim: int = 1024
    ) -> None:
        self.model = model
        self.dim = dim
        self._st_model = None
        self._clip_model = None
        self._log = _log.bind(provider="LocalEmbeddings", model=model)

    def _ensure_text_model(self):
        if self._st_model is not None:
            return self._st_model
        from sentence_transformers import SentenceTransformer

        self._st_model = SentenceTransformer(self.model)
        return self._st_model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        try:
            st_model = self._ensure_text_model()
        except ImportError as exc:
            self._log.warning(
                "sentence-transformers not installed; returning zero vectors",
                error=str(exc),
            )
            return [[0.0] * self.dim for _ in texts]
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline
            self._log.warning(
                "failed to load local embedding model; returning zero vectors",
                error=str(exc),
            )
            return [[0.0] * self.dim for _ in texts]

        try:
            vectors = st_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline
            self._log.warning(
                "local embedding inference failed; returning zero vectors",
                error=str(exc),
            )
            return [[0.0] * self.dim for _ in texts]

        out = [[float(x) for x in vec] for vec in vectors]
        if out and len(out[0]) != self.dim:
            self._log.info(
                "local embedding model dim differs from configured dim; "
                "updating self.dim to match the model's actual output",
                configured_dim=self.dim,
                actual_dim=len(out[0]),
            )
            self.dim = len(out[0])
        return out

    def embed_image(self, image_path: str) -> Optional[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer
            from PIL import Image
        except ImportError as exc:
            self._log.warning(
                "image-embedding deps not installed; skipping image embedding",
                error=str(exc),
            )
            return None
        try:
            if self._clip_model is None:
                self._clip_model = SentenceTransformer("clip-ViT-B-32")
            image = Image.open(image_path).convert("RGB")
            vec = self._clip_model.encode(image, normalize_embeddings=True)
            return [float(x) for x in vec]
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline
            self._log.warning("local image embedding failed", error=str(exc))
            return None


class APIEmbeddings(EmbeddingProvider):
    """API embedding adapter.

    Implemented over :mod:`urllib` against an OpenAI-compatible
    ``POST /embeddings`` endpoint (works against OpenAI directly, and against
    any OpenAI-compatible gateway pointed at via ``base_url``).
    """

    DEFAULT_TIMEOUT = 60.0
    _OPENAI_BASE = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dim: int = 3072,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.dim = dim
        self.timeout = self.DEFAULT_TIMEOUT
        self._log = _log.bind(provider="APIEmbeddings", model=model)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        if not self.api_key:
            raise RuntimeError(
                "APIEmbeddings requires an api_key; configure embedding_api_key "
                "in Settings (embedding_mode='api')."
            )
        url = f"{(self.base_url or self._OPENAI_BASE).rstrip('/')}/embeddings"
        payload = {"model": self.model, "input": texts}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"APIEmbeddings request failed ({exc.code}): {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"APIEmbeddings request failed: {exc}") from exc

        items = sorted(body.get("data") or [], key=lambda d: d.get("index", 0))
        vectors = [item["embedding"] for item in items]
        if vectors:
            self.dim = len(vectors[0])
        return vectors
