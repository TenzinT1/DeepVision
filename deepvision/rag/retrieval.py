"""Retriever — grounded chunk retrieval for chat and agents.

Embeds a query and returns the most relevant chunks for a single paper. This is
the RAG entry point the chat route and agents call.
"""

from __future__ import annotations

import abc
from typing import Optional, Sequence

from deepvision.models.chunks import AnyChunk
from deepvision.providers.base import EmbeddingProvider
from deepvision.rag.vector_store import VectorStore, chunk_from_hit
from deepvision.utils.logger import get_logger

__all__ = ["Retriever", "ChunkRetriever"]

log = get_logger(__name__)


class Retriever(abc.ABC):
    """Retrieves the top-``k`` chunks grounding an answer in one paper."""

    @abc.abstractmethod
    def retrieve(self, query: str, paper_id: str, k: int = 6) -> list[AnyChunk]:
        """Return the ``k`` most relevant chunks of ``paper_id`` for ``query``.

        Chat grounding is strictly single-paper, so ``paper_id`` is required.
        """
        raise NotImplementedError


class ChunkRetriever(Retriever):
    """Embeds the query once, then does a filtered nearest-neighbour lookup.

    Wraps an :class:`EmbeddingProvider` (query embedding) and a
    :class:`VectorStore` (nearest-neighbour search), and reconstructs typed
    ``AnyChunk`` instances from the stored metadata so citations/media can be
    built directly off the result.

    Also exposes :meth:`retrieve_many` for multi-paper retrieval (used by
    ``SynthesisAgent.compare``): the ``Retriever`` ABC's ``retrieve`` signature
    is single-paper by contract, so multi-paper support is an additive method
    on this concrete class rather than a change to the frozen interface.
    """

    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore) -> None:
        self._embeddings = embedding_provider
        self._store = vector_store

    def retrieve(
        self,
        query: str,
        paper_id: str,
        k: int = 6,
        *,
        page_range: Optional[tuple[int, int]] = None,
    ) -> list[AnyChunk]:
        """Return the ``k`` best chunks of ``paper_id`` for ``query``.

        ``page_range`` is an **additive, keyword-only** inclusive
        ``(page_start, page_end)`` filter pushed down to the vector store's
        native metadata filter — the fast half of chapter scoping. It is not
        part of the frozen :class:`Retriever` ABC signature; the scoping
        guarantee itself lives in
        :class:`~deepvision.rag.chapter_scope.PageScopedRetriever`, which
        post-filters unconditionally. Default ``None`` leaves every existing
        call site (chat, compare, whole-paper reports) unchanged.
        """
        if not query or not query.strip():
            return []
        vector = self._embeddings.embed_query(query)
        hits = self._query_store(vector, paper_id=paper_id, k=k, page_range=page_range)
        return self._hits_to_chunks(hits)

    def retrieve_many(
        self,
        query: str,
        paper_ids: Sequence[str],
        k: int = 6,
        *,
        page_range: Optional[tuple[int, int]] = None,
    ) -> dict[str, list[AnyChunk]]:
        """Retrieve top-``k`` chunks per paper for a compare-style query.

        Embeds ``query`` once and issues one paper-filtered vector-store query
        per id in ``paper_ids`` (2-4 papers for ``/compare``), returning a
        dict keyed by paper_id in the same order as the input. ``page_range``
        behaves exactly as in :meth:`retrieve` and applies to every paper.
        """
        if not query or not query.strip():
            return {pid: [] for pid in paper_ids}
        vector = self._embeddings.embed_query(query)
        results: dict[str, list[AnyChunk]] = {}
        for pid in paper_ids:
            hits = self._query_store(vector, paper_id=pid, k=k, page_range=page_range)
            results[pid] = self._hits_to_chunks(hits)
        return results

    def _query_store(
        self,
        vector: Sequence[float],
        *,
        paper_id: str,
        k: int,
        page_range: Optional[tuple[int, int]],
    ):
        """Query the store, keeping the unscoped call byte-identical to before.

        A ``VectorStore`` implementation that predates the ``page_range`` kwarg
        raises ``TypeError``; rather than crash the pipeline we fall back to the
        unfiltered query and let the caller
        (:class:`~deepvision.rag.chapter_scope.PageScopedRetriever`) enforce the
        page window in Python.
        """
        if page_range is None:
            return self._store.query(vector, paper_id=paper_id, k=k)
        try:
            return self._store.query(
                vector, paper_id=paper_id, k=k, page_range=page_range
            )
        except TypeError:
            log.warning(
                "vector store does not support page_range; falling back to an "
                "unfiltered query (the caller must post-filter)",
                paper_id=paper_id,
            )
            return self._store.query(vector, paper_id=paper_id, k=k)

    @staticmethod
    def _hits_to_chunks(hits) -> list[AnyChunk]:
        chunks: list[AnyChunk] = []
        for hit in hits:
            chunk = chunk_from_hit(hit)
            if chunk is not None:
                chunks.append(chunk)
            else:
                log.warning("dropping unreconstructable vector hit", chunk_id=hit.chunk_id)
        return chunks
