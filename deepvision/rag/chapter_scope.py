"""Chapter scoping — a Retriever decorator that hides everything outside a page range.

This is the load-bearing half of per-chapter reports: when a chapter scope is
active, ``ResearchAgent.plan_outline`` (and therefore every downstream agent)
must only ever see chunks whose ``page`` falls inside
``[page_start, page_end]``.

Why a decorator rather than a parameter on the agents: the agent ensemble has
exactly **one** retrieval entry point — the :class:`~deepvision.rag.retrieval.Retriever`
injected into ``AgentOrchestrator.__init__``, which it hands to
``ResearchAgent.plan_outline(paper_id, settings, retriever)`` and to
``SynthesisAgent(llm, retriever=...)``. Wrapping that single object scopes the
whole pipeline with zero edits to any agent, and the abstract
``Retriever.retrieve`` signature stays frozen (same precedent as
``retrieve_many``, which was added as a concrete-class method rather than a
change to the ABC).

The scope is enforced twice on purpose:

1. **Natively**, by passing the additive ``page_range`` kwarg down to
   ``ChunkRetriever.retrieve`` → ``VectorStore.query`` (Chroma metadata filter,
   or the equivalent test in the JSON fallback index) — this is what keeps
   recall high, since the nearest-neighbour search itself only considers
   in-chapter chunks.
2. **In Python**, unconditionally, on every returned chunk. This is the actual
   guarantee: no out-of-range chunk reaches the agents regardless of backend
   behaviour, Chroma version, or an inner retriever that predates the kwarg.

It never widens: if nothing survives the filter it returns ``[]`` and logs a
warning. An empty, honest section beats a "chapter" report grounded in the
wrong chapter.
"""

from __future__ import annotations

from typing import Optional, Sequence

from deepvision.models.chunks import AnyChunk
from deepvision.rag.retrieval import Retriever
from deepvision.utils.logger import get_logger

__all__ = ["PageScopedRetriever", "scoped_retriever"]

log = get_logger(__name__)

#: Never issue a scoped query narrower than this — a chapter is a small slice of
#: a paper, so post-filtering a handful of hits would starve the report.
_MIN_WIDE = 24

#: Upper bound on the over-fetch, so escalation can't turn into a full scan.
_MAX_WIDE = 256

#: How many times :meth:`PageScopedRetriever.retrieve` may widen ``k`` when the
#: page filter left it short.
_MAX_ESCALATIONS = 2


class PageScopedRetriever(Retriever):
    """Wraps any :class:`Retriever` and hides every chunk outside a page range.

    Args:
        inner: the real retriever (normally a
            :class:`~deepvision.rag.retrieval.ChunkRetriever`).
        page_start: 1-based first page of the chapter (inclusive).
        page_end: 1-based last page of the chapter (inclusive).
        over_fetch: multiplier applied to ``k`` before filtering, so a chapter
            that is a small slice of the paper still yields ``k`` usable chunks.
    """

    def __init__(
        self,
        inner: Retriever,
        *,
        page_start: int,
        page_end: int,
        over_fetch: int = 6,
    ) -> None:
        start = max(1, int(page_start))
        end = max(1, int(page_end))
        if end < start:
            log.warning(
                "page range is inverted; swapping bounds",
                page_start=start,
                page_end=end,
            )
            start, end = end, start
        self.inner = inner
        self.page_start = start
        self.page_end = end
        self.over_fetch = max(1, int(over_fetch))
        #: Flipped off permanently the first time the inner retriever rejects
        #: the ``page_range`` kwarg, so we don't pay a TypeError per query.
        self._supports_page_range = True

    # -- Retriever -----------------------------------------------------

    def retrieve(self, query: str, paper_id: str, k: int = 6) -> list[AnyChunk]:
        """Return up to ``k`` chunks of ``paper_id``, all inside the page range."""
        if not query or not query.strip():
            return []
        k = max(1, int(k))
        wide = max(k * self.over_fetch, _MIN_WIDE)

        kept: list[AnyChunk] = []
        for attempt in range(_MAX_ESCALATIONS + 1):
            hits = self._inner_retrieve(query, paper_id, wide)
            kept = self.filter_chunks(hits)
            if len(kept) >= k or wide >= _MAX_WIDE:
                break
            if attempt < _MAX_ESCALATIONS:
                wide = min(wide * 4, _MAX_WIDE)

        if not kept:
            log.warning(
                "no chunks inside the chapter page range; returning empty rather "
                "than widening the scope",
                paper_id=paper_id,
                page_start=self.page_start,
                page_end=self.page_end,
            )
            return []
        return kept[:k]

    def retrieve_many(
        self, query: str, paper_ids: Sequence[str], k: int = 6
    ) -> dict[str, list[AnyChunk]]:
        """Per-paper :meth:`retrieve`, so multi-paper callers stay scoped too.

        ``SynthesisAgent`` holds the same retriever instance this decorator
        wraps, so it must honour the page window as well.
        """
        return {pid: self.retrieve(query, pid, k=k) for pid in paper_ids}

    # -- internals -----------------------------------------------------

    def in_scope(self, chunk: AnyChunk) -> bool:
        """True if ``chunk``'s page lies inside the scope (unknown page = out)."""
        page = getattr(chunk, "page", None)
        try:
            page_int = int(page)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return self.page_start <= page_int <= self.page_end

    def filter_chunks(self, chunks: Sequence[AnyChunk]) -> list[AnyChunk]:
        """Drop every chunk outside the page range, preserving rank order."""
        return [c for c in chunks if self.in_scope(c)]

    def _inner_retrieve(self, query: str, paper_id: str, k: int) -> list[AnyChunk]:
        """Call the inner retriever, preferring the native page filter."""
        if self._supports_page_range:
            try:
                return list(
                    self.inner.retrieve(  # type: ignore[call-arg]
                        query,
                        paper_id,
                        k=k,
                        page_range=(self.page_start, self.page_end),
                    )
                )
            except TypeError:
                # An inner retriever that landed before the kwarg existed.
                # Degrade to the plain call and rely on the post-filter — never
                # crash the pipeline (project rule).
                self._supports_page_range = False
                log.warning(
                    "inner retriever does not accept page_range; relying on the "
                    "in-process page filter",
                    paper_id=paper_id,
                )
            except Exception:
                log.warning(
                    "scoped retrieval failed", paper_id=paper_id, exc_info=True
                )
                return []
        try:
            return list(self.inner.retrieve(query, paper_id, k=k))
        except Exception:
            log.warning("retrieval failed", paper_id=paper_id, exc_info=True)
            return []


def scoped_retriever(
    inner: Retriever, page_range: Optional[tuple[int, int]]
) -> Retriever:
    """Return ``inner`` wrapped in a :class:`PageScopedRetriever`, or as-is.

    Convenience for call sites that carry an optional page range and shouldn't
    have to branch.
    """
    if page_range is None:
        return inner
    return PageScopedRetriever(
        inner, page_start=page_range[0], page_end=page_range[1]
    )
