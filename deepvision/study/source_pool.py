"""Source pool — the material flashcards and quizzes are actually built from.

Both generators used to read **only the persisted report**
(``quiz_agent.collect_sources(sections)``, ``flashcard_agent._collect(report)``).
That is why every regenerated deck and quiz felt like the same deck and quiz,
and the cause is arithmetic rather than opinion. Measured over this project's
own library:

===========  ==============  ==============  ==================================
paper        report chars    paper chars     report as a share of the source
===========  ==============  ==============  ==================================
1002-2191    9,236           25,208          37%
2304-06790   6,853           20,045          34%
1706-03762   13,823          45,788          30%
===========  ==============  ==============  ==================================

So roughly **two thirds of every paper was invisible** to the study tools — and
the third they could see was the *summarised* third, i.e. the same handful of
claims restated. Worse, the report is a fixed artifact: re-rolling the RNG on
regeneration reshuffles a pool that never changes, so a "new" quiz can only be a
new arrangement of the same questions.

This module supplies the missing two thirds. It sweeps the paper's whole chunk
corpus, cleans it with :mod:`deepvision.rag.chunk_quality` (so the junk terms
that produced real cards fronted ``**AI**``, ``**BS**``, ``**CL**`` and
``**CVF**`` never become study material), and hands back a page-ordered pool the
generators can draw breadth from. The report stays in the mix — it is the
quality-filtered, citation-bearing part — but it is no longer the whole world.

**No LLM here.** Retrieval plus deterministic filtering only, so this cannot
fail in a way that invents study material.
"""

from __future__ import annotations

from typing import Optional, Sequence

from deepvision.models.chunks import AnyChunk, Modality
from deepvision.rag.chunk_quality import narrative_chunks
from deepvision.rag.retrieval import Retriever
from deepvision.utils.logger import get_logger

__all__ = [
    "SWEEP_QUERIES",
    "pool_for_paper",
    "build_paper_pool",
    "round_robin_by_page",
]

log = get_logger(__name__)

#: Broad queries chosen to sweep a paper's whole shape rather than answer any
#: one question. There is no "give me every chunk" call on the vector store —
#: only similarity search — so breadth has to be manufactured by asking about
#: every part of a paper at once and unioning the results. Deliberately generic:
#: these are not meant to be *good* queries, they are meant to be *disjoint*
#: ones, and each pulls a different region of the document.
SWEEP_QUERIES: tuple[str, ...] = (
    "introduction motivation problem statement why this matters",
    "background related work prior approaches previous methods",
    "definition terminology notation concept we define refers to",
    "method approach architecture algorithm implementation steps",
    "experiment dataset benchmark setup evaluation protocol",
    "results performance accuracy comparison baseline improvement",
    "figure table chart diagram shows illustrates",
    "limitation drawback assumption future work open question",
    "conclusion discussion implication application impact",
)

#: How many chunks each sweep query asks for. High on purpose — the union is
#: de-duplicated and then quality-filtered, so over-asking costs a little
#: compute and buys coverage.
_K_PER_QUERY = 60

#: Hard ceiling on the returned pool, so a book-length upload cannot hand a
#: generator an unbounded list.
_DEFAULT_LIMIT = 400


def build_paper_pool(
    paper_id: str,
    retriever: Retriever,
    *,
    limit: int = _DEFAULT_LIMIT,
    queries: Sequence[str] = SWEEP_QUERIES,
) -> list[AnyChunk]:
    """Return a cleaned, page-ordered pool of this paper's text chunks.

    Unions :data:`SWEEP_QUERIES` (each a different region of a paper), keeps
    text-bearing modalities, runs the result through
    :func:`~deepvision.rag.chunk_quality.narrative_chunks` to drop captions,
    running headers, title blocks and page furniture, then sorts by page so a
    caller can take an even spread across the document instead of whatever
    similarity search happened to rank first.

    Never raises: a failing query contributes nothing and is logged, and a
    completely failed sweep returns ``[]`` so the caller falls back to the
    report exactly as before.
    """
    seen: dict[str, AnyChunk] = {}
    for query in queries:
        try:
            for chunk in retriever.retrieve(query, paper_id, k=_K_PER_QUERY):
                seen.setdefault(chunk.id, chunk)
        except Exception:  # noqa: BLE001 - one bad query must not kill the sweep
            log.warning(
                "source-pool sweep query failed",
                extra={"paper_id": paper_id, "query": query},
                exc_info=True,
            )

    # Image chunks carry no prose to build a card from; vision/OCR text does.
    textual = [
        c for c in seen.values()
        if c.modality is not Modality.IMAGE and (c.text or "").strip()
    ]
    if not textual:
        log.info("source pool is empty", extra={"paper_id": paper_id})
        return []

    cleaned = narrative_chunks(textual, allow_fallback=False)
    if not cleaned:
        # Everything was furniture. Better to study the raw text than nothing.
        cleaned = textual

    cleaned.sort(key=lambda c: (getattr(c, "page", 0) or 0, c.id))
    log.info(
        "built source pool",
        extra={
            "paper_id": paper_id,
            "swept": len(seen),
            "kept": min(len(cleaned), limit),
        },
    )
    return cleaned[:limit]


def round_robin_by_page(
    chunks: Sequence[AnyChunk], *, buckets: int = 6
) -> list[AnyChunk]:
    """Reorder ``chunks`` so consecutive picks come from different pages.

    A generator that takes the first N candidates off a relevance- or
    page-ordered list ends up drilling one region of the paper — every card
    about the method, nothing about the results. Splitting the document into
    ``buckets`` page ranges and interleaving them means "take the first 20"
    spans the whole paper.

    Stable and deterministic: no RNG, so the same pool always yields the same
    order, and the *variety* comes from the pool being bigger rather than from
    shuffling a small one.
    """
    if not chunks:
        return []
    pages = [getattr(c, "page", 0) or 0 for c in chunks]
    low, high = min(pages), max(pages)
    span = max(1, high - low + 1)
    width = max(1, span // max(1, buckets))

    grouped: dict[int, list[AnyChunk]] = {}
    for chunk in chunks:
        page = getattr(chunk, "page", 0) or 0
        grouped.setdefault(min((page - low) // width, buckets - 1), []).append(chunk)

    out: list[AnyChunk] = []
    index = 0
    while len(out) < len(chunks):
        progressed = False
        for key in sorted(grouped):
            bucket = grouped[key]
            if index < len(bucket):
                out.append(bucket[index])
                progressed = True
        if not progressed:
            break
        index += 1
    return out


def pool_for_paper(
    paper_id: str, retriever: Optional[Retriever], *, limit: int = _DEFAULT_LIMIT
) -> list[AnyChunk]:
    """:func:`build_paper_pool` + :func:`round_robin_by_page`, or ``[]``.

    The convenience entry point the generators call. A ``None`` retriever (the
    caller could not build one) yields an empty pool rather than an error, which
    degrades the generators to their old report-only behaviour.
    """
    if retriever is None:
        return []
    try:
        return round_robin_by_page(build_paper_pool(paper_id, retriever, limit=limit))
    except Exception:  # noqa: BLE001 - study generation must never break on this
        log.warning(
            "could not build a source pool; falling back to the report alone",
            extra={"paper_id": paper_id},
            exc_info=True,
        )
        return []
