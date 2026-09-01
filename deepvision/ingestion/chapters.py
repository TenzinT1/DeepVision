"""Chapter detection — derives a picker-ready chapter list from a paper's PDF.. Pure PyMuPDF work, **no model calls**, so it's cheap
enough to run on every Report-page mount. Three detection tiers, in order:

1. ``outline`` — the PDF's own embedded outline/bookmarks
   (``fitz.Document.get_toc()``). Authoritative when present: real titles, real
   start pages, real nesting.
2. ``headings`` — reconstructed from the same heading heuristic
   ``ingestion/pdf_parser.py`` already tags every ``TextChunk`` with
   (``_looks_like_heading``), taken in page order. This is the arXiv case:
   papers almost never ship an outline but do have detectable headings.
3. Page-range buckets ("Pages 1-10", "Pages 11-20", ...) as a last resort, so
   the dropdown is never empty for a multi-page PDF with neither an outline
   nor recognizable headings. Tagged ``ChapterSource.HEADINGS`` (the frozen
   enum has no third value; buckets are, like headings, a non-outline
   heuristic derivation).

Fewer than 2 chapters at the end of all three tiers => ``[]`` (a one-item
picker is just the whole paper, which the plain report already covers).

Chapter ids are always minted via :meth:`Chapter.make_id` — stable across
re-derivations of the same PDF, because they are half of the persisted
``chapter_reports`` primary key. Results are memoised in-process, keyed by
``(paper_id, pdf mtime_ns, size)``, so a re-ingest (which rewrites the PDF)
invalidates the cache for free.

This module must **never raise**: a missing PDF, a corrupt PDF, or a fitz
import failure all degrade to an empty chapter list (log + fall back), per the
project's hard "never crash the pipeline" rule.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

from deepvision.ingestion.paths import pdf_path
from deepvision.ingestion.pdf_parser import PDFParserService
from deepvision.models.report import Chapter, ChapterSource
from deepvision.utils import get_logger

__all__ = ["derive_chapters", "page_count_of"]

log = get_logger(__name__)

#: Sanity cap so a corrupt/huge outline never produces an unusable dropdown.
_MAX_CHAPTERS = 300
_MIN_TITLE_LEN = 3
_MAX_OUTLINE_LEVEL = 2
_BUCKET_SIZE = 10

# Reuse the non-content section-label filter from research_agent (references,
# bibliography, acknowledgements, appendix, ...) rather than maintaining a
# second copy of the same regex. Fall back to a local copy if the agents
# module ever fails to import, so chapter extraction stays usable either way.
try:
    from deepvision.agents.research_agent import (  # type: ignore
        _NONCONTENT_LABEL_RE as _NONCONTENT_RE,
    )
except Exception:  # pragma: no cover - defensive
    _NONCONTENT_RE = re.compile(
        r"\b(reference|bibliograph|acknowledg|appendix|author\s+contribution|"
        r"funding|conflict|supplementary|visuali[sz]ation)",
        re.IGNORECASE,
    )

# In-process memo: paper_id -> (mtime_ns, size, page_count, chapters).
_CacheEntry = tuple[int, int, int, list[Chapter]]
_CACHE: dict[str, _CacheEntry] = {}


def derive_chapters(paper_id: str) -> list[Chapter]:
    """Return the chapter list for ``paper_id``'s PDF, or ``[]`` if unusable.

    Never raises. Memoised on the PDF's ``(mtime_ns, size)`` so repeated
    dropdown opens don't re-parse; a re-ingested PDF invalidates the cache.
    """
    try:
        entry = _load(paper_id)
    except Exception as exc:  # pragma: no cover - defensive, belt & suspenders
        log.warning(
            "chapter derivation raised unexpectedly (returning empty list)",
            extra={"paper_id": paper_id, "error": str(exc)},
        )
        return []
    if entry is None:
        return []
    return list(entry[3])


def page_count_of(paper_id: str) -> int:
    """Return the PDF's page count, or ``0`` if it cannot be determined."""
    try:
        entry = _load(paper_id)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(
            "page count lookup raised unexpectedly (returning 0)",
            extra={"paper_id": paper_id, "error": str(exc)},
        )
        return 0
    if entry is None:
        return 0
    return entry[2]


# --------------------------------------------------------------------------
# Internal
# --------------------------------------------------------------------------


def _import_fitz():
    try:
        import fitz  # type: ignore

        return fitz
    except Exception as exc:  # pragma: no cover - environment dependent
        log.warning("fitz unavailable for chapter extraction", extra={"error": str(exc)})
        return None


def _stat(pdf: Path) -> Optional[tuple[int, int]]:
    try:
        st = pdf.stat()
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


def _load(paper_id: str) -> Optional[_CacheEntry]:
    pdf = pdf_path(paper_id)
    stat = _stat(pdf)
    if stat is None:
        log.info("no PDF on disk for chapter extraction", extra={"paper_id": paper_id})
        return None
    mtime_ns, size = stat

    cached = _CACHE.get(paper_id)
    if cached is not None and cached[0] == mtime_ns and cached[1] == size:
        return cached

    fitz = _import_fitz()
    if fitz is None:
        return None

    try:
        doc = fitz.open(str(pdf))
    except Exception as exc:
        log.warning(
            "failed to open PDF for chapter extraction",
            extra={"paper_id": paper_id, "error": str(exc)},
        )
        return None

    try:
        page_count = int(doc.page_count)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(
            "failed to read page count", extra={"paper_id": paper_id, "error": str(exc)}
        )
        doc.close()
        return None

    if page_count < 1:
        doc.close()
        return None

    try:
        chapters = _derive_from_outline(doc, paper_id, page_count)
        if len(chapters) < 2:
            chapters = _derive_from_headings(pdf, paper_id, page_count)
        if len(chapters) < 2:
            chapters = _derive_from_buckets(paper_id, page_count)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(
            "chapter derivation failed (returning empty list)",
            extra={"paper_id": paper_id, "error": str(exc)},
        )
        chapters = []
    finally:
        doc.close()

    if len(chapters) < 2:
        chapters = []

    entry: _CacheEntry = (mtime_ns, size, page_count, chapters)
    _CACHE[paper_id] = entry
    return entry


def _clamp_page(page: int, page_count: int) -> int:
    return max(1, min(int(page), page_count))


def _dedupe_same_page_starts(
    paper_id: str, entries: list[tuple[int, str, int]]
) -> list[tuple[int, str, int]]:
    """Keep only the first entry per ``(level, page_start)`` pair.

    See :func:`_build_chapters` for why. Order is preserved; nothing else is
    touched.
    """
    seen: set[tuple[int, int]] = set()
    kept: list[tuple[int, str, int]] = []
    dropped = 0
    for level, title, page_start in entries:
        key = (level, page_start)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append((level, title, page_start))
    if dropped:
        log.info(
            "collapsed chapter entries sharing a start page",
            extra={"paper_id": paper_id, "dropped": dropped, "kept": len(kept)},
        )
    return kept


def _build_chapters(
    paper_id: str,
    entries: list[tuple[int, str, int]],
    page_count: int,
    source: ChapterSource,
) -> list[Chapter]:
    """Turn ``(level, title, page_start)`` triples (ascending by page) into
    :class:`Chapter`s, computing ``page_end`` per the load-bearing rule:

    ``page_end`` = (``page_start`` of the next entry whose ``level <= this
    level``) - 1; the last chapter (or the last one at its level) runs to
    ``page_count``. This is deliberate: a level-1 chapter must span its own
    level-2 subsections rather than stop where the first one starts.

    Entries are first normalized so that, **at any one level, no two chapters
    share a start page**. Two headings on the same page (very common: "2 Related
    Work" and "3 Method" both land on page 4) would otherwise produce a
    degenerate zero-width range that clamps into an *overlapping* pair, breaking
    the contiguous, non-overlapping invariant the whole page-scoping model rests
    on. Since a chapter's only handle on content is its page range, two chapters
    starting on the same page are indistinguishable to retrieval anyway — the
    first one wins and absorbs the rest of the page. Deeper levels are left
    alone: a level-2 subsection nesting inside its level-1 parent is intended.
    """
    entries = _dedupe_same_page_starts(paper_id, entries)
    n = len(entries)
    chapters: list[Chapter] = []
    for i, (level, title, page_start) in enumerate(entries):
        page_end = page_count
        for j in range(i + 1, n):
            next_level, _next_title, next_page = entries[j]
            if next_level <= level:
                page_end = next_page - 1
                break
        page_end = max(page_start, min(page_end, page_count))
        try:
            chapter = Chapter(
                id=Chapter.make_id(paper_id, title, page_start, level=level),
                index=len(chapters),
                title=title,
                level=level,
                page_start=page_start,
                page_end=page_end,
                source=source,
            )
        except Exception as exc:
            log.warning(
                "dropping invalid chapter entry",
                extra={"paper_id": paper_id, "title": title, "error": str(exc)},
            )
            continue
        chapters.append(chapter)
    return chapters


def _derive_from_outline(doc, paper_id: str, page_count: int) -> list[Chapter]:
    try:
        toc = doc.get_toc(simple=True)
    except Exception as exc:
        log.warning("get_toc failed", extra={"paper_id": paper_id, "error": str(exc)})
        return []
    if not toc:
        return []

    entries: list[tuple[int, str, int]] = []
    seen: set[tuple[int, str, int]] = set()
    for row in toc:
        try:
            level, title, page = row[0], row[1], row[2]
        except (IndexError, TypeError):
            continue
        if page is None or int(page) < 1:
            continue  # unresolved bookmark
        if level is None or int(level) < 1 or int(level) > _MAX_OUTLINE_LEVEL:
            continue
        title = (title or "").strip()
        if len(title) < _MIN_TITLE_LEN:
            continue
        level = int(level)
        page = _clamp_page(int(page), page_count)
        key = (level, title.casefold(), page)
        if key in seen:
            continue
        seen.add(key)
        entries.append((level, title, page))

    if not entries:
        return []

    # The outline is expected in document order already; sort defensively so
    # an out-of-order or malformed outline can't break the page-range math
    # (ascending page, and shallower levels first at a shared page).
    entries.sort(key=lambda e: (e[2], e[0]))

    if len(entries) > _MAX_CHAPTERS:
        log.info(
            "truncating absurdly large outline",
            extra={"paper_id": paper_id, "count": len(entries)},
        )
        entries = entries[:_MAX_CHAPTERS]

    return _build_chapters(paper_id, entries, page_count, ChapterSource.OUTLINE)


def _derive_from_headings(pdf: Path, paper_id: str, page_count: int) -> list[Chapter]:
    try:
        chunks = PDFParserService().parse(str(pdf), paper_id)
    except Exception as exc:
        log.warning(
            "headings fallback parse failed", extra={"paper_id": paper_id, "error": str(exc)}
        )
        return []

    entries: list[tuple[int, str, int]] = []
    seen_labels: set[str] = set()
    for chunk in sorted(chunks, key=lambda c: c.ordinal):
        ref = chunk.source_ref
        if ref is None:
            continue
        label = (ref.section_label or "").strip()
        if not label or label == "Document":
            continue
        key = label.casefold()
        if key in seen_labels:
            continue
        seen_labels.add(key)
        if len(label) < _MIN_TITLE_LEN:
            continue
        if _NONCONTENT_RE.search(label):
            continue
        page = _clamp_page(ref.page, page_count)
        entries.append((1, label, page))

    if not entries:
        return []

    # Chunk ordinal order is document order, so pages are already ascending in
    # practice — sort anyway (stable, so first-seen order is kept within a page)
    # so a multi-column parse that emits an early page late can never produce a
    # descending, and therefore overlapping, chapter list.
    entries.sort(key=lambda e: e[2])

    if len(entries) > _MAX_CHAPTERS:
        entries = entries[:_MAX_CHAPTERS]

    return _build_chapters(paper_id, entries, page_count, ChapterSource.HEADINGS)


def _derive_from_buckets(paper_id: str, page_count: int) -> list[Chapter]:
    """Last-resort page-range buckets ("Pages 1-10", ...), so the picker is
    never empty for a multi-page PDF with neither an outline nor detectable
    headings. Tagged ``HEADINGS`` — the frozen ``ChapterSource`` enum has no
    third value, and buckets are, like headings, a non-outline heuristic.
    """
    if page_count < 2:
        return []

    bucket_size = _BUCKET_SIZE
    if math.ceil(page_count / bucket_size) < 2:
        bucket_size = max(1, page_count // 2)

    entries: list[tuple[int, str, int]] = []
    start = 1
    while start <= page_count and len(entries) < _MAX_CHAPTERS:
        end = min(start + bucket_size - 1, page_count)
        entries.append((1, f"Pages {start}-{end}", start))
        start = end + 1

    return _build_chapters(paper_id, entries, page_count, ChapterSource.HEADINGS)
