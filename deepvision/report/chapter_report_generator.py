"""Per-chapter report generation and persistence.

A chapter report is the *same* eleven-section :class:`~deepvision.models.Report`
as a whole-paper report — same agents, same ``SECTION_ORDER``, same
``SectionCard`` render path in the UI — with exactly two things changed:

1. the retriever handed to the agent ensemble is a
   :class:`~deepvision.rag.chapter_scope.PageScopedRetriever`, so every chunk
   the agents ever see comes from the chapter's page range, and
2. the result is stamped with ``scope='chapter'`` + the ``chapter_*`` fields and
   persisted to the ``chapter_reports`` table, keyed by
   ``(paper_id, chapter_id)``.

Two deliberate deltas from the whole-paper path, both easy to get wrong:

- ``ReportStats.pages`` is the **chapter's** page span, not the document's.
- ``_sync_paper_counts`` is **not** called and ``ReportRow`` is never touched —
  a chapter report must never overwrite the paper's own counts or its
  whole-paper report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from deepvision.agents.agent_orchestrator import AgentOrchestrator
from deepvision.db import session_scope
from deepvision.db.schema import ChapterReportRow, PaperRow
from deepvision.models import (
    SECTION_ORDER,
    AppSettings,
    Chapter,
    ChapterSource,
    Report,
    ReportScope,
    ReportStats,
    Section,
)
from deepvision.rag.chapter_scope import PageScopedRetriever
from deepvision.report.agent_bridge import build_retriever
from deepvision.report.interactive_sections import (
    build_section,
    normalize_sections,
    sections_from_rows,
)
from deepvision.report.report_generator import (
    PaperNotFoundError,
    compute_section_stats,
    model_used_for,
    paper_meta_from_row,
    report_exists_in_session,
)
from deepvision.utils import get_logger
from deepvision.utils.ids import report_id

__all__ = [
    "ChapterReportGenerator",
    "ChapterNotFoundError",
    "resolve_chapter",
    "list_chapters",
]

log = get_logger(__name__)

#: Optional ``(percent, message)`` progress sink used by the job runner.
ProgressFn = Callable[[int, str], None]


class ChapterNotFoundError(LookupError):
    """Raised when a chapter_id cannot be resolved against the paper's PDF."""


def list_chapters(paper_id: str) -> list[Chapter]:
    """Return the paper's derived chapters, or ``[]`` if unavailable.

    Chapters are derived on demand from ``data/<paper_id>/main.pdf`` by Domain
    A's ``ingestion.chapters.derive_chapters`` — they are never stored. The
    import is done lazily (and defensively) so this module, the report router,
    and therefore the whole app still import cleanly if that module is absent;
    the caller then simply sees "no chapters", which surfaces as a 404 on an
    unknown chapter id rather than an import-time crash.
    """
    try:
        from deepvision.ingestion.chapters import derive_chapters
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(
            "chapter detection unavailable",
            extra={"paper_id": paper_id, "error": str(exc)},
        )
        return []
    try:
        return list(derive_chapters(paper_id) or [])
    except Exception as exc:  # pragma: no cover - derive_chapters never raises
        log.warning(
            "chapter detection failed",
            extra={"paper_id": paper_id, "error": str(exc)},
        )
        return []


def resolve_chapter(paper_id: str, chapter_id: str) -> Chapter:
    """Resolve ``chapter_id`` to a :class:`Chapter` of ``paper_id``.

    The frontend always posts back an id it was given by
    ``GET /papers/{id}/chapters`` — never a title — so an id that no longer
    resolves means the PDF was re-ingested and the chapters re-derived. That is
    a 404, not a guess: resolving by title instead would silently report on the
    wrong pages.
    """
    for chapter in list_chapters(paper_id):
        if chapter.id == chapter_id:
            return chapter
    raise ChapterNotFoundError(
        f"unknown chapter_id {chapter_id!r} for paper {paper_id!r}"
    )


class ChapterReportGenerator:
    """Builds, persists, and loads reports scoped to one chapter of a paper."""

    def generate(
        self,
        paper_id: str,
        chapter: Chapter,
        settings: AppSettings,
        *,
        on_progress: Optional[ProgressFn] = None,
    ) -> Report:
        """Build and persist the chapter report for ``chapter`` of ``paper_id``.

        Idempotent per ``(paper_id, chapter_id)``: re-generating overwrites the
        existing ``chapter_reports`` row instead of appending a second one.
        """
        glog = log.bind(paper_id=paper_id, chapter_id=chapter.id)
        page_range = (chapter.page_start, chapter.page_end)

        with session_scope() as session:
            paper_row = session.get(PaperRow, paper_id)
            if paper_row is None:
                raise PaperNotFoundError(f"paper not found: {paper_id}")
            # has_report tracks the WHOLE-PAPER report row; a chapter report
            # existing says nothing about it, so look it up rather than default.
            paper_meta = paper_meta_from_row(
                paper_row, has_report=report_exists_in_session(session, paper_id)
            )

        report: Optional[Report] = None
        try:
            # The ONLY difference from the whole-paper pipeline: the retriever
            # is wrapped so nothing outside the chapter's pages can reach any
            # agent. `page_range` is then passed explicitly to build_report as
            # the independent second guarantee (and to scope the Figures group).
            retriever = PageScopedRetriever(
                build_retriever(settings),
                page_start=chapter.page_start,
                page_end=chapter.page_end,
            )
            report = AgentOrchestrator(retriever).build_report(
                paper_id, settings, page_range=page_range
            )
        except Exception as exc:
            glog.error(
                "chapter agent pipeline failed; using fallback chapter report",
                extra={"error": str(exc)},
            )

        if report is None:
            report = _fallback_chapter_report(paper_id, chapter)

        report.id = report.id or report_id()
        report.paper_id = paper_id
        report.paper = report.paper or paper_meta
        report.sections = normalize_sections(report.sections)
        report.model_used = report.model_used or model_used_for(settings)
        report.generated_at = report.generated_at or datetime.utcnow()

        report.scope = ReportScope.CHAPTER
        report.chapter_id = chapter.id
        report.chapter_title = chapter.title
        report.chapter_page_start = chapter.page_start
        report.chapter_page_end = chapter.page_end

        # stats.pages is the CHAPTER's span, not the document's. figure_count is
        # deliberately not passed: the paper-wide extraction count is not a
        # meaningful fallback for one chapter.
        report.stats = compute_section_stats(
            report.sections, page_count=chapter.page_count
        )

        if on_progress:
            _safe_progress(on_progress, 90, "Saving chapter report")

        _persist_chapter_report(report, chapter)
        glog.info(
            "chapter report generated",
            extra={
                "pages": f"{chapter.page_start}-{chapter.page_end}",
                "citations": report.stats.citations_extracted,
                "figures": report.stats.figures,
            },
        )
        return report

    def load(self, paper_id: str, chapter_id: str) -> Report:
        """Load the persisted chapter report, or raise :class:`LookupError`.

        Reads only ``chapter_reports`` — the denormalized ``chapter_*`` columns
        mean the PDF never has to be re-opened to render a stored report, so
        this keeps working even if the chapter list would derive differently
        today.
        """
        with session_scope() as session:
            row = session.exec(
                select(ChapterReportRow)
                .where(ChapterReportRow.paper_id == paper_id)
                .where(ChapterReportRow.chapter_id == chapter_id)
            ).first()
            if row is None:
                raise LookupError(
                    f"chapter report not found for paper_id={paper_id!r} "
                    f"chapter_id={chapter_id!r}"
                )
            paper_row = session.get(PaperRow, paper_id)
            paper_meta = (
                paper_meta_from_row(
                    paper_row, has_report=report_exists_in_session(session, paper_id)
                )
                if paper_row
                else None
            )
            report = Report(
                id=row.id,
                paper_id=row.paper_id,
                paper=paper_meta,
                stats=ReportStats.model_validate(row.stats or {}),
                sections=sections_from_rows(row.sections or []),
                generated_at=row.generated_at,
                model_used=row.model_used,
                scope=ReportScope.CHAPTER,
                chapter_id=row.chapter_id,
                chapter_title=row.chapter_title,
                chapter_page_start=row.chapter_page_start,
                chapter_page_end=row.chapter_page_end,
            )
        # Same normalisation as the whole-paper path: always eleven sections in
        # SECTION_ORDER, so the frontend never has to handle a short list.
        report.sections = normalize_sections(report.sections)
        return report

    def exists(self, paper_id: str, chapter_id: str) -> bool:
        """True if a chapter report is already persisted for the pair."""
        with session_scope() as session:
            row = session.exec(
                select(ChapterReportRow)
                .where(ChapterReportRow.paper_id == paper_id)
                .where(ChapterReportRow.chapter_id == chapter_id)
            ).first()
            return row is not None


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------
def _safe_progress(fn: ProgressFn, percent: int, message: str) -> None:
    try:
        fn(percent, message)
    except Exception:  # pragma: no cover - a progress sink must never break a build
        log.warning("chapter progress callback raised", extra={"message": message})


def _fallback_chapter_report(paper_id: str, chapter: Chapter) -> Report:
    """A valid, honest eleven-section report used when the pipeline is unavailable.

    Mirrors ``report_generator._fallback_report``: degrade, never 500. The body
    says what actually happened instead of leaving empty strings behind.
    """
    note = (
        f"The chapter report for **{chapter.title}** (pages "
        f"{chapter.page_start}-{chapter.page_end}) could not be generated: the "
        "summarization pipeline is unavailable. Nothing outside this chapter's "
        "pages was substituted. Try again once the models are configured."
    )
    return Report(
        id=report_id(),
        paper_id=paper_id,
        stats=ReportStats(pages=chapter.page_count),
        sections=[build_section(name=name, body_markdown=note) for name in SECTION_ORDER],
        model_used=None,
    )


def _persist_chapter_report(report: Report, chapter: Chapter) -> None:
    """Upsert the ``chapter_reports`` row for ``(paper_id, chapter_id)``.

    Never touches ``reports`` (the whole-paper row) or ``papers``.

    The select-then-insert is not atomic, so a second writer that slipped past
    the job runner's in-flight guard can lose the ``INSERT`` to the
    ``(paper_id, chapter_id)`` unique constraint. That must not fail the job —
    the row it wanted *does* now exist — so an ``IntegrityError`` is retried
    once, which takes the update branch. Degrade, never crash.
    """
    try:
        _upsert_chapter_report_row(report, chapter)
    except IntegrityError:
        log.warning(
            "chapter report row was written concurrently; retrying as an update",
            extra={"paper_id": report.paper_id, "chapter_id": chapter.id},
        )
        _upsert_chapter_report_row(report, chapter)


def _upsert_chapter_report_row(report: Report, chapter: Chapter) -> None:
    """One select-then-insert-or-update pass over ``chapter_reports``."""
    sections_payload = [s.model_dump(mode="json") for s in report.sections]
    stats_payload = report.stats.model_dump(mode="json")
    source = chapter.source
    source_value = source.value if isinstance(source, ChapterSource) else str(source)

    with session_scope() as session:
        existing = session.exec(
            select(ChapterReportRow)
            .where(ChapterReportRow.paper_id == report.paper_id)
            .where(ChapterReportRow.chapter_id == chapter.id)
        ).first()
        if existing is not None:
            existing.id = report.id
            existing.chapter_title = chapter.title
            existing.chapter_level = chapter.level
            existing.chapter_page_start = chapter.page_start
            existing.chapter_page_end = chapter.page_end
            existing.chapter_source = source_value
            existing.stats = stats_payload
            existing.sections = sections_payload
            existing.model_used = report.model_used
            existing.generated_at = report.generated_at
            session.add(existing)
            return
        session.add(
            ChapterReportRow(
                id=report.id,
                paper_id=report.paper_id,
                chapter_id=chapter.id,
                chapter_title=chapter.title,
                chapter_level=chapter.level,
                chapter_page_start=chapter.page_start,
                chapter_page_end=chapter.page_end,
                chapter_source=source_value,
                stats=stats_payload,
                sections=sections_payload,
                model_used=report.model_used,
                generated_at=report.generated_at,
            )
        )
