"""Report generator — produce and persist a paper's Report.

Resolves a concrete :class:`~deepvision.agents.agent_orchestrator.AgentOrchestrator`
(see :mod:`deepvision.report.agent_bridge` for why this isn't a hardcoded
import), drives it to build the report, normalizes/repairs the section list
per the fixed design, computes the :class:`ReportStats` bar, and persists the
result to the ``reports`` table (JSON columns). ``generate`` is idempotent —
re-running it for the same paper overwrites the existing row rather than
duplicating it.
"""

from __future__ import annotations

import abc
import math
from datetime import date, datetime
from typing import Optional, Sequence

from sqlmodel import select

from deepvision.agents.agent_orchestrator import AgentOrchestrator
from deepvision.config import get_config
from deepvision.db import session_scope
from deepvision.db.schema import PaperRow, ReportRow
from deepvision.models import (
    SECTION_ORDER,
    AppSettings,
    PaperMeta,
    PaperStatus,
    ProviderMode,
    Report,
    ReportStats,
    Section,
    SectionName,
)
from deepvision.report.agent_bridge import build_retriever
from deepvision.report.interactive_sections import (
    build_section,
    normalize_sections,
    sections_from_rows,
)
from deepvision.utils import get_logger
from deepvision.utils.ids import report_id

__all__ = [
    "ReportGenerator",
    "DefaultReportGenerator",
    "PaperNotFoundError",
    "ReportDegradedError",
    "compute_section_stats",
    "paper_meta_from_row",
    "report_exists_in_session",
    "model_used_for",
]

log = get_logger(__name__)

#: Average adult silent-reading speed, used for ReportStats.reading_time_min.
WORDS_PER_MINUTE = 200


class PaperNotFoundError(LookupError):
    """Raised when a report is requested/generated for an unknown paper_id."""


class ReportDegradedError(RuntimeError):
    """The agent ensemble failed; a *placeholder* report was persisted instead.

    ``generate()`` still degrades gracefully — it never leaves the paper without
    a renderable report row, and ``.report`` on this exception is that persisted
    placeholder, so ``GET /report/{paper_id}`` keeps returning 200. What this
    exception adds is **audibility**: the ingestion job's ``Generate report``
    stage exists precisely so a failed report is visible in the UI, and that
    only works if the failure actually reaches the caller. Absorbing the agent
    exception here and returning a placeholder as if it were a real report is
    the original "empty report, no error anywhere" bug wearing a new hat.

    Callers that only need *something* to render should catch this and use
    ``.report``; callers that report status (the orchestrator's report stage)
    should let it mark the stage errored.
    """

    def __init__(self, message: str, report: Report) -> None:
        super().__init__(message)
        #: The placeholder report that was persisted despite the failure.
        self.report = report


class ReportGenerator(abc.ABC):
    """Generates, persists, and loads reports."""

    @abc.abstractmethod
    def generate(self, paper_id: str, settings: AppSettings) -> Report:
        """Build and persist the full report for ``paper_id``."""
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, paper_id: str) -> Report:
        """Load the persisted report for ``paper_id`` (404 if absent)."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_stats(self, paper_id: str) -> ReportStats:
        """Compute the stats bar (pages, figures, citations, reading time)."""
        raise NotImplementedError


class DefaultReportGenerator(ReportGenerator):
    """Concrete generator: orchestrates the agents layer's agents, computes stats, persists."""

    def generate(self, paper_id: str, settings: AppSettings) -> Report:
        glog = log.bind(paper_id=paper_id)

        with session_scope() as session:
            paper_row = session.get(PaperRow, paper_id)
            if paper_row is None:
                raise PaperNotFoundError(f"paper not found: {paper_id}")
            paper_meta = _paper_meta_from_row(
                paper_row, has_report=report_exists_in_session(session, paper_id)
            )
            page_count = paper_row.page_count
            figure_count = paper_row.figure_count

        report: Optional[Report] = None
        agent_error: Optional[Exception] = None
        try:
            # AgentOrchestrator is the concrete class named in;
            # it takes a Retriever (NOT an LLM) and builds its own LLM per call.
            # (The old resolve_concrete(AgentOrchestrator) path always returned
            # None — it searches for *subclasses*, and there are none — so the
            # report silently degraded to the placeholder fallback with no
            # retrieval/citations/media. Wire it directly instead.)
            retriever = build_retriever(settings)
            orchestrator = AgentOrchestrator(retriever)
            report = orchestrator.build_report(paper_id, settings)
        except Exception as exc:
            glog.error(
                "agent orchestrator failed; using fallback report",
                extra={"error": str(exc)},
            )
            agent_error = exc

        if report is None:
            report = _fallback_report(paper_id, paper_meta)

        report.id = report.id or report_id()
        report.paper_id = paper_id
        report.paper = report.paper or paper_meta
        report.sections = normalize_sections(report.sections)
        report.model_used = report.model_used or _model_used(settings)
        if report.generated_at is None:
            report.generated_at = datetime.utcnow()
        report.stats = _compute_stats(
            report.sections, page_count=page_count, figure_count=figure_count
        )

        _persist(report)
        _sync_paper_counts(paper_id, report.stats)
        if report.paper is not None:
            # A report row for this paper now definitively exists (placeholder
            # or not), so the embedded PaperMeta must say so — it was built
            # before _persist ran, and an agent-built one may have been loaded
            # even earlier.
            report.paper.has_report = True

        if agent_error is not None:
            # The placeholder is persisted (nothing 500s, chat/compare/report
            # page all keep working) — but the caller MUST be told, or the
            # ingestion job's "Generate report" stage reports `done` for a
            # report that is a stub. That is exactly the "empty report, no
            # error anywhere in the UI" bug the tracked stage exists to kill.
            raise ReportDegradedError(
                f"agent pipeline failed, placeholder report saved: {agent_error}",
                report,
            )
        return report

    def load(self, paper_id: str) -> Report:
        with session_scope() as session:
            report_row = session.exec(
                select(ReportRow).where(ReportRow.paper_id == paper_id)
            ).first()
            if report_row is None:
                raise LookupError(f"report not found for paper_id={paper_id!r}")
            paper_row = session.get(PaperRow, paper_id)
            # report_row is not None here, so has_report is True by definition.
            paper_meta = (
                _paper_meta_from_row(paper_row, has_report=True) if paper_row else None
            )
            report = Report(
                id=report_row.id,
                paper_id=report_row.paper_id,
                paper=paper_meta,
                stats=ReportStats.model_validate(report_row.stats or {}),
                sections=sections_from_rows(report_row.sections or []),
                generated_at=report_row.generated_at,
                model_used=report_row.model_used,
            )
        # Reports persisted before the study-oriented sections existed only have
        # the original five rows. Normalising on load backfills the missing names
        # as empty placeholders (and re-applies default_open/badge), so
        # GET /report always returns one section per SECTION_ORDER entry and the
        # frontend never has to handle a short list. Content still needs a re-run.
        report.sections = normalize_sections(report.sections)
        return report

    def compute_stats(self, paper_id: str) -> ReportStats:
        with session_scope() as session:
            report_row = session.exec(
                select(ReportRow).where(ReportRow.paper_id == paper_id)
            ).first()
            if report_row is None:
                raise LookupError(f"report not found for paper_id={paper_id!r}")
            paper_row = session.get(PaperRow, paper_id)
            sections = sections_from_rows(report_row.sections or [])
            page_count = paper_row.page_count if paper_row else None
            figure_count = paper_row.figure_count if paper_row else None
        return _compute_stats(sections, page_count=page_count, figure_count=figure_count)


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------
def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None


def report_exists_in_session(session, paper_id: str) -> bool:
    """True iff a whole-paper ``ReportRow`` exists, using an OPEN session.

    ``PaperMeta.has_report`` is derived, not stored, so every ``PaperRow`` →
    ``PaperMeta`` conversion needs this answer. Taking the caller's session
    keeps it to one extra indexed lookup and avoids nesting a second
    ``session_scope`` inside an open one. (Per-*chapter* reports live in their
    own table and deliberately do not count — ``has_report`` gates the Library's
    whole-paper "Open report" button.)
    """
    return (
        session.exec(
            select(ReportRow.paper_id).where(ReportRow.paper_id == paper_id)
        ).first()
        is not None
    )


def _paper_meta_from_row(row: PaperRow, *, has_report: bool = False) -> PaperMeta:
    """``PaperRow`` → :class:`PaperMeta`.

    ``has_report`` is derived from the ``reports`` table, never from ``row``
    (there is no such column), so the caller — which is always inside a session
    that already knows the answer — supplies it.
    """
    return PaperMeta(
        id=row.id,
        arxiv_id=row.arxiv_id,
        arxiv_label=row.arxiv_label,
        version=row.version,
        title=row.title,
        authors=list(row.authors or []),
        abstract=row.abstract,
        categories=list(row.categories or []),
        published=_parse_date(row.published),
        updated=_parse_date(row.updated),
        pdf_url=row.pdf_url,
        abs_url=row.abs_url,
        status=PaperStatus(row.status) if row.status else PaperStatus.QUEUED,
        ingested=row.ingested,
        has_report=has_report,
        thumbnail_path=row.thumbnail_path,
        page_count=row.page_count,
        figure_count=row.figure_count,
        error_message=row.error_message,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _word_count(markdown: str) -> int:
    return len(markdown.split())


def _compute_stats(
    sections: Sequence[Section],
    *,
    page_count: Optional[int],
    figure_count: Optional[int],
) -> ReportStats:
    """Compute the stats bar for ``sections``.

    ``stats.figures`` counts the figures the report *actually renders* — the
    media cards in the Figures section — not ``PaperRow.figure_count``, which is
    the raw number of image crops ingestion pulled out of the PDF. Those two
    numbers legitimately differ: a crop only becomes a card if retrieval
    surfaced it and it survived caption-joining. Pairing the raw count with the
    rendered cards is what produced the "7 of 10 shown" mismatch in the UI, so
    the stat must be derived from the same list the cards come from.

    ``figure_count`` (the raw extraction count) is kept only as a last-resort
    fallback for the pathological case of a report with no Figures section at
    all; it stays available in its own right on ``PaperMeta.figure_count``.
    """
    total_words = 0
    citations_extracted = 0
    media_count = 0
    figures_media = 0
    has_figures_section = False
    for sec in sections:
        total_words += _word_count(sec.body_markdown) + _word_count(sec.deep_dive_markdown or "")
        citations_extracted += len(sec.citations)
        media_count += len(sec.media)
        if sec.name is SectionName.FIGURES:
            has_figures_section = True
            figures_media += len(sec.media)

    if has_figures_section:
        figures = figures_media
    elif media_count:
        figures = media_count
    else:
        figures = figure_count or 0

    reading_time = math.ceil(total_words / WORDS_PER_MINUTE) if total_words else 0
    return ReportStats(
        pages=page_count or 0,
        figures=figures,
        citations_extracted=citations_extracted,
        reading_time_min=reading_time,
    )


def compute_section_stats(
    sections: Sequence[Section],
    *,
    page_count: Optional[int],
    figure_count: Optional[int] = None,
) -> ReportStats:
    """Public wrapper over the stats computation, shared with chapter reports.

    ``deepvision.report.chapter_report_generator`` computes the *same* stats bar
    from the same rules; the only difference is what it passes as ``page_count``
    (the chapter's page span rather than the document's).
    """
    return _compute_stats(sections, page_count=page_count, figure_count=figure_count)


def paper_meta_from_row(row: PaperRow, *, has_report: Optional[bool] = None) -> PaperMeta:
    """Public alias of the ``PaperRow`` → :class:`PaperMeta` conversion.

    See :func:`_paper_meta_from_row` for why ``has_report`` is a parameter.

    Unlike the private helper (whose callers are all already inside a session
    that knows the answer), this exported entry point **defaults to looking the
    answer up** rather than to ``False``. Defaulting to ``False`` here silently
    emits ``has_report=False`` for a paper that does have a report — the exact
    "right in one response, wrong in another" failure ``has_report`` was added
    to avoid — and it made this helper disagree with its identically-named
    sibling ``agents.base.paper_meta_from_row``, which does look it up. Pass the
    flag explicitly whenever you already know it; that stays free.
    """
    if has_report is None:
        with session_scope() as session:
            has_report = report_exists_in_session(session, row.id)
    return _paper_meta_from_row(row, has_report=bool(has_report))


def model_used_for(settings: AppSettings) -> str:
    """Public alias: the model id to record on a report built with ``settings``."""
    return _model_used(settings)


def _model_used(settings: AppSettings) -> str:
    cfg = get_config()
    if settings.llm_mode is ProviderMode.API:
        return settings.llm_model or cfg.api_llm_model
    return settings.llm_model or cfg.local_llm_model


def _fallback_report(paper_id: str, paper_meta: PaperMeta) -> Report:
    """A minimal-but-valid Report used when the agent ensemble produced nothing.

    Keeps ``GET /report/{paper_id}`` (and the ingest job's report stage, which
    calls ``generate``) from hard-crashing when the agents layer's agents are missing or
    their model backend is down — degrade gracefully, never 500 the whole page.

    The note must name a recovery that actually works: re-opening the report
    page does **not** regenerate anything (``load()`` returns this very
    placeholder once it is persisted), so it points at re-running ingestion.
    """
    note = (
        "Automated summarization did not produce a report for this paper — the "
        "agent pipeline is unavailable (no model backend configured, or it "
        "failed). This is a placeholder: re-run this paper from the Library to "
        "try again."
    )
    overview = build_section(name=SectionName.OVERVIEW, body_markdown=note)
    sections = [overview] + [
        build_section(name=name) for name in SECTION_ORDER if name != SectionName.OVERVIEW
    ]
    return Report(
        id=report_id(),
        paper_id=paper_id,
        paper=paper_meta,
        stats=ReportStats(),
        sections=sections,
        model_used=None,
    )


def _persist(report: Report) -> None:
    with session_scope() as session:
        existing = session.exec(
            select(ReportRow).where(ReportRow.paper_id == report.paper_id)
        ).first()
        sections_payload = [s.model_dump(mode="json") for s in report.sections]
        stats_payload = report.stats.model_dump(mode="json")
        if existing is not None:
            existing.id = report.id
            existing.stats = stats_payload
            existing.sections = sections_payload
            existing.model_used = report.model_used
            existing.generated_at = report.generated_at
            session.add(existing)
        else:
            session.add(
                ReportRow(
                    id=report.id,
                    paper_id=report.paper_id,
                    stats=stats_payload,
                    sections=sections_payload,
                    model_used=report.model_used,
                    generated_at=report.generated_at,
                )
            )


def _sync_paper_counts(paper_id: str, stats: ReportStats) -> None:
    """Reflect freshly-computed stats back onto PaperRow (ingested flag, page count).

    ``PaperRow.figure_count`` is deliberately **not** written here. It is
    ingestion's raw count of extracted image crops, while ``stats.figures`` is
    now the number of figures the report renders — overwriting the former with
    the latter would destroy the extraction count and, worse, make it shrink on
    every re-generation.
    """
    with session_scope() as session:
        row = session.get(PaperRow, paper_id)
        if row is None:
            return
        row.ingested = True
        if stats.pages:
            row.page_count = stats.pages
        row.updated_at = datetime.utcnow()
        session.add(row)
