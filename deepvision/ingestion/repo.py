"""SQLite row <-> Pydantic model helpers for the ingestion domain.

Centralizes ``PaperRow``/``JobRow`` <-> ``PaperMeta``/``IngestJob`` conversion
and the small set of persistence operations the orchestrator and the
search/ingest/jobs/papers route bodies all need. Kept separate from
``orchestrator.py`` so the route bodies can reuse it without importing the
orchestrator (and its heavier PDF/vision dependencies). internal helper module, not part of the fixed
``deepvision.db.schema`` tables; never redefines their columns.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Optional

from sqlmodel import Session, select

from deepvision.db import session_scope
from deepvision.db.schema import (
    ChapterReportRow,
    ChatRow,
    FlashcardReviewRow,
    FlashcardRow,
    JobRow,
    PaperRow,
    QuizAnswerRow,
    QuizAttemptRow,
    QuizQuestionRow,
    QuizRow,
    ReportRow,
)
from deepvision.models import IngestJob, PaperMeta, PaperStatus
from deepvision.utils import get_logger

log = get_logger(__name__)

#: Plain (non-relationship) PaperRow columns safe to bulk-copy on upsert.
#: Excludes SQLModel relationship attrs ("jobs", "report", "chats") — copying
#: those via setattr would hand SQLAlchemy an empty transient list/None and
#: cascade-delete the paper's related rows.
#:
#: NOTE: ``PaperMeta.has_report`` is deliberately absent — it is NOT a column.
#: It is derived at read time from the existence of a ``reports`` row (see
#: :func:`report_exists` / :func:`paper_ids_with_reports`), so it can never go
#: stale the way a persisted mirror would (ingest, rerun, delete, and any future
#: report-producing path would each have to remember to update it).
_PAPER_COLUMNS = (
    "arxiv_id",
    "arxiv_label",
    "version",
    "title",
    "authors",
    "abstract",
    "categories",
    "published",
    "updated",
    "pdf_url",
    "abs_url",
    "status",
    "ingested",
    "thumbnail_path",
    "page_count",
    "figure_count",
    "error_message",
    "updated_at",
)

__all__ = [
    "paper_row_to_meta",
    "meta_to_paper_row",
    "report_exists",
    "paper_ids_with_reports",
    "job_row_to_model",
    "get_paper_meta",
    "get_paper_row_exists",
    "upsert_paper_from_meta",
    "set_paper_status",
    "update_paper_after_text",
    "update_paper_figure_count",
    "persist_job",
    "get_job",
    "latest_job_for_paper",
    "latest_ingestion_job_for_paper",
    "list_papers",
    "delete_paper_rows",
    "delete_chapter_reports",
]


def _iso_date(value: Optional[str]):
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _report_paper_ids(
    session: Session, paper_ids: Optional[Iterable[str]] = None
) -> set[str]:
    """Paper ids that have a persisted whole-paper report, in ONE query.

    ``reports.paper_id`` is unique and indexed, so this is a single indexed scan
    for the whole library — the point being that ``list_papers()`` stays one
    papers query + one reports query, never one report lookup per paper.
    """
    ids = None if paper_ids is None else [pid for pid in paper_ids]
    if ids is not None and not ids:
        return set()
    stmt = select(ReportRow.paper_id)
    if ids is not None:
        stmt = stmt.where(ReportRow.paper_id.in_(ids))
    return {pid for pid in session.exec(stmt).all() if pid}


def paper_ids_with_reports(paper_ids: Optional[Iterable[str]] = None) -> set[str]:
    """Public wrapper over :func:`_report_paper_ids` opening its own session."""
    with session_scope() as session:
        return _report_paper_ids(session, paper_ids)


def report_exists(paper_id: str) -> bool:
    """True iff a whole-paper ``ReportRow`` is persisted for ``paper_id``."""
    with session_scope() as session:
        return bool(_report_paper_ids(session, [paper_id]))


def paper_row_to_meta(row: PaperRow, *, has_report: bool = False) -> PaperMeta:
    """Validate a persisted ``PaperRow`` back into a :class:`PaperMeta`.

    ``has_report`` is not a ``papers`` column — it is derived from the existence
    of a ``reports`` row and must be supplied by the caller, which is the only
    place that knows whether it already looked the reports table up (one grouped
    query for a list, one lookup for a single paper). It defaults to ``False``
    rather than doing a hidden per-row query, so a list caller cannot
    accidentally turn a grid render into an N+1.
    """
    return PaperMeta.model_validate(
        {
            "id": row.id,
            "arxiv_id": row.arxiv_id,
            "arxiv_label": row.arxiv_label,
            "version": row.version,
            "title": row.title,
            "authors": row.authors,
            "abstract": row.abstract,
            "categories": row.categories,
            "published": _iso_date(row.published),
            "updated": _iso_date(row.updated),
            "pdf_url": row.pdf_url,
            "abs_url": row.abs_url,
            "status": row.status,
            "ingested": row.ingested,
            "has_report": has_report,
            "thumbnail_path": row.thumbnail_path,
            "page_count": row.page_count,
            "figure_count": row.figure_count,
            "error_message": row.error_message,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
    )


def meta_to_paper_row(meta: PaperMeta) -> PaperRow:
    """Build a (detached) ``PaperRow`` from a :class:`PaperMeta`.

    ``meta.has_report`` is intentionally dropped: it is derived from the
    ``reports`` table, not stored on ``papers``.
    """
    return PaperRow(
        id=meta.id,
        arxiv_id=meta.arxiv_id,
        arxiv_label=meta.arxiv_label,
        version=meta.version,
        title=meta.title,
        authors=list(meta.authors),
        abstract=meta.abstract,
        categories=list(meta.categories),
        published=meta.published.isoformat() if meta.published else None,
        updated=meta.updated.isoformat() if meta.updated else None,
        pdf_url=meta.pdf_url,
        abs_url=meta.abs_url,
        status=meta.status.value
        if isinstance(meta.status, PaperStatus)
        else str(meta.status),
        ingested=meta.ingested,
        thumbnail_path=meta.thumbnail_path,
        page_count=meta.page_count,
        figure_count=meta.figure_count,
        error_message=meta.error_message,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
    )


def job_row_to_model(row: JobRow) -> IngestJob:
    """Validate a persisted ``JobRow`` back into an :class:`IngestJob`."""
    return IngestJob.model_validate(
        {
            "id": row.id,
            "paper_id": row.paper_id,
            "arxiv_id": row.arxiv_id,
            "state": row.state,
            "current_stage": row.current_stage,
            "percent": row.percent,
            "stages": row.stages,
            "log": row.log,
            "error_message": row.error_message,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
    )


def get_paper_meta(paper_id: str) -> Optional[PaperMeta]:
    with session_scope() as session:
        row = session.get(PaperRow, paper_id)
        if row is None:
            return None
        return paper_row_to_meta(
            row, has_report=bool(_report_paper_ids(session, [paper_id]))
        )


def get_paper_row_exists(paper_id: str) -> bool:
    with session_scope() as session:
        return session.get(PaperRow, paper_id) is not None


def upsert_paper_from_meta(meta: PaperMeta, *, reset_status: bool = False) -> None:
    """Insert or update the ``PaperRow`` for ``meta``.

    When ``reset_status`` is True (a fresh ingest/rerun), the row's status is
    forced back to ``queued`` and any prior error is cleared, regardless of
    what ``meta.status`` says. That same flag is what marks the paper's content
    as about to be rebuilt, so any persisted **per-chapter** reports are dropped
    here: re-ingestion can shift page ranges, which changes every derived
    ``Chapter.id`` and would leave the old chapter reports both unreachable and
    wrong. The whole-paper ``ReportRow`` is left alone (it is overwritten in
    place by the report generator once ingestion finishes).
    """
    if reset_status:
        removed = delete_chapter_reports(meta.id)
        if removed:
            log.info(
                "dropped chapter reports invalidated by re-ingest",
                extra={"paper_id": meta.id, "count": removed},
            )

    with session_scope() as session:
        row = session.get(PaperRow, meta.id)
        new_row = meta_to_paper_row(meta)
        if row is None:
            if reset_status:
                new_row.status = PaperStatus.QUEUED.value
                new_row.error_message = None
            session.add(new_row)
            return
        for field in _PAPER_COLUMNS:
            setattr(row, field, getattr(new_row, field))
        if reset_status:
            row.status = PaperStatus.QUEUED.value
            row.error_message = None
        row.updated_at = datetime.utcnow()
        session.add(row)


def set_paper_status(
    paper_id: str,
    status: PaperStatus,
    *,
    ingested: Optional[bool] = None,
    error_message: Optional[str] = None,
) -> None:
    with session_scope() as session:
        row = session.get(PaperRow, paper_id)
        if row is None:
            log.warning("set_paper_status: unknown paper", extra={"paper_id": paper_id})
            return
        row.status = status.value
        if ingested is not None:
            row.ingested = ingested
        if status == PaperStatus.ERROR:
            row.error_message = error_message
        elif status == PaperStatus.READY:
            row.error_message = None
        row.updated_at = datetime.utcnow()
        session.add(row)


def update_paper_after_text(
    paper_id: str, *, page_count: int, thumbnail_path: Optional[str]
) -> None:
    with session_scope() as session:
        row = session.get(PaperRow, paper_id)
        if row is None:
            return
        row.page_count = page_count
        if thumbnail_path:
            row.thumbnail_path = thumbnail_path
        row.updated_at = datetime.utcnow()
        session.add(row)


def update_paper_figure_count(paper_id: str, figure_count: int) -> None:
    with session_scope() as session:
        row = session.get(PaperRow, paper_id)
        if row is None:
            return
        row.figure_count = figure_count
        row.updated_at = datetime.utcnow()
        session.add(row)


def persist_job(job: IngestJob) -> None:
    """Insert or update the ``JobRow`` mirroring ``job`` (JSON columns)."""
    with session_scope() as session:
        row = session.get(JobRow, job.id)
        if row is None:
            row = JobRow(id=job.id, paper_id=job.paper_id)
        row.paper_id = job.paper_id
        row.arxiv_id = job.arxiv_id
        row.state = job.state.value
        row.current_stage = job.current_stage.value if job.current_stage else None
        row.percent = job.percent
        row.stages = [s.model_dump(mode="json") for s in job.stages]
        row.log = [entry.model_dump(mode="json") for entry in job.log]
        row.error_message = job.error_message
        row.created_at = job.created_at
        row.updated_at = datetime.utcnow()
        session.add(row)


def get_job(job_id: str) -> Optional[IngestJob]:
    with session_scope() as session:
        row = session.get(JobRow, job_id)
        return job_row_to_model(row) if row else None


def latest_job_for_paper(paper_id: str) -> Optional[IngestJob]:
    """The most recently created job row for ``paper_id`` — *any* kind.

    Chapter-report, flashcard and quiz jobs share the ``jobs`` table under the
    same ``paper_id``, so this can return a non-ingestion job. Callers that
    specifically mean "the paper's ingestion job" must use
    :func:`latest_ingestion_job_for_paper` instead.
    """
    with session_scope() as session:
        stmt = (
            select(JobRow)
            .where(JobRow.paper_id == paper_id)
            .order_by(JobRow.created_at.desc())
        )
        row = session.exec(stmt).first()
        return job_row_to_model(row) if row else None


def latest_ingestion_job_for_paper(paper_id: str) -> Optional[IngestJob]:
    """The most recently created *ingestion* job row for ``paper_id``.

    An ingestion job is identified by carrying pipeline stages: chapter-report,
    flashcard and quiz jobs reuse this table with ``stages: []`` (the
    ``JobStage`` names would be lies for them). Picking "the latest job row" and
    then rejecting it when it is stage-less is NOT equivalent — a study or
    chapter job queued while ingestion is still running becomes the latest row
    and hides the live ingestion job entirely, which is exactly what
    ``GET /report``'s race guard must not miss.
    """
    with session_scope() as session:
        stmt = (
            select(JobRow)
            .where(JobRow.paper_id == paper_id)
            .order_by(JobRow.created_at.desc())
        )
        for row in session.exec(stmt).all():
            if row.stages:
                return job_row_to_model(row)
        return None


def list_papers() -> list[PaperMeta]:
    """The whole library, newest-updated first.

    Two queries total: the papers, then one grouped lookup of every paper id
    that has a report row. ``has_report`` is derived from that set — no per-row
    report query (N+1), and no persisted mirror column to keep in sync.
    """
    with session_scope() as session:
        rows = session.exec(
            select(PaperRow).order_by(PaperRow.updated_at.desc())
        ).all()
        with_report = _report_paper_ids(session)
        return [paper_row_to_meta(r, has_report=r.id in with_report) for r in rows]


def delete_chapter_reports(paper_id: str) -> int:
    """Delete every persisted per-chapter report for ``paper_id``.

    ``ChapterReportRow`` deliberately declares no ORM ``Relationship`` back to
    ``PaperRow`` (it would disturb PaperRow's existing mapper configuration), so
    nothing cascades — these rows must be deleted explicitly on paper delete and
    dropped on re-ingest, since re-ingestion can shift page ranges and therefore
    invalidate every ``Chapter.id`` a stored report was keyed by.

    Returns the number of rows removed.
    """
    with session_scope() as session:
        rows = session.exec(
            select(ChapterReportRow).where(ChapterReportRow.paper_id == paper_id)
        ).all()
        for row in rows:
            session.delete(row)
        return len(rows)


def delete_paper_rows(paper_id: str) -> bool:
    """Delete every SQLite row owned by the paper.

    That is: the paper itself, its jobs, report, chapter reports, chat — and the
    whole Study layer (flashcards + their append-only review log, quizzes +
    questions + attempts + answers). ``flashcards.paper_id``,
    ``quizzes.paper_id`` etc. are declared foreign keys to ``papers.id``, so
    leaving them behind would orphan rows that point at a paper that no longer
    exists; the Study screen's cross-paper due queue would then surface cards for
    a deleted paper.

    Deletion order respects the intra-Study foreign keys: answers before
    attempts/questions, questions/attempts before quizzes, reviews before cards.

    Returns ``True`` if a paper row existed and was removed.
    """
    with session_scope() as session:
        row = session.get(PaperRow, paper_id)
        if row is None:
            return False

        # --- Study layer: quizzes -------------------------------------------
        attempt_rows = session.exec(
            select(QuizAttemptRow).where(QuizAttemptRow.paper_id == paper_id)
        ).all()
        attempt_ids = [a.id for a in attempt_rows]
        if attempt_ids:
            for answer_row in session.exec(
                select(QuizAnswerRow).where(QuizAnswerRow.attempt_id.in_(attempt_ids))
            ).all():
                session.delete(answer_row)
        for attempt_row in attempt_rows:
            session.delete(attempt_row)
        for question_row in session.exec(
            select(QuizQuestionRow).where(QuizQuestionRow.paper_id == paper_id)
        ).all():
            session.delete(question_row)
        for quiz_row in session.exec(
            select(QuizRow).where(QuizRow.paper_id == paper_id)
        ).all():
            session.delete(quiz_row)

        # --- Study layer: flashcards (reviews first, they FK the card) -------
        for review_row in session.exec(
            select(FlashcardReviewRow).where(FlashcardReviewRow.paper_id == paper_id)
        ).all():
            session.delete(review_row)
        for card_row in session.exec(
            select(FlashcardRow).where(FlashcardRow.paper_id == paper_id)
        ).all():
            session.delete(card_row)

        for job_row in session.exec(
            select(JobRow).where(JobRow.paper_id == paper_id)
        ).all():
            session.delete(job_row)
        for chapter_report_row in session.exec(
            select(ChapterReportRow).where(ChapterReportRow.paper_id == paper_id)
        ).all():
            session.delete(chapter_report_row)
        for chat_row in session.exec(
            select(ChatRow).where(ChatRow.paper_id == paper_id)
        ).all():
            session.delete(chat_row)
        report_row = session.exec(
            select(ReportRow).where(ReportRow.paper_id == paper_id)
        ).first()
        if report_row is not None:
            session.delete(report_row)
        session.delete(row)
        return True
