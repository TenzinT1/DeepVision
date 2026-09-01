"""Report router — whole-paper and per-chapter reports.

Owned by: REPORT domain.

- ``GET /report/{paper_id}`` returns the full interactive :class:`Report`
  (sections, stats, citations, media) for a paper, building it eagerly the
  first time it's requested if it hasn't been persisted yet — *unless* the
  paper's ingestion job is still on its way to writing that row, in which case
  it answers 409 "still generating" rather than racing the ingest thread for
  the unique ``reports.paper_id`` row.
- ``POST /report/{paper_id}/chapter`` queues a report scoped to ONE chapter's
  page range and returns a pollable job (202), or the already-persisted chapter
  report inline (200, ``cached: true``).
- ``GET /report/{paper_id}/chapter/{chapter_id}`` returns that chapter report
  (404 until generation finishes).

A chapter report has the same eleven sections in the same order as any other
report — only the retrieval scope, the ``scope``/``chapter_*`` fields and
``stats.pages`` differ — so the frontend renders it with the existing section
components and just swaps the header line.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Response

from deepvision.api.chapter_job_runner import launch_chapter_report
from deepvision.api.deps import get_settings
from deepvision.api.schemas import ChapterReportRequest, ChapterReportResponse
from deepvision.models import Report
from deepvision.models.job import JobStage, JobState, StageStatus
from deepvision.report.chapter_report_generator import (
    ChapterNotFoundError,
    ChapterReportGenerator,
    resolve_chapter,
)
from deepvision.report.report_generator import (
    DefaultReportGenerator,
    PaperNotFoundError,
    ReportDegradedError,
)
from deepvision.utils import get_logger

router = APIRouter(tags=["report"])

log = get_logger(__name__)

_generator = DefaultReportGenerator()
_chapter_generator = ChapterReportGenerator()


#: HTTP status for "the report is being generated right now, come back".
#: 409 (not 404) because the paper is fine and the resource is on its way — a
#: 404 here reads as "this paper has no report and never will".
REPORT_GENERATING_STATUS = 409


def _report_stage_in_flight(paper_id: str) -> Optional[str]:
    """Return a human message iff an ingestion job is *currently* on its way to
    writing this paper's report row, else ``None``.

    ``GET /report`` generates eagerly when no report row exists, and that
    generation is synchronous. If the ingest thread is still running, that
    eager build is a *second* agent ensemble racing the first for the unique
    ``reports.paper_id`` row — two 20-minute local-model runs, one of which
    loses on insert. Detecting the live job is what stops it.

    Only ingestion jobs count: chapter-report and study jobs share the ``jobs``
    table under the same paper_id but carry ``stages: []`` and never touch
    ``reports``. Crucially, we ask for the latest *ingestion* job rather than
    the latest job of any kind — one of those stage-less rows created while
    ingestion is still running (generate flashcards / a quiz / a chapter report
    on a paper that went ``ready`` after the embedding stage) becomes "the
    latest job" and would hide the live ingestion job completely, re-opening the
    exact race this guard exists to prevent.
    """
    from deepvision.ingestion import repo  # local import: keeps router import cheap

    job = repo.latest_ingestion_job_for_paper(paper_id)
    if job is None or not job.stages:
        return None
    if job.state not in (JobState.QUEUED, JobState.RUNNING):
        # A finished/failed job cannot still be writing a report row, so the
        # eager build below is safe — and is the only way an older paper whose
        # job predates the tracked report stage ever gets a report at all.
        return None

    report_stage = next(
        (sp for sp in job.stages if sp.stage == JobStage.REPORT), None
    )
    if report_stage is not None and report_stage.status in (
        StageStatus.DONE,
        StageStatus.ERROR,
    ):
        # The running job is already past the report stage (it failed
        # non-fatally, or finished it and is winding down); nothing left to race.
        return None

    where = (
        "generating the report now"
        if report_stage is not None and report_stage.status == StageStatus.RUNNING
        else "still ingesting this paper"
    )
    return (
        f"Report not ready yet — {where}. On a local model this can take around "
        "20 minutes. Chat and Compare already work for this paper; reopen this "
        "page when the ingestion job finishes."
    )


@router.get("/report/{paper_id}", response_model=Report)
def get_report(paper_id: str) -> Report:
    """Return the full report for ``paper_id``.

    404 only when the paper itself is unknown. When the paper exists but its
    report has not been written yet *and* its ingestion job is still running,
    this returns :data:`REPORT_GENERATING_STATUS` ("still generating") instead
    of kicking off a competing synchronous agent run.
    """
    try:
        return _generator.load(paper_id)
    except LookupError:
        pass  # no persisted report yet -- build it eagerly below

    from deepvision.ingestion import repo  # local import: keeps router import cheap

    if not repo.get_paper_row_exists(paper_id):
        raise HTTPException(status_code=404, detail=f"paper not found: {paper_id}")

    in_flight = _report_stage_in_flight(paper_id)
    if in_flight is not None:
        log.info(
            "report requested while ingestion is still running",
            extra={"paper_id": paper_id},
        )
        raise HTTPException(status_code=REPORT_GENERATING_STATUS, detail=in_flight)

    settings = get_settings()
    try:
        return _generator.generate(paper_id, settings)
    except PaperNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ReportDegradedError as exc:
        # The agent ensemble failed but a placeholder report WAS persisted (its
        # Overview explains why, and points at the Library's re-run). Rendering
        # that beats a 500.
        #
        # This route is NOT a retry path: ``load()`` above short-circuits as soon
        # as *any* report row exists, and the ingest job's report stage persists a
        # placeholder even when it fails — so a later GET returns that placeholder
        # rather than re-running the agents. Retrying means re-running ingestion
        # (POST /papers/{id}/rerun — the Library's re-run button), which is what
        # the failed stage's log line and the ingestion modal tell the user.
        log.warning(
            "serving degraded placeholder report",
            extra={"paper_id": paper_id, "error": str(exc)},
        )
        return exc.report
    except Exception as exc:  # pragma: no cover - defensive
        log.error(
            "report generation failed", extra={"paper_id": paper_id, "error": str(exc)}
        )
        raise HTTPException(
            status_code=500, detail=f"failed to build report for {paper_id}: {exc}"
        ) from exc


@router.post(
    "/report/{paper_id}/chapter",
    response_model=ChapterReportResponse,
    status_code=202,
)
def generate_chapter_report(
    paper_id: str, request: ChapterReportRequest, response: Response
) -> ChapterReportResponse:
    """Queue a chapter-scoped report (202), or return the cached one (200).

    Generation runs the full agent ensemble, so it is never synchronous. The
    returned job is the ordinary :class:`IngestJob` envelope and is polled
    through the existing ``GET /jobs/{job_id}``.
    """
    from deepvision.ingestion import repo  # local import: keeps router import cheap

    paper = repo.get_paper_meta(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"paper not found: {paper_id}")

    # Cache check first: a persisted chapter report renders from its own
    # denormalized columns, so this path works even if the PDF (and therefore
    # chapter derivation) is unavailable right now.
    if not request.force:
        try:
            cached = _chapter_generator.load(paper_id, request.chapter_id)
        except LookupError:
            cached = None
        if cached is not None:
            response.status_code = 200
            return ChapterReportResponse(
                paper_id=paper_id,
                chapter_id=request.chapter_id,
                cached=True,
                job=None,
                report=cached,
            )

    try:
        chapter = resolve_chapter(paper_id, request.chapter_id)
    except ChapterNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    settings = request.settings_override or get_settings()
    try:
        job = launch_chapter_report(
            paper_id, chapter, settings, arxiv_id=paper.arxiv_id
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.error(
            "failed to queue chapter report",
            extra={"paper_id": paper_id, "chapter_id": chapter.id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=500, detail=f"failed to queue chapter report: {exc}"
        ) from exc

    response.status_code = 202
    return ChapterReportResponse(
        paper_id=paper_id,
        chapter_id=chapter.id,
        cached=False,
        job=job,
        report=None,
    )


@router.get("/report/{paper_id}/chapter/{chapter_id}", response_model=Report)
def get_chapter_report(paper_id: str, chapter_id: str) -> Report:
    """Return the persisted chapter report (404 until generation finishes)."""
    try:
        return _chapter_generator.load(paper_id, chapter_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no chapter report for paper {paper_id!r} chapter {chapter_id!r} "
                "(generate it first)"
            ),
        ) from exc
