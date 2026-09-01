"""Background runner for per-chapter report generation.

Chapter generation runs the full agent ensemble — real LLM calls, minutes on a
local model — so it must never block the HTTP request. It reuses the **existing**
async machinery wholesale rather than inventing a second one: the same
:class:`~deepvision.models.IngestJob` model, the same ``jobs`` table (via
``api.job_runner.save_job``), the same ``GET /jobs/{job_id}``, and therefore the
frontend's existing ``pollJob``.

Two honest departures from an ingestion job:

- ``stages`` is ``[]`` and ``current_stage`` is ``None``. The seven ``JobStage``
  values describe PDF ingestion (Download, OCR, …, Generate report) and would be
  lies here — leaving the list empty is also what keeps a stage added to
  ``STAGE_ORDER`` from rendering as a phantom row that never completes.
  Progress is ``percent`` + ``log``, which is exactly what the Report page's
  inline progress row renders (the IngestionModal is not involved).
- The milestones bracket the one opaque ``build_report`` call rather than faking
  granularity: 5 "Scoping to pages X-Y", 15 "Generating sections …", 90 "Saving
  chapter report", 100 done.

Concurrency: two POSTs for the same ``(paper_id, chapter_id)`` while one is in
flight return the **same** job instead of starting a second generation. The
check-and-reserve is one atomic critical section (see
:func:`launch_chapter_report`) — split in two it is a real race, since FastAPI
runs sync route handlers on a threadpool.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Optional

from deepvision.api.job_runner import save_job
from deepvision.models import AppSettings, Chapter, IngestJob, JobState, LogEntry
from deepvision.report.chapter_report_generator import ChapterReportGenerator
from deepvision.utils import get_logger
from deepvision.utils.ids import job_id as new_job_id

__all__ = ["launch_chapter_report", "inflight_job"]

log = get_logger(__name__)

_generator = ChapterReportGenerator()

#: (paper_id, chapter_id) -> the live :class:`IngestJob` this process is running
#: for that pair. Guarded by ``_LOCK``. The job **object** is held, not its id,
#: so the reservation is valid the instant it is made — looking the id up in the
#: ``jobs`` table instead would miss a job that has been reserved but not yet
#: persisted, and hand a second caller a duplicate generation.
_INFLIGHT: dict[tuple[str, str], IngestJob] = {}
_LOCK = threading.Lock()


def _now_hhmmss() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log_progress(job: IngestJob, percent: int, message: str) -> None:
    """Set ``percent``, append a log line, and persist the job snapshot."""
    job.percent = max(0, min(100, int(percent)))
    job.log.append(LogEntry(t=_now_hhmmss(), m=message))
    job.updated_at = datetime.utcnow()
    save_job(job)


def _inflight_locked(paper_id: str, chapter_id: str) -> Optional[IngestJob]:
    """``inflight_job`` without taking the lock. **Caller must hold ``_LOCK``.**

    Split out so :func:`launch_chapter_report` can do check-and-reserve as one
    atomic step: doing them as two separate lock acquisitions leaves a window in
    which two concurrent POSTs for the same ``(paper_id, chapter_id)`` both see
    "nothing in flight" and both start a full agent run — duplicated LLM cost,
    and a race to ``INSERT`` the same ``(paper_id, chapter_id)`` row that the
    ``chapter_reports`` unique constraint would reject, failing one of the jobs.

    The worker mutates the very object stored here, so its ``state`` is live.
    """
    job = _INFLIGHT.get((paper_id, chapter_id))
    if job is None:
        return None
    if job.state in (JobState.QUEUED, JobState.RUNNING):
        return job
    _INFLIGHT.pop((paper_id, chapter_id), None)
    return None


def inflight_job(paper_id: str, chapter_id: str) -> Optional[IngestJob]:
    """Return the still-running job for this pair, if this process has one."""
    with _LOCK:
        return _inflight_locked(paper_id, chapter_id)


def _forget(paper_id: str, chapter_id: str) -> None:
    with _LOCK:
        _INFLIGHT.pop((paper_id, chapter_id), None)


def launch_chapter_report(
    paper_id: str,
    chapter: Chapter,
    settings: AppSettings,
    *,
    arxiv_id: str,
) -> IngestJob:
    """Queue chapter-report generation and return the (already persisted) job.

    The returned job is saved before this function returns, so ``GET
    /jobs/{id}`` resolves immediately — before the worker thread has done
    anything.

    Check, persist and reserve all happen under a **single** hold of ``_LOCK``,
    so two simultaneous POSTs for the same ``(paper_id, chapter_id)`` can never
    both win: the loser gets the winner's — already persisted — job back.
    FastAPI runs sync route handlers on a threadpool, so this really does happen
    (two tabs, a double-click that outruns the disabled state, a retry), and
    without it the same chapter is generated twice at full LLM cost, with both
    writers racing the single ``(paper_id, chapter_id)`` row and one of them
    losing to the unique constraint. The critical section is one small SQLite
    insert on a user-initiated POST — not a hot path.
    """
    job = IngestJob.new(id=new_job_id(), paper_id=paper_id, arxiv_id=arxiv_id)
    # Not an ingestion: no PDF stages, no current stage. Progress is percent+log.
    job.stages = []
    job.current_stage = None

    with _LOCK:
        existing = _inflight_locked(paper_id, chapter.id)
        if existing is not None:
            log.info(
                "chapter report already generating; returning the in-flight job",
                extra={
                    "paper_id": paper_id,
                    "chapter_id": chapter.id,
                    "job_id": existing.id,
                },
            )
            return existing
        # Persist before reserving: a failed save must leave no reservation
        # behind, or every later attempt for this chapter would be blocked by a
        # job that does not exist.
        save_job(job)
        _INFLIGHT[(paper_id, chapter.id)] = job

    thread = threading.Thread(
        target=_run,
        args=(job, paper_id, chapter, settings),
        name=f"chapter-report-{paper_id}-{chapter.id}",
        daemon=True,
    )
    thread.start()
    return job


def _run(
    job: IngestJob, paper_id: str, chapter: Chapter, settings: AppSettings
) -> None:
    """Worker body: bracket the one opaque build with honest milestones."""
    try:
        job.state = JobState.RUNNING
        _log_progress(
            job, 5, f"Scoping to pages {chapter.page_start}-{chapter.page_end}"
        )
        _log_progress(
            job,
            15,
            "Generating sections (this can take several minutes on a local model)",
        )

        _generator.generate(
            paper_id,
            chapter,
            settings,
            on_progress=lambda pct, msg: _log_progress(job, pct, msg),
        )

        job.state = JobState.DONE
        job.current_stage = None
        _log_progress(job, 100, f"Chapter report ready: {chapter.title}")
    except Exception as exc:  # noqa: BLE001 - worker guard; must never escape
        log.exception(
            "chapter report generation failed",
            extra={"paper_id": paper_id, "chapter_id": chapter.id},
        )
        job.state = JobState.ERROR
        job.error_message = str(exc)
        job.log.append(
            LogEntry(t=_now_hhmmss(), m=f"chapter report failed: {exc}")
        )
        job.updated_at = datetime.utcnow()
        save_job(job)
    finally:
        _forget(paper_id, chapter.id)
