"""Background ingestion job runner (the api layer — Core / Settings / API glue).

This module is the api layer's half of the async ingestion contract: a small,
ingestion-agnostic helper that

1. persists :class:`~deepvision.models.IngestJob` snapshots to the SQLite
   ``jobs`` table so ``GET /jobs/{id}`` always reflects live state, and
2. drains a ``StageProgress`` iterator (i.e.
   :meth:`~deepvision.ingestion.orchestrator.IngestionOrchestrator.run`) as a
   FastAPI ``BackgroundTasks`` task, updating + persisting the job after every
   yielded stage transition.

It deliberately knows nothing about PDFs, OCR, or embeddings — only how to turn
"an iterator of StageProgress" into a live, polled ``IngestJob`` row, in a
single process with in-memory-cheap SQLite writes. The ingestion orchestrator
(the ingestion layer) is expected to call :func:`launch` from its
``IngestionOrchestrator.start_async`` implementation, e.g.::

    class MyOrchestrator(IngestionOrchestrator):
        def start_async(self, paper_id, settings):
            return job_runner.launch(
                background_tasks,
                self.run,
                paper_id=paper_id,
                arxiv_id=arxiv_id,
                settings=settings,
            )

**Do not use ``on_complete`` to generate the report.** Report generation is the
last *tracked* stage of ``run()`` (``JobStage.REPORT``) precisely so the job
cannot reach ``done`` before the report exists; hanging it off a completion hook
is the old bug — the modal offers "Open report →" while generation has not
started. ``on_complete``/``on_error`` are for genuine post-job side effects
(notifications, cache warming).

Note also that :func:`launch` is currently **unused** — the shipped
``DefaultIngestionOrchestrator.start_async`` runs its own thread and persists
the job itself, because ``run()`` distinguishes fatal stages (which set
``state=error`` and stop) from non-fatal ones (images/OCR/vision/report, which
are marked ``error`` while the job still finishes ``done``). ``_apply_progress``
below cannot make that distinction from a ``StageProgress`` alone and treats
*any* errored stage as a failed job, so wiring this helper to ingestion would
report a job as failed when only the report degraded.

``BackgroundTasks`` instance, so in practice the ``ingest``/``rerun`` route
bodies (the ingestion layer) will call :func:`launch` directly rather than going through
``start_async``; either path is supported since both just need a
``(paper_id, settings) -> Iterator[StageProgress]`` callable.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Callable, Iterator, Optional

from fastapi import BackgroundTasks
from sqlmodel import Session, select

from deepvision.db import get_engine
from deepvision.db.schema import JobRow
from deepvision.models import (
    STAGE_ORDER,
    AppSettings,
    IngestJob,
    JobState,
    LogEntry,
    StageProgress,
    StageStatus,
)
from deepvision.utils import get_logger
from deepvision.utils.ids import job_id as new_job_id

__all__ = ["save_job", "load_job", "list_jobs_for_paper", "launch"]

log = get_logger(__name__)

# Serializes job-row writes across background threads. FastAPI BackgroundTasks
# run (post-response) on the threadpool, and multiple ingest jobs can be
# in-flight at once in this single-process app; a simple lock keeps SQLite
# writes from interleaving without reaching for a real queue.
_WRITE_LOCK = threading.Lock()


def _row_fields(job: IngestJob) -> dict:
    return {
        "paper_id": job.paper_id,
        "arxiv_id": job.arxiv_id,
        "state": job.state.value,
        "current_stage": job.current_stage.value if job.current_stage else None,
        "percent": job.percent,
        "stages": [s.model_dump(mode="json") for s in job.stages],
        "log": [entry.model_dump(mode="json") for entry in job.log],
        "error_message": job.error_message,
    }


def save_job(job: IngestJob, session: Optional[Session] = None) -> None:
    """Upsert ``job`` into the ``jobs`` table (create or update the row)."""
    owns_session = session is None
    sess = session or Session(get_engine())
    try:
        with _WRITE_LOCK:
            row = sess.get(JobRow, job.id)
            fields = _row_fields(job)
            if row is None:
                row = JobRow(id=job.id, created_at=job.created_at, **fields)
            else:
                for key, value in fields.items():
                    setattr(row, key, value)
            row.updated_at = datetime.utcnow()
            sess.add(row)
            sess.commit()
    finally:
        if owns_session:
            sess.close()


def _job_from_row(row: JobRow) -> IngestJob:
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


def load_job(job_id: str, session: Optional[Session] = None) -> Optional[IngestJob]:
    """Load + validate an :class:`IngestJob` snapshot by id, or ``None``."""
    owns_session = session is None
    sess = session or Session(get_engine())
    try:
        row = sess.get(JobRow, job_id)
        return _job_from_row(row) if row is not None else None
    finally:
        if owns_session:
            sess.close()


def list_jobs_for_paper(paper_id: str, session: Optional[Session] = None) -> list[IngestJob]:
    """List all jobs for ``paper_id``, most recent first."""
    owns_session = session is None
    sess = session or Session(get_engine())
    try:
        rows = sess.exec(
            select(JobRow)
            .where(JobRow.paper_id == paper_id)
            .order_by(JobRow.created_at.desc())
        ).all()
        return [_job_from_row(row) for row in rows]
    finally:
        if owns_session:
            sess.close()


def _now_hhmmss() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _apply_progress(job: IngestJob, progress: StageProgress) -> None:
    """Mutate ``job`` in place from one yielded :class:`StageProgress`."""
    for i, existing in enumerate(job.stages):
        if existing.stage == progress.stage:
            job.stages[i] = progress
            break
    else:
        job.stages.append(progress)

    if progress.status == StageStatus.RUNNING:
        job.current_stage = progress.stage

    done_count = sum(1 for s in job.stages if s.status == StageStatus.DONE)
    job.percent = min(100, round(done_count / max(1, len(STAGE_ORDER)) * 100))

    detail = progress.detail or progress.status.value
    job.log.append(
        LogEntry(t=_now_hhmmss(), m=f"{progress.stage.value}: {detail}", stage=progress.stage)
    )
    job.updated_at = datetime.utcnow()

    if progress.status == StageStatus.ERROR:
        job.state = JobState.ERROR
        job.error_message = job.error_message or detail


def launch(
    background_tasks: BackgroundTasks,
    run_fn: Callable[[str, AppSettings], Iterator[StageProgress]],
    *,
    paper_id: str,
    arxiv_id: str,
    settings: AppSettings,
    on_complete: Optional[Callable[[IngestJob], None]] = None,
    on_error: Optional[Callable[[IngestJob, BaseException], None]] = None,
) -> IngestJob:
    """Create a queued job, schedule ``run_fn`` to drain in the background, return it.

    ``run_fn`` is normally ``orchestrator.run`` — a
    ``(paper_id, settings) -> Iterator[StageProgress]`` callable. This function
    does not import or depend on the ingestion package, keeping the ownership
    boundary clean: the ingestion layer supplies the generator, the api layer persists it.

    The returned :class:`IngestJob` is already persisted (``state=queued``) so
    ``GET /jobs/{id}`` resolves immediately, even before the background task
    gets a worker thread.

    ``on_complete``/``on_error`` let the caller hook post-pipeline behavior
    without this module needing to know about reports or agents. They are *not*
    where report generation belongs — that is a tracked stage of ``run_fn``
    itself (see the module docstring).
    """
    job = IngestJob.new(id=new_job_id(), paper_id=paper_id, arxiv_id=arxiv_id)
    save_job(job)

    def _drain() -> None:
        current = job
        current.state = JobState.RUNNING
        current.updated_at = datetime.utcnow()
        save_job(current)
        try:
            for progress in run_fn(paper_id, settings):
                _apply_progress(current, progress)
                save_job(current)
                if current.state == JobState.ERROR:
                    log.error(
                        "ingestion stage reported error",
                        extra={"job_id": current.id, "paper_id": paper_id},
                    )
                    if on_error:
                        _safe_call(on_error, current, RuntimeError(current.error_message or "stage error"))
                    return

            current.state = JobState.DONE
            current.percent = 100
            current.current_stage = None
            current.updated_at = datetime.utcnow()
            save_job(current)
            if on_complete:
                _safe_call(on_complete, current)
        except Exception as exc:  # pragma: no cover - defensive worker guard
            log.exception(
                "ingestion job crashed", extra={"job_id": current.id, "paper_id": paper_id}
            )
            current.state = JobState.ERROR
            current.error_message = str(exc)
            current.log.append(
                LogEntry(t=_now_hhmmss(), m=f"job failed: {exc}", stage=current.current_stage)
            )
            current.updated_at = datetime.utcnow()
            save_job(current)
            if on_error:
                _safe_call(on_error, current, exc)

    background_tasks.add_task(_drain)
    return job


def _safe_call(fn: Callable, *args) -> None:
    """Invoke a caller-supplied hook without letting it break the worker."""
    try:
        fn(*args)
    except Exception:  # pragma: no cover - defensive
        log.exception("job_runner callback raised")
