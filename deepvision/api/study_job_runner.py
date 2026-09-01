"""Background runner for Study-layer generation (decks and quizzes).

Deck and quiz generation both run LLM calls — seconds on an API model, minutes
on a local one — so neither may block the HTTP request. Like
``api/chapter_job_runner.py``, this reuses the **existing** async machinery
wholesale rather than inventing a third one: the same
:class:`~deepvision.models.IngestJob`, the same ``jobs`` table (via
``api.job_runner.save_job``), the same ``GET /jobs/{job_id}``, and therefore the
frontend's existing ``pollJob``.

Same two honest departures from an ingestion job:

- ``stages`` is ``[]`` and ``current_stage`` is ``None`` — the seven ``JobStage``
  values describe PDF ingestion and would be lies here. Progress is ``percent``
  plus ``log``. An empty list also means growing ``STAGE_ORDER`` can never make
  these jobs render a stage that will never complete.
- Milestones bracket the one opaque generation call rather than faking
  granularity.

**Deliberately generic.** This module is shared: the flashcards router launches
``kind="flashcards"`` and the quiz router launches ``kind="quiz"``. It knows
nothing about cards or questions — it takes a ``work`` callable that reports
progress and returns a closing message. Keeping it that way is what stops a
fourth copy of this file appearing the next time something in Study needs a job.

Concurrency: a second POST for the same ``(kind, paper_id)`` while one is in
flight returns the **same** job instead of paying for a second generation. The
check-persist-reserve is one atomic critical section (see
:func:`launch_study_job`) — split in two it is a real race, because FastAPI runs
sync route handlers on a threadpool, so two tabs or one impatient double-click
really can arrive together.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Callable, NamedTuple, Optional

from deepvision.api.job_runner import save_job
from deepvision.models import IngestJob, JobState, LogEntry
from deepvision.utils import get_logger
from deepvision.utils.ids import job_id as new_job_id

__all__ = [
    "ProgressFn",
    "StudyWork",
    "StudyJobHandle",
    "JobAbort",
    "inflight_study_job",
    "launch_study_job",
]

log = get_logger(__name__)

#: ``(percent, message)`` — what a worker calls to report progress.
ProgressFn = Callable[[int, str], None]

#: The unit of work: given a progress reporter, do the generation and return the
#: line to close the job's log with (an empty string falls back to a default).
StudyWork = Callable[[ProgressFn], str]


class JobAbort(RuntimeError):
    """An expected, user-fixable reason a job cannot proceed.

    Distinguished from a genuine bug so :func:`_run` can log it as a warning
    with no stack trace: "you have no report yet" is a normal state of the app,
    not an incident. Either way the job ends in ``error`` carrying the message,
    which is what the user sees — the HTTP request itself already returned 202.
    """


class StudyJobHandle(NamedTuple):
    """What :func:`launch_study_job` returns.

    ``payload`` is whatever the caller passed in — and, crucially, on a lost
    race it is the **winner's** payload, not the loser's. That is what lets the
    quiz router mint a ``quiz_id`` up front and still have a second concurrent
    POST resolve to the quiz that is actually being generated rather than to an
    id nothing will ever write.
    """

    job: IngestJob
    payload: Any
    reused: bool


#: (kind, paper_id) -> (live job, caller payload) for the generation this
#: process is running for that pair. Guarded by ``_LOCK``. The job **object** is
#: held, not its id, so the reservation is valid the instant it is made —
#: looking the id up in the ``jobs`` table instead would miss a job that has
#: been reserved but not yet persisted and hand a second caller a duplicate
#: generation.
_INFLIGHT: dict[tuple[str, str], tuple[IngestJob, Any]] = {}
_LOCK = threading.Lock()


def _now_hhmmss() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log_progress(job: IngestJob, percent: int, message: str) -> None:
    """Set ``percent``, append a log line, and persist the job snapshot."""
    job.percent = max(0, min(100, int(percent)))
    job.log.append(LogEntry(t=_now_hhmmss(), m=message))
    job.updated_at = datetime.utcnow()
    save_job(job)


def _inflight_locked(key: tuple[str, str]) -> Optional[tuple[IngestJob, Any]]:
    """``inflight_study_job`` without taking the lock. **Caller holds ``_LOCK``.**

    Split out so :func:`launch_study_job` can check-and-reserve as one atomic
    step. Two separate lock acquisitions leave a window in which two concurrent
    POSTs both see "nothing in flight" and both start a full generation —
    duplicated LLM cost, and two writers racing the same rows.

    The worker mutates the very object stored here, so its ``state`` is live.
    """
    entry = _INFLIGHT.get(key)
    if entry is None:
        return None
    if entry[0].state in (JobState.QUEUED, JobState.RUNNING):
        return entry
    _INFLIGHT.pop(key, None)
    return None


def inflight_study_job(kind: str, paper_id: str) -> Optional[IngestJob]:
    """Return the still-running job for this ``(kind, paper_id)``, if any."""
    with _LOCK:
        entry = _inflight_locked((kind, paper_id))
        return entry[0] if entry else None


def _forget(key: tuple[str, str]) -> None:
    with _LOCK:
        _INFLIGHT.pop(key, None)


def launch_study_job(
    kind: str,
    paper_id: str,
    work: StudyWork,
    *,
    arxiv_id: str = "",
    start_message: str = "",
    done_message: str = "",
    payload: Any = None,
) -> StudyJobHandle:
    """Queue a Study generation job and return it **already persisted**.

    The job is saved before this returns, so ``GET /jobs/{id}`` resolves
    immediately — before the worker thread has done anything. Check, persist and
    reserve happen under a single hold of ``_LOCK``, so two simultaneous POSTs
    for the same ``(kind, paper_id)`` can never both win: the loser gets the
    winner's job (and the winner's ``payload``) back, and both tabs poll the
    same id. ``reused`` on the returned handle says which happened.
    """
    key = (kind, paper_id)
    job = IngestJob.new(id=new_job_id(), paper_id=paper_id, arxiv_id=arxiv_id)
    # Not an ingestion: no PDF stages, no current stage. Progress is percent+log.
    job.stages = []
    job.current_stage = None

    with _LOCK:
        existing = _inflight_locked(key)
        if existing is not None:
            log.info(
                "study job already running; returning the in-flight job",
                extra={"kind": kind, "paper_id": paper_id, "job_id": existing[0].id},
            )
            return StudyJobHandle(job=existing[0], payload=existing[1], reused=True)
        # Persist before reserving: a failed save must leave no reservation
        # behind, or every later attempt for this paper would be blocked by a
        # job that does not exist.
        save_job(job)
        _INFLIGHT[key] = (job, payload)

    thread = threading.Thread(
        target=_run,
        args=(job, key, work, start_message, done_message),
        name=f"study-{kind}-{paper_id}",
        daemon=True,
    )
    thread.start()
    return StudyJobHandle(job=job, payload=payload, reused=False)


def _run(
    job: IngestJob,
    key: tuple[str, str],
    work: StudyWork,
    start_message: str,
    done_message: str,
) -> None:
    """Worker body: bracket the one opaque generation with honest milestones."""
    kind, paper_id = key
    try:
        job.state = JobState.RUNNING
        _log_progress(job, 5, start_message or f"Preparing {kind} generation")

        closing = work(lambda pct, msg: _log_progress(job, pct, msg))

        job.state = JobState.DONE
        job.current_stage = None
        _log_progress(job, 100, closing or done_message or f"{kind} ready")
    except JobAbort as exc:
        # Expected and user-fixable ("generate the report first") — a warning,
        # not a stack trace. The job still ends in `error` with the message.
        log.warning(
            "study job stopped: %s", exc, extra={"kind": kind, "paper_id": paper_id}
        )
        _fail(job, kind, exc)
    except Exception as exc:  # noqa: BLE001 - worker guard; must never escape
        log.exception(
            "study job failed", extra={"kind": kind, "paper_id": paper_id}
        )
        _fail(job, kind, exc)
    finally:
        _forget(key)


def _fail(job: IngestJob, kind: str, exc: BaseException) -> None:
    """Mark ``job`` failed with ``exc``'s message and persist the snapshot."""
    job.state = JobState.ERROR
    job.error_message = str(exc)
    job.log.append(LogEntry(t=_now_hhmmss(), m=f"{kind} generation failed: {exc}"))
    job.updated_at = datetime.utcnow()
    save_job(job)
