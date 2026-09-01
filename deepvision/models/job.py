"""Async ingestion job model.

Ingestion runs asynchronously through seven named stages. ``IngestJob`` is the
object polled by ``GET /jobs/{id}`` and rendered by the ingestion modal: it
exposes overall ``state``, current ``stage``, ``percent``, per-stage status, and
a live ``log``. The seven stages and their order are fixed by the UI.

The final stage is **Generate report**: report generation is part of the job,
not something fired off after it. A job that reports ``done`` therefore means
"the report exists", which is what the modal's "Open report ->" affordance
promises. When only that last stage fails the job still reaches ``done`` (the
chunks/vectors/images are persisted and the paper is genuinely usable for chat
and compare) but the stage is left in ``error`` so the failure is visible
instead of swallowed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "JobStage",
    "JobState",
    "StageStatus",
    "StageProgress",
    "LogEntry",
    "IngestJob",
    "STAGE_ORDER",
]


class JobStage(str, Enum):
    """The seven pipeline stages, in execution order.

    String values are the exact labels shown in the ingestion modal, so keep
    them stable — they are mirrored byte-identically in
    ``frontend/src/api/types.ts`` (``JobStage`` union + ``STAGE_ORDER``).
    """

    DOWNLOAD = "Download"
    EXTRACT_TEXT = "Extract text"
    EXTRACT_IMAGES = "Extract images"
    OCR = "OCR"
    VISION = "Vision analysis"
    EMBEDDING = "Embedding"
    REPORT = "Generate report"


#: Canonical stage order — iterate this rather than relying on enum order.
STAGE_ORDER: list[JobStage] = [
    JobStage.DOWNLOAD,
    JobStage.EXTRACT_TEXT,
    JobStage.EXTRACT_IMAGES,
    JobStage.OCR,
    JobStage.VISION,
    JobStage.EMBEDDING,
    JobStage.REPORT,
]


class JobState(str, Enum):
    """Overall lifecycle state of an ingest job."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class StageStatus(str, Enum):
    """Per-stage status, matching the modal's per-row indicator."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class LogEntry(BaseModel):
    """One line in the live ingestion log.

    Mirrors the UI shape ``{t, m}`` (timestamp, message) and adds the stage it
    belongs to for filtering.
    """

    t: str = Field(..., description="Wall-clock timestamp 'HH:MM:SS'.", examples=["14:03:22"])
    m: str = Field(..., description="Log message.")
    stage: Optional[JobStage] = Field(default=None, description="Originating stage.")


class StageProgress(BaseModel):
    """Progress of a single named stage."""

    stage: JobStage = Field(..., description="Which stage.")
    status: StageStatus = Field(
        default=StageStatus.PENDING, description="pending/running/done/error."
    )
    detail: Optional[str] = Field(
        default=None, description="Optional short status text (e.g. 'conf 0.97')."
    )
    degraded: bool = Field(
        default=False,
        description=(
            "The stage completed (``status == done``) but not at full quality. "
            "Used by the report stage: a local model that exceeds its timeout is "
            "swallowed by the agents' draft-fallback path, so the report is "
            "produced from extractive drafts with no error anywhere. A structured "
            "flag, not a substring of ``detail``, so the UI never has to sniff "
            "status text. Old persisted rows lack the key and default to False."
        ),
    )
    started_at: Optional[datetime] = Field(default=None)
    finished_at: Optional[datetime] = Field(default=None)


class IngestJob(BaseModel):
    """A full ingestion job, as returned by ``GET /jobs/{id}``.

    The orchestrator advances ``stages``, appends to ``log``, and updates
    ``percent`` / ``state``. A newly-created job seeds ``stages`` with every
    :data:`STAGE_ORDER` entry in ``pending`` via :meth:`new`.

    Non-ingestion job runners (chapter reports, study generation) reuse this
    envelope with ``stages = []`` — the pipeline stage names would be lies
    there — so consumers must tolerate an empty or short ``stages`` list rather
    than assuming it matches :data:`STAGE_ORDER`.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(..., description="Job id (see utils.ids.job_id).")
    paper_id: str = Field(..., description="Paper being ingested.")
    arxiv_id: str = Field(..., description="Source arXiv id.")
    state: JobState = Field(default=JobState.QUEUED, description="Overall state.")
    current_stage: Optional[JobStage] = Field(
        default=None, description="Stage currently running, if any."
    )
    percent: int = Field(default=0, ge=0, le=100, description="Overall progress 0-100.")
    stages: list[StageProgress] = Field(
        default_factory=list, description="Per-stage progress, ordered by STAGE_ORDER."
    )
    log: list[LogEntry] = Field(
        default_factory=list, description="Live log lines, oldest first."
    )
    error_message: Optional[str] = Field(
        default=None, description="Populated when state == error."
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def new(cls, *, id: str, paper_id: str, arxiv_id: str) -> "IngestJob":
        """Create a fresh queued job with every STAGE_ORDER stage as pending."""
        return cls(
            id=id,
            paper_id=paper_id,
            arxiv_id=arxiv_id,
            state=JobState.QUEUED,
            percent=0,
            stages=[StageProgress(stage=s) for s in STAGE_ORDER],
            log=[],
        )
