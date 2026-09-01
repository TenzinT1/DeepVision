"""Jobs router — GET /jobs/{job_id}.

Owned by: INGESTION domain. Returns the current job snapshot (state, current
stage, percent, per-stage status, live log) for the ingestion modal to poll.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from deepvision.api.schemas import JobStatusResponse
from deepvision.ingestion import repo

router = APIRouter(tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    """Return the current state of ingest job ``job_id``."""
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse(job=job)
