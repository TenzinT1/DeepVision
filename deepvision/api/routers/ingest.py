"""Ingest router — POST /ingest.

Owned by: INGESTION domain (orchestrator). Creates a paper row + an async
ingest job and kicks off the pipeline; returns the job to poll.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from deepvision.api.deps import get_settings
from deepvision.api.schemas import IngestRequest, IngestResponse
from deepvision.ingestion import repo
from deepvision.ingestion.arxiv_search import ArxivSearcherService
from deepvision.ingestion.orchestrator import DefaultIngestionOrchestrator
from deepvision.utils import get_logger
from deepvision.utils.ids import normalize_arxiv_id, paper_id_from_arxiv

router = APIRouter(tags=["ingest"])

log = get_logger(__name__)

_searcher = ArxivSearcherService()
_orchestrator = DefaultIngestionOrchestrator()


@router.post("/ingest", response_model=IngestResponse, status_code=202)
def ingest(request: IngestRequest) -> IngestResponse:
    """Queue an async ingestion job for ``request.arxiv_id``."""
    try:
        arxiv_id = normalize_arxiv_id(request.arxiv_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    paper_id = paper_id_from_arxiv(arxiv_id)
    settings = request.settings_override or get_settings()

    paper = repo.get_paper_meta(paper_id)
    if paper is None:
        try:
            paper = _searcher.fetch_meta(arxiv_id)
        except Exception as exc:
            log.error("fetch_meta failed", extra={"arxiv_id": arxiv_id, "error": str(exc)})
            raise HTTPException(
                status_code=502, detail=f"could not fetch arXiv metadata: {exc}"
            ) from exc
    repo.upsert_paper_from_meta(paper, reset_status=True)

    try:
        job = _orchestrator.start_async(paper_id, settings)
    except Exception as exc:
        log.error("failed to start ingestion", extra={"paper_id": paper_id, "error": str(exc)})
        raise HTTPException(status_code=500, detail=f"failed to start ingestion: {exc}") from exc

    return IngestResponse(job=job, paper_id=paper_id)
