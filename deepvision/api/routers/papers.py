"""Papers router — GET /papers, DELETE /papers/{id}, POST /papers/{id}/rerun.

Owned by: INGESTION domain (library management). Backs the Library grid: list,
delete, and re-run pipeline.
"""

from __future__ import annotations

import shutil

from fastapi import APIRouter, HTTPException

from deepvision.api.deps import get_settings
from deepvision.api.schemas import (
    ChaptersResponse,
    DeletePaperResponse,
    PapersListResponse,
    RerunResponse,
)
from deepvision.config import get_config
from deepvision.ingestion import repo
from deepvision.ingestion.chapters import derive_chapters, page_count_of
from deepvision.ingestion.orchestrator import DefaultIngestionOrchestrator
from deepvision.utils import get_logger

router = APIRouter(tags=["papers"])

log = get_logger(__name__)

_orchestrator = DefaultIngestionOrchestrator()


def _delete_paper_vectors(paper_id: str) -> None:
    try:
        from deepvision.ingestion.orchestrator import build_vector_store

        vector_store = build_vector_store()
        vector_store.delete_paper(paper_id)
    except Exception as exc:
        log.warning(
            "failed to delete vectors for paper (continuing)",
            extra={"paper_id": paper_id, "error": str(exc)},
        )


@router.get("/papers", response_model=PapersListResponse)
def list_papers() -> PapersListResponse:
    """List all papers in the library (any status)."""
    papers = repo.list_papers()
    return PapersListResponse(papers=papers, count=len(papers))


@router.delete("/papers/{paper_id}", response_model=DeletePaperResponse)
def delete_paper(paper_id: str) -> DeletePaperResponse:
    """Delete a paper, its report, chunks, vectors, and local assets."""
    existed = repo.delete_paper_rows(paper_id)
    if not existed:
        raise HTTPException(status_code=404, detail="paper not found")

    _delete_paper_vectors(paper_id)

    try:
        directory = get_config().data_dir / paper_id
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
    except Exception as exc:
        log.warning(
            "failed to remove paper data directory (continuing)",
            extra={"paper_id": paper_id, "error": str(exc)},
        )

    return DeletePaperResponse(paper_id=paper_id, deleted=True)


@router.get("/papers/{paper_id}/chapters", response_model=ChaptersResponse)
def get_paper_chapters(paper_id: str) -> ChaptersResponse:
    """Return the chapter list for the Report page's picker.

    Cheap by contract: opens the PDF (or re-runs the pure-PyMuPDF heading
    heuristic) and makes no model calls. 404 only when the paper itself is
    unknown; a missing/corrupt PDF on disk degrades to an empty, safe payload
    (``detected: false``) rather than raising.
    """
    paper = repo.get_paper_meta(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")

    try:
        chapters = derive_chapters(paper_id)
        pages = page_count_of(paper_id)
    except Exception as exc:  # pragma: no cover - defensive, must never 500
        log.warning(
            "chapter extraction failed (returning empty payload)",
            extra={"paper_id": paper_id, "error": str(exc)},
        )
        chapters, pages = [], 0

    return ChaptersResponse(
        paper_id=paper_id,
        chapters=chapters,
        source=chapters[0].source if chapters else None,
        page_count=pages,
        detected=bool(chapters),
    )


@router.post("/papers/{paper_id}/rerun", response_model=RerunResponse, status_code=202)
def rerun_paper(paper_id: str) -> RerunResponse:
    """Re-run the full ingestion pipeline for an existing paper."""
    paper = repo.get_paper_meta(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")

    repo.upsert_paper_from_meta(paper, reset_status=True)
    _delete_paper_vectors(paper_id)

    settings = get_settings()
    try:
        job = _orchestrator.start_async(paper_id, settings)
    except Exception as exc:
        log.error("failed to start rerun", extra={"paper_id": paper_id, "error": str(exc)})
        raise HTTPException(status_code=500, detail=f"failed to start rerun: {exc}") from exc

    return RerunResponse(job=job, paper_id=paper_id)
