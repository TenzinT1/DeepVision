"""Search router — GET /search.

Owned by: INGESTION domain (arxiv_search). Architect-owned decorators/signatures;
body raises 501 until implemented.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from deepvision.api.schemas import (
    SearchDateRange,
    SearchResponse,
    SearchResultItem,
    SearchSort,
)
from deepvision.db import session_scope
from deepvision.db.schema import PaperRow
from deepvision.ingestion.arxiv_search import ArxivSearcherService
from deepvision.ingestion.repo import paper_ids_with_reports, paper_row_to_meta
from deepvision.utils import get_logger

router = APIRouter(tags=["search"])

log = get_logger(__name__)

_searcher = ArxivSearcherService()


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Search query (titles, authors, abstracts)."),
    category: str | None = Query(
        default=None, description="arXiv category code, e.g. 'cs.CL'."
    ),
    date_range: SearchDateRange = Query(
        default=SearchDateRange.ANY, description="Date-range filter."
    ),
    sort: SearchSort = Query(default=SearchSort.RELEVANCE, description="Sort order."),
    limit: int = Query(default=25, ge=1, le=100),
) -> SearchResponse:
    """Search arXiv and mark which hits are already in the library."""
    try:
        raw_results = _searcher.search(
            q, category=category, date_range=date_range, sort=sort, limit=limit
        )
    except Exception as exc:
        log.error("arxiv search failed", extra={"query": q, "error": str(exc)})
        raise HTTPException(status_code=502, detail=f"arXiv search failed: {exc}") from exc

    # One grouped reports lookup for the whole result page (not one per hit) so
    # PaperMeta.has_report is populated here exactly as it is on GET /papers —
    # a field that is right in one response and wrong in another is worse than
    # not having it.
    with_report = paper_ids_with_reports(item.paper.id for item in raw_results)

    enriched: list[SearchResultItem] = []
    with session_scope() as session:
        for item in raw_results:
            row = session.get(PaperRow, item.paper.id)
            if row is not None:
                enriched.append(
                    SearchResultItem(
                        paper=paper_row_to_meta(
                            row, has_report=row.id in with_report
                        ),
                        ingested=row.ingested,
                        in_library=True,
                    )
                )
            else:
                enriched.append(item)

    return SearchResponse(query=q, results=enriched, count=len(enriched))
