"""Paper metadata model and lifecycle status.

``PaperMeta`` is the canonical description of a paper in the library. It backs
the search-result cards, the library grid cards, and the report header. The
``PaperStatus`` enum drives the coloured status badge on library cards
(READY / PROCESSING / ERROR) and the queued state before ingestion starts.

**Two independent readiness signals, on purpose.** ``status``/``ingested`` mean
"ingestion finished, this paper's content is usable" — they flip as soon as the
embedding stage lands, and they gate Chat and Compare. ``has_report`` means "a
report row actually exists" and gates only the Library card's "Open report".
Collapsing them back into one flag re-creates one of two bugs: either the card
offers a report that does not exist yet (whose click starts a second, competing
agent run), or chat/compare stay locked for the whole length of report
generation — up to ~20 minutes on a local model.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["PaperStatus", "PaperMeta"]


class PaperStatus(str, Enum):
    """Lifecycle state of a paper in the library.

    UI mapping: ``ready`` -> green READY, ``processing`` -> amber spinner,
    ``error`` -> red ERROR. ``queued`` is the pre-ingestion state (accepted but
    not yet started).
    """

    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class PaperMeta(BaseModel):
    """Bibliographic metadata for one paper.

    Populated at search time from arXiv (title/authors/abstract/categories) and
    enriched during ingestion (``status``, ``thumbnail_path``, stat counts).
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(
        ..., description="Internal paper id (see utils.ids.paper_id_from_arxiv)."
    )
    arxiv_id: str = Field(
        ..., description="Bare arXiv id, e.g. '2411.10842'.", examples=["2411.10842"]
    )
    arxiv_label: str = Field(
        default="",
        description="Display label, e.g. 'arXiv:2411.10842'.",
        examples=["arXiv:2411.10842"],
    )
    version: Optional[str] = Field(
        default=None, description="arXiv version suffix, e.g. 'v2'."
    )
    title: str = Field(..., description="Paper title.")
    authors: list[str] = Field(
        default_factory=list, description="Ordered list of author display names."
    )
    abstract: str = Field(default="", description="Abstract text.")
    categories: list[str] = Field(
        default_factory=list,
        description="arXiv category codes, e.g. ['cs.CL', 'cs.LG'].",
    )
    published: Optional[date] = Field(
        default=None, description="Publication date (arXiv 'published')."
    )
    updated: Optional[date] = Field(
        default=None, description="Last-updated date on arXiv."
    )
    pdf_url: Optional[str] = Field(default=None, description="Canonical arXiv PDF URL.")
    abs_url: Optional[str] = Field(default=None, description="arXiv abstract page URL.")

    # ---- Library / ingestion state -------------------------------------
    status: PaperStatus = Field(
        default=PaperStatus.QUEUED, description="Current lifecycle status."
    )
    ingested: bool = Field(
        default=False,
        description=(
            "True once ingestion finished and the paper's content is usable "
            "(chunks + vectors + images persisted). Gates chat and compare. "
            "Says nothing about whether a report exists — see has_report."
        ),
    )
    has_report: bool = Field(
        default=False,
        description=(
            "True iff a whole-paper report row exists for this paper. Gates the "
            "Library card's 'Open report' button. DERIVED at read time from the "
            "existence of a ``reports`` row — never persisted on ``papers``, so "
            "it cannot go stale."
        ),
    )
    thumbnail_path: Optional[str] = Field(
        default=None,
        description="Path (relative to data_dir) to the page-thumbnail image.",
    )
    page_count: Optional[int] = Field(default=None, ge=0)
    figure_count: Optional[int] = Field(default=None, ge=0)
    error_message: Optional[str] = Field(
        default=None, description="Populated when status == error."
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def authors_str(self) -> str:
        """Comma-joined author string for compact card display."""
        return ", ".join(self.authors)
