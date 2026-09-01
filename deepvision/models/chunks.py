"""Chunk data models — the atomic retrievable units of a paper.

Every piece of extracted content becomes a ``Chunk`` subtype carrying its page,
bounding box, source reference and provenance. Chunks are what the RAG layer
embeds and retrieves, what citations point at, and what ``GET /media/{chunk_id}``
serves for image-bearing chunks.

Provenance (``text`` / ``ocr`` / ``vision``) drives the coloured badges in the
UI (plain text, amber "OCR", teal "VISION"). Keep the string values stable — the
frontend switches on them directly.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Modality",
    "Provenance",
    "BBox",
    "SourceRef",
    "Chunk",
    "TextChunk",
    "OCRChunk",
    "VisionInsightChunk",
    "ImageChunk",
    "AnyChunk",
]


class Modality(str, Enum):
    """The kind of content a chunk carries."""

    TEXT = "text"
    OCR = "ocr"
    VISION = "vision"
    IMAGE = "image"


class Provenance(str, Enum):
    """Which pipeline processor produced the content.

    Rendered as a badge in the report and figures UI:
    ``TEXT`` (plain), ``OCR`` (amber), ``VISION`` (teal).
    """

    TEXT = "text"
    OCR = "ocr"
    VISION = "vision"


class BBox(BaseModel):
    """A bounding box on a rendered page, in PDF points (origin top-left).

    Used to locate a citation snippet, figure crop, or OCR region on the page
    image so the UI can highlight it.
    """

    model_config = ConfigDict(frozen=True)

    x0: float = Field(..., description="Left edge, in points.")
    y0: float = Field(..., description="Top edge, in points.")
    x1: float = Field(..., description="Right edge, in points.")
    y1: float = Field(..., description="Bottom edge, in points.")

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(x0, y0, x1, y1)``."""
        return (self.x0, self.y0, self.x1, self.y1)


class SourceRef(BaseModel):
    """Human-readable pointer back to where content came from.

    Mirrors the citation popover in the UI: a section label and a page label,
    plus the machine ``page`` number for rendering the page image.
    """

    section_label: str = Field(
        ...,
        description="e.g. 'Section 1 · Introduction' or 'Table 4'.",
        examples=["Section 1 · Introduction"],
    )
    page_label: str = Field(
        ..., description="Human page label, e.g. 'page 2'.", examples=["page 2"]
    )
    page: int = Field(..., ge=1, description="1-based page number.")


class Chunk(BaseModel):
    """Base class for all retrievable units.

    Subtypes set a fixed ``modality`` discriminator so a heterogeneous list of
    chunks can be validated and dispatched on. ``embedding`` is populated by the
    RAG layer; the primary store of vectors is ChromaDB, this field is optional
    and mainly for transport/debug.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(..., description="Deterministic chunk id (see utils.ids.chunk_id).")
    paper_id: str = Field(..., description="Owning paper id.")
    modality: Modality = Field(..., description="Content kind discriminator.")
    provenance: Provenance = Field(
        ..., description="Which processor produced this content."
    )
    text: str = Field(
        default="",
        description="Text payload (the searchable content). Empty for pure images.",
    )
    page: int = Field(..., ge=1, description="1-based source page number.")
    bbox: Optional[BBox] = Field(
        default=None, description="Region on the page image, if localized."
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Path (relative to data_dir) to a crop/page image, if any.",
    )
    source_ref: Optional[SourceRef] = Field(
        default=None, description="Human-readable back-reference for citations."
    )
    ordinal: int = Field(
        default=0, ge=0, description="Position of this chunk within the paper."
    )
    token_count: Optional[int] = Field(
        default=None, ge=0, description="Approximate token length of `text`."
    )
    embedding: Optional[list[float]] = Field(
        default=None, description="Optional inline embedding vector."
    )


class TextChunk(Chunk):
    """A chunk of natively-extracted PDF text (provenance = TEXT)."""

    modality: Literal[Modality.TEXT] = Modality.TEXT
    provenance: Literal[Provenance.TEXT] = Provenance.TEXT


class OCRChunk(Chunk):
    """Text recovered by OCR from a raster region (provenance = OCR).

    ``ocr_confidence`` is the mean recognizer confidence in ``[0, 1]``.
    """

    modality: Literal[Modality.OCR] = Modality.OCR
    provenance: Literal[Provenance.OCR] = Provenance.OCR
    ocr_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Mean OCR confidence."
    )
    language: Optional[str] = Field(
        default=None, description="OCR language code used (e.g. 'eng')."
    )


class VisionInsightChunk(Chunk):
    """A vision-model description/caption of a figure (provenance = VISION).

    ``image_path`` points at the figure crop the insight describes; ``text`` is
    the generated caption/analysis.
    """

    modality: Literal[Modality.VISION] = Modality.VISION
    provenance: Literal[Provenance.VISION] = Provenance.VISION
    figure_label: Optional[str] = Field(
        default=None, description="e.g. 'Fig 2' or 'Table 4'."
    )
    caption: Optional[str] = Field(
        default=None, description="Original caption text near the figure, if found."
    )


class ImageChunk(Chunk):
    """A raw figure/table image crop (modality = IMAGE).

    Carries no primary text; may be embedded visually when the advanced
    image-embedding toggle is on. ``thumbnail_path`` backs the small preview in
    the library and figures grid.
    """

    modality: Literal[Modality.IMAGE] = Modality.IMAGE
    figure_label: Optional[str] = Field(default=None, description="e.g. 'Fig 3'.")
    caption: Optional[str] = Field(default=None, description="Figure caption text.")
    thumbnail_path: Optional[str] = Field(
        default=None, description="Path to a small thumbnail crop, relative to data_dir."
    )
    width: Optional[int] = Field(default=None, ge=1, description="Image width in px.")
    height: Optional[int] = Field(default=None, ge=1, description="Image height in px.")


# Discriminated union of concrete chunk types, dispatched on ``modality``.
AnyChunk = Annotated[
    TextChunk | OCRChunk | VisionInsightChunk | ImageChunk,
    Field(discriminator="modality"),
]
