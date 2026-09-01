"""PDF text parser — stage 2 (Extract text).

Extracts native text (paragraphs, sections, page numbers, bboxes) via PyMuPDF
into :class:`TextChunk` objects.
"""

from __future__ import annotations

import abc
import re
from collections import Counter
from typing import Any, Optional

from deepvision.models.chunks import BBox, SourceRef, TextChunk
from deepvision.utils import get_logger
from deepvision.utils.ids import chunk_id

__all__ = ["PDFParser", "PDFParserService"]

log = get_logger(__name__)


class PDFParser(abc.ABC):
    """Parses native PDF text into localized text chunks."""

    @abc.abstractmethod
    def parse(self, pdf_path: str, paper_id: str) -> list[TextChunk]:
        """Return ordered :class:`TextChunk`s with page/bbox/source_ref populated."""
        raise NotImplementedError

    @abc.abstractmethod
    def page_count(self, pdf_path: str) -> int:
        """Return the number of pages in the PDF."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# Concrete implementation
# --------------------------------------------------------------------------

#: Ordinal namespace for text chunks; other stages start at higher bases so
#: chunk ids never collide within a paper (see image_extractor/ocr/vision).
TEXT_ORDINAL_BASE = 0

_HEADING_KEYWORDS = {
    "abstract",
    "references",
    "acknowledgments",
    "acknowledgements",
    "appendix",
    "introduction",
    "conclusion",
    "conclusions",
    "related work",
}
_NUMBERED_HEADING_RE = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,2}\.?)\s+[A-Z][A-Za-z].{0,80}$")


def _looks_like_heading(text: str, size: float, body_size: float) -> bool:
    text = text.strip()
    if not text or len(text) > 100:
        return False
    if size >= body_size + 1.5:
        return True
    if _NUMBERED_HEADING_RE.match(text):
        return True
    if text.rstrip(":.").lower() in _HEADING_KEYWORDS:
        return True
    return False


def _dominant_body_size(doc: Any) -> float:
    """Estimate the paper's base body-text font size from a sample of pages."""
    sizes: Counter[float] = Counter()
    sample_pages = list(range(min(doc.page_count, 6)))
    for idx in sample_pages:
        page = doc[idx]
        try:
            raw = page.get_text("dict")
        except Exception:  # pragma: no cover - defensive
            continue
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(float(span.get("size", 0.0)), 1)
                    text_len = len(span.get("text", "").strip())
                    if size > 0 and text_len > 0:
                        sizes[size] += text_len
    if not sizes:
        return 10.0
    return sizes.most_common(1)[0][0]


class PDFParserService(PDFParser):
    """PyMuPDF-based text extractor with a light section-heading heuristic."""

    def page_count(self, pdf_path: str) -> int:
        fitz = self._import_fitz()
        doc = fitz.open(pdf_path)
        try:
            return doc.page_count
        finally:
            doc.close()

    def parse(self, pdf_path: str, paper_id: str) -> list[TextChunk]:
        fitz = self._import_fitz()
        doc = fitz.open(pdf_path)
        try:
            body_size = _dominant_body_size(doc)
            chunks: list[TextChunk] = []
            ordinal = TEXT_ORDINAL_BASE
            current_section = "Document"

            for page_index in range(doc.page_count):
                page = doc[page_index]
                page_num = page_index + 1
                try:
                    raw = page.get_text("dict")
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning(
                        "page text extraction failed",
                        extra={"paper_id": paper_id, "page": page_num, "error": str(exc)},
                    )
                    continue

                for block in raw.get("blocks", []):
                    if block.get("type") != 0:
                        continue
                    lines = block.get("lines", [])
                    if not lines:
                        continue
                    text_parts: list[str] = []
                    max_size = 0.0
                    for line in lines:
                        line_text = "".join(
                            span.get("text", "") for span in line.get("spans", [])
                        )
                        text_parts.append(line_text)
                        for span in line.get("spans", []):
                            max_size = max(max_size, float(span.get("size", 0.0)))
                    text = re.sub(r"[ \t]+", " ", " ".join(text_parts)).strip()
                    if not text:
                        continue

                    if _looks_like_heading(text, max_size, body_size):
                        current_section = text.rstrip(":.")
                        continue

                    bbox_raw = block.get("bbox")
                    bbox: Optional[BBox] = None
                    if bbox_raw and len(bbox_raw) == 4:
                        bbox = BBox(
                            x0=float(bbox_raw[0]),
                            y0=float(bbox_raw[1]),
                            x1=float(bbox_raw[2]),
                            y1=float(bbox_raw[3]),
                        )

                    cid = chunk_id(paper_id, ordinal)
                    chunks.append(
                        TextChunk(
                            id=cid,
                            paper_id=paper_id,
                            text=text,
                            page=page_num,
                            bbox=bbox,
                            source_ref=SourceRef(
                                section_label=current_section,
                                page_label=f"page {page_num}",
                                page=page_num,
                            ),
                            ordinal=ordinal,
                            token_count=max(1, len(text) // 4),
                        )
                    )
                    ordinal += 1
            return chunks
        finally:
            doc.close()

    @staticmethod
    def _import_fitz():
        try:
            import fitz  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "PyMuPDF ('fitz') is required for PDF parsing but is not installed"
            ) from exc
        return fitz
