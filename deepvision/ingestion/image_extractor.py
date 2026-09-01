"""Figure/table image extractor — stage 3 (Extract images).

Detects and crops figure/table regions from the PDF into image files plus
thumbnails, producing :class:`ImageChunk` objects.
"""

from __future__ import annotations

import abc
import re
from typing import Any, Optional

from deepvision.ingestion.paths import figures_dir, to_relpath
from deepvision.models.chunks import BBox, ImageChunk, Provenance, SourceRef
from deepvision.utils import get_logger
from deepvision.utils.ids import chunk_id, slugify

__all__ = ["ImageExtractor", "ImageExtractorService"]

log = get_logger(__name__)


class ImageExtractor(abc.ABC):
    """Extracts and crops figure/table images from a PDF."""

    @abc.abstractmethod
    def extract(self, pdf_path: str, paper_id: str) -> list[ImageChunk]:
        """Return :class:`ImageChunk`s with ``image_path``/``thumbnail_path`` set."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# Concrete implementation
# --------------------------------------------------------------------------

#: Ordinal namespace for image chunks — kept disjoint from text/OCR/vision so
#: chunk ids never collide (see pdf_parser/ocr_processor/vision_processor).
IMAGE_ORDINAL_BASE = 100_000

_MIN_DIM_PT = 40.0  # skip tiny icons/glyph-sized raster images
_CAPTION_RE = re.compile(
    r"^(Fig(?:ure)?\.?\s*(\d+)|Table\s*(\d+))\s*[:.]?\s*(.*)$", re.IGNORECASE
)
_CAPTION_SEARCH_MARGIN_PT = 80.0
_RENDER_DPI = 200
_THUMB_DPI = 60


def _find_captions(page: Any, fitz: Any) -> list[tuple[Any, str, str]]:
    """Return ``(rect, label, caption_text)`` for caption-looking blocks on the page."""
    found: list[tuple[Any, str, str]] = []
    try:
        raw = page.get_text("dict")
    except Exception:  # pragma: no cover - defensive
        return found
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        text = " ".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ).strip()
        text = re.sub(r"\s+", " ", text)
        match = _CAPTION_RE.match(text)
        if not match:
            continue
        num = match.group(2) or match.group(3)
        label = f"Fig {num}" if match.group(2) else f"Table {num}"
        bbox = block.get("bbox")
        if not bbox:
            continue
        found.append((fitz.Rect(bbox), label, text[:400]))
    return found


def _nearest_caption(
    rect: Any, captions: list[tuple[Any, str, str]]
) -> tuple[Optional[str], Optional[str]]:
    best: tuple[float, str, str] | None = None
    for cap_rect, label, text in captions:
        # Prefer captions vertically adjacent to the image (below or above).
        vertical_gap = min(
            abs(cap_rect.y0 - rect.y1), abs(rect.y0 - cap_rect.y1)
        )
        horizontal_overlap = min(cap_rect.x1, rect.x1) - max(cap_rect.x0, rect.x0)
        if vertical_gap > _CAPTION_SEARCH_MARGIN_PT or horizontal_overlap < -20:
            continue
        if best is None or vertical_gap < best[0]:
            best = (vertical_gap, label, text)
    if best is None:
        return None, None
    return best[1], best[2]


def _rect_overlaps_any(rect: Any, seen: list[Any]) -> bool:
    for other in seen:
        inter = rect & other
        if not inter.is_empty:
            inter_area = inter.get_area()
            if inter_area >= 0.6 * min(rect.get_area(), other.get_area()):
                return True
    return False


class ImageExtractorService(ImageExtractor):
    """PyMuPDF-based figure/table crop extractor.

    Detects embedded raster images (figures) via ``Page.get_image_info`` and,
    where available, ruled tables via ``Page.find_tables``; crops each region
    to ``data_dir/<paper_id>/figures/`` plus a small thumbnail, and pairs each
    crop with the nearest "Fig N"/"Table N" caption block on the page.
    """

    def extract(self, pdf_path: str, paper_id: str) -> list[ImageChunk]:
        fitz = self._import_fitz()
        doc = fitz.open(pdf_path)
        try:
            chunks: list[ImageChunk] = []
            ordinal = IMAGE_ORDINAL_BASE
            out_dir = figures_dir(paper_id)
            fig_counter = 0

            for page_index in range(doc.page_count):
                page = doc[page_index]
                page_num = page_index + 1
                captions = _find_captions(page, fitz)
                seen_rects: list[Any] = []
                regions: list[tuple[Any, str]] = []

                try:
                    for info in page.get_image_info(xrefs=True):
                        bbox = info.get("bbox")
                        if not bbox:
                            continue
                        rect = fitz.Rect(bbox)
                        if rect.width < _MIN_DIM_PT or rect.height < _MIN_DIM_PT:
                            continue
                        regions.append((rect, "figure"))
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning(
                        "image info extraction failed",
                        extra={"paper_id": paper_id, "page": page_num, "error": str(exc)},
                    )

                try:
                    tables = page.find_tables()
                    for table in tables.tables:
                        rect = fitz.Rect(table.bbox)
                        if rect.width < _MIN_DIM_PT or rect.height < _MIN_DIM_PT:
                            continue
                        regions.append((rect, "table"))
                except Exception:
                    # find_tables() may be unavailable on older PyMuPDF builds,
                    # or find nothing — table detection is best-effort.
                    pass

                for rect, kind in regions:
                    if _rect_overlaps_any(rect, seen_rects):
                        continue
                    seen_rects.append(rect)
                    fig_counter += 1

                    label, caption = _nearest_caption(rect, captions)
                    if not label:
                        label = f"{'Table' if kind == 'table' else 'Fig'} {fig_counter}"
                    slug = slugify(label) or f"fig{fig_counter}"

                    image_path = out_dir / f"{slug}.png"
                    thumb_path = out_dir / f"{slug}_thumb.png"
                    try:
                        zoom = _RENDER_DPI / 72.0
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(zoom, zoom), clip=rect
                        )
                        pix.save(str(image_path))
                        thumb_zoom = _THUMB_DPI / 72.0
                        thumb = page.get_pixmap(
                            matrix=fitz.Matrix(thumb_zoom, thumb_zoom), clip=rect
                        )
                        thumb.save(str(thumb_path))
                    except Exception as exc:  # pragma: no cover - defensive
                        log.warning(
                            "figure crop render failed",
                            extra={
                                "paper_id": paper_id,
                                "page": page_num,
                                "label": label,
                                "error": str(exc),
                            },
                        )
                        continue

                    cid = chunk_id(paper_id, ordinal)
                    chunks.append(
                        ImageChunk(
                            id=cid,
                            paper_id=paper_id,
                            # ImageChunk (unlike the other chunk subtypes) does
                            # not pin a default provenance on the frozen model;
                            # a raw extracted crop is a visual artifact, tagged
                            # VISION so its badge/metadata are well-formed.
                            provenance=Provenance.VISION,
                            page=page_num,
                            bbox=BBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1),
                            image_path=to_relpath(image_path),
                            thumbnail_path=to_relpath(thumb_path),
                            source_ref=SourceRef(
                                section_label=label,
                                page_label=f"page {page_num}",
                                page=page_num,
                            ),
                            ordinal=ordinal,
                            figure_label=label,
                            caption=caption,
                            width=pix.width,
                            height=pix.height,
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
                "PyMuPDF ('fitz') is required for image extraction but is not installed"
            ) from exc
        return fitz
