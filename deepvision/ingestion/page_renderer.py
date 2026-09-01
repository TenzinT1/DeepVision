"""Page renderer — full-page raster images.

Renders each PDF page to a PNG (via PyMuPDF) for citation popovers, library
thumbnails, and OCR input.
"""

from __future__ import annotations

import abc

from deepvision.ingestion.paths import pages_dir, to_relpath
from deepvision.utils import get_logger

__all__ = ["PageRenderer", "PageRendererService"]

log = get_logger(__name__)


class PageRenderer(abc.ABC):
    """Renders PDF pages to raster images on disk."""

    @abc.abstractmethod
    def render_all(self, pdf_path: str, paper_id: str, *, dpi: int = 150) -> list[str]:
        """Render every page; return the list of page-image paths (page-ordered)."""
        raise NotImplementedError

    @abc.abstractmethod
    def render_page(
        self, pdf_path: str, paper_id: str, page: int, *, dpi: int = 150
    ) -> str:
        """Render a single 1-based ``page`` and return its image path."""
        raise NotImplementedError

    @abc.abstractmethod
    def thumbnail(self, pdf_path: str, paper_id: str) -> str:
        """Render the first-page thumbnail used on library/search cards."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# Concrete implementation
# --------------------------------------------------------------------------

class PageRendererService(PageRenderer):
    """PyMuPDF-based page rasterizer.

    Pages render to ``data_dir/<paper_id>/pages/p{n}.png``; the model
    ``*_path`` fields store the path relative to ``data_dir``.
    """

    def render_all(self, pdf_path: str, paper_id: str, *, dpi: int = 150) -> list[str]:
        fitz = self._import_fitz()
        doc = fitz.open(pdf_path)
        try:
            paths: list[str] = []
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            out_dir = pages_dir(paper_id)
            for page_index in range(doc.page_count):
                page = doc[page_index]
                page_num = page_index + 1
                out_path = out_dir / f"p{page_num}.png"
                pix = page.get_pixmap(matrix=matrix)
                pix.save(str(out_path))
                paths.append(to_relpath(out_path))
            return paths
        finally:
            doc.close()

    def render_page(
        self, pdf_path: str, paper_id: str, page: int, *, dpi: int = 150
    ) -> str:
        fitz = self._import_fitz()
        doc = fitz.open(pdf_path)
        try:
            if page < 1 or page > doc.page_count:
                raise ValueError(f"page {page} out of range (1..{doc.page_count})")
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            out_path = pages_dir(paper_id) / f"p{page}.png"
            pix = doc[page - 1].get_pixmap(matrix=matrix)
            pix.save(str(out_path))
            return to_relpath(out_path)
        finally:
            doc.close()

    def thumbnail(self, pdf_path: str, paper_id: str) -> str:
        # The library thumbnail reuses the full-resolution first-page render
        # typically invoke render_all() immediately beforehand, which already
        # writes pages/p1.png at the standard page DPI -- reuse that file
        # instead of re-opening the PDF and re-rasterizing page 1. Only fall
        # back to rendering when the file isn't already on disk (e.g.
        # thumbnail() called on its own, or render_all() produced 0 pages).
        existing = pages_dir(paper_id) / "p1.png"
        if existing.exists():
            return to_relpath(existing)
        try:
            return self.render_page(pdf_path, paper_id, 1, dpi=150)
        except Exception as exc:
            log.warning(
                "thumbnail render failed",
                extra={"paper_id": paper_id, "error": str(exc)},
            )
            raise

    @staticmethod
    def _import_fitz():
        try:
            import fitz  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "PyMuPDF ('fitz') is required for page rendering but is not installed"
            ) from exc
        return fitz
