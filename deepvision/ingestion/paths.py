"""Local file-store path helpers shared across the ingestion stages.

Every asset the pipeline writes (PDFs, page renders, figure crops, thumbnails)
lives under ``config.data_dir/<paper_id>/...``. Model ``*_path`` fields are
stored **relative to** ``data_dir``; these
helpers centralize the absolute/relative conversions so every stage module
agrees on the same layout. internal helper module, not part of the fixed
"""

from __future__ import annotations

from pathlib import Path

from deepvision.config import get_config

__all__ = [
    "paper_dir",
    "pdf_path",
    "pages_dir",
    "figures_dir",
    "to_relpath",
    "to_abspath",
]


def paper_dir(paper_id: str) -> Path:
    """Return (and ensure) the root data directory for ``paper_id``."""
    d = get_config().data_dir / paper_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def pdf_path(paper_id: str) -> Path:
    """Return the canonical local path for a paper's main PDF."""
    return paper_dir(paper_id) / "main.pdf"


def pages_dir(paper_id: str) -> Path:
    """Return (and ensure) the rendered-pages directory for ``paper_id``."""
    d = paper_dir(paper_id) / "pages"
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir(paper_id: str) -> Path:
    """Return (and ensure) the figure-crop directory for ``paper_id``."""
    d = paper_dir(paper_id) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def to_relpath(abs_path: Path | str) -> str:
    """Convert an absolute path under ``data_dir`` to the stored relative form."""
    p = Path(abs_path)
    try:
        return str(p.relative_to(get_config().data_dir))
    except ValueError:
        # Already relative or outside data_dir; return as-is (posix style).
        return str(p)


def to_abspath(rel_path: Path | str) -> Path:
    """Convert a stored relative (to ``data_dir``) path to an absolute path."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return get_config().data_dir / p
