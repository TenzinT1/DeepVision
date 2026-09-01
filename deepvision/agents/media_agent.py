"""Media agent — assembles the Figures section.

Turns image + vision-insight + OCR chunks into :class:`MediaRef` cards (full
image, thumbnail, caption, VISION/OCR provenance) and the Figures
:class:`Section`.
Purely structural (no LLM involved): it joins an :class:`ImageChunk` crop back
to the :class:`VisionInsightChunk` (caption/analysis) and/or
:class:`OCRChunk` (recovered table text) that describe the *same* crop.
``VisionProcessor``/``OCRProcessor`` both derive their output from the paper's
``ImageChunk`` list, so a shared ``image_path`` is the primary join key; a
same-page fallback covers cases where the image path wasn't preserved. Orphan
vision/OCR chunks (e.g. only the insight was retrieved, not its source crop —
common for chat's top-k context) still produce a usable, if image-less, card.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath
from typing import Optional, Sequence

from deepvision.agents.base import clean_snippet
from deepvision.models.chunks import (
    AnyChunk,
    ImageChunk,
    OCRChunk,
    Provenance,
    VisionInsightChunk,
)
from deepvision.models.report import SECTION_QUESTIONS, MediaRef, Section, SectionName
from deepvision.utils.ids import media_id, section_id

__all__ = ["MediaAgent"]


def _filename(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return PurePosixPath(path).name


def _infer_kind(label: Optional[str]) -> str:
    return "table" if label and "table" in label.lower() else "figure"


class MediaAgent:
    """Builds figure/table media refs and the Figures section."""

    def build_media(self, chunks: Sequence[AnyChunk]) -> list[MediaRef]:
        """Return :class:`MediaRef`s for the paper's figures/tables."""
        chunk_list = list(chunks)
        image_chunks = [c for c in chunk_list if isinstance(c, ImageChunk)]
        vision_chunks = [c for c in chunk_list if isinstance(c, VisionInsightChunk)]
        ocr_chunks = [c for c in chunk_list if isinstance(c, OCRChunk)]

        vision_by_path: dict[str, VisionInsightChunk] = {}
        vision_by_page: dict[int, list[VisionInsightChunk]] = defaultdict(list)
        for v in vision_chunks:
            if v.image_path:
                vision_by_path[v.image_path] = v
            else:
                vision_by_page[v.page].append(v)

        ocr_by_path: dict[str, OCRChunk] = {}
        ocr_by_page: dict[int, list[OCRChunk]] = defaultdict(list)
        for o in ocr_chunks:
            if o.image_path:
                ocr_by_path[o.image_path] = o
            else:
                ocr_by_page[o.page].append(o)

        used_vision_ids: set[str] = set()
        used_ocr_ids: set[str] = set()
        media: list[MediaRef] = []

        for img in sorted(image_chunks, key=lambda c: (c.page, c.ordinal)):
            vi = vision_by_path.get(img.image_path) if img.image_path else None
            if vi is None and vision_by_page.get(img.page):
                vi = vision_by_page[img.page].pop(0)
            oc = ocr_by_path.get(img.image_path) if img.image_path else None
            if oc is None and ocr_by_page.get(img.page):
                oc = ocr_by_page[img.page].pop(0)
            if vi is not None:
                used_vision_ids.add(vi.id)
            if oc is not None:
                used_ocr_ids.add(oc.id)

            label = img.figure_label or (vi.figure_label if vi else None) or f"Figure (p{img.page})"
            caption = (
                img.caption
                or (vi.text.strip() if vi and vi.text else None)
                or (vi.caption if vi and vi.caption else None)
                or (clean_snippet(oc.text, max_chars=220) if oc else None)
                or ""
            )
            if vi is not None:
                provenance = Provenance.VISION
                chunk_ref = vi.id
            elif oc is not None:
                provenance = Provenance.OCR
                chunk_ref = oc.id
            else:
                # No downstream insight yet (vision/OCR stage skipped, disabled,
                # or degraded) — still surface the raw crop rather than drop it.
                provenance = Provenance.VISION
                chunk_ref = img.id

            media.append(
                MediaRef(
                    id=media_id(),
                    kind=_infer_kind(label),
                    label=label,
                    caption=caption,
                    provenance=provenance,
                    image_path=img.image_path,
                    thumbnail_path=img.thumbnail_path,
                    filename=_filename(img.image_path),
                    page=img.page,
                    chunk_id=chunk_ref,
                )
            )

        # Orphan vision insights: their source ImageChunk wasn't in this chunk
        # set (e.g. chat top-k only surfaced the insight text).
        for vi in vision_chunks:
            if vi.id in used_vision_ids:
                continue
            label = vi.figure_label or f"Figure (p{vi.page})"
            media.append(
                MediaRef(
                    id=media_id(),
                    kind=_infer_kind(label),
                    label=label,
                    caption=vi.caption or clean_snippet(vi.text, max_chars=220),
                    provenance=Provenance.VISION,
                    image_path=vi.image_path,
                    thumbnail_path=None,
                    filename=_filename(vi.image_path),
                    page=vi.page,
                    chunk_id=vi.id,
                )
            )

        # Orphan OCR-derived tables: likewise, no companion ImageChunk present.
        for oc in ocr_chunks:
            if oc.id in used_ocr_ids:
                continue
            media.append(
                MediaRef(
                    id=media_id(),
                    kind="table",
                    label=f"Table (p{oc.page})",
                    caption=clean_snippet(oc.text, max_chars=220),
                    provenance=Provenance.OCR,
                    image_path=oc.image_path,
                    thumbnail_path=None,
                    filename=_filename(oc.image_path),
                    page=oc.page,
                    chunk_id=oc.id,
                )
            )

        return media

    def build_figures_section(self, chunks: Sequence[AnyChunk]) -> Section:
        """Return the assembled Figures :class:`Section`."""
        media = self.build_media(chunks)
        provenance: list[Provenance] = []
        for ref in media:
            if ref.provenance not in provenance:
                provenance.append(ref.provenance)
        return Section(
            id=section_id(),
            name=SectionName.FIGURES,
            question=SECTION_QUESTIONS.get(SectionName.FIGURES),
            body_markdown="",
            deep_dive_markdown=None,
            # `None` means "no badge chosen here"; `normalize_sections` fills it
            # from SECTION_BADGES. Only the report layer knows the badge table.
            badge=None,
            provenance=provenance,
            citations=[],
            media=media,
            confidence=None,
            default_open=False,
        )
