"""Vision processor — stage 5 (Vision analysis).

Captions/analyzes figure crops with the configured :class:`VisionProvider`,
producing :class:`VisionInsightChunk` objects.
"""

from __future__ import annotations

import abc

from deepvision.ingestion.paths import to_abspath
from deepvision.models.chunks import ImageChunk, SourceRef, VisionInsightChunk
from deepvision.providers.base import VisionProvider
from deepvision.utils import get_logger
from deepvision.utils.ids import chunk_id

__all__ = ["VisionProcessor", "VisionProcessorService"]

log = get_logger(__name__)


class VisionProcessor(abc.ABC):
    """Generates vision-model insights for figure/table crops."""

    @abc.abstractmethod
    def process(
        self, images: list[ImageChunk], paper_id: str, vision: VisionProvider
    ) -> list[VisionInsightChunk]:
        """Describe each image via ``vision``; return insight chunks."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# Concrete implementation
# --------------------------------------------------------------------------

#: Ordinal namespace for vision-insight chunks — kept disjoint from the rest.
VISION_ORDINAL_BASE = 300_000

_PROMPT = (
    "Describe this figure or table from a research paper in 2-3 sentences. "
    "Focus on what it shows and any notable trend or result."
)


def _placeholder_text(image: ImageChunk) -> str:
    label = image.figure_label or "This figure"
    if image.caption:
        return f"{label}: {image.caption}"
    return f"{label} — vision analysis unavailable; showing the extracted crop only."


class VisionProcessorService(VisionProcessor):
    """Captions each figure/table crop via the injected :class:`VisionProvider`.

    Never raises: if the provider is unavailable, unimplemented, or a single
    call fails, that image gets a placeholder insight (built from its nearby
    caption when one was recovered) so the pipeline always completes.
    """

    def process(
        self, images: list[ImageChunk], paper_id: str, vision: VisionProvider
    ) -> list[VisionInsightChunk]:
        if not images:
            return []

        available = True
        try:
            available = bool(vision.availability().available)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "vision availability check failed, will attempt per-image anyway",
                extra={"paper_id": paper_id, "error": str(exc)},
            )

        chunks: list[VisionInsightChunk] = []
        ordinal = VISION_ORDINAL_BASE

        for image in images:
            text: str
            if available and image.image_path:
                try:
                    abs_path = str(to_abspath(image.image_path))
                    text = vision.describe_image(
                        abs_path, prompt=_PROMPT, caption_hint=image.caption
                    )
                    text = (text or "").strip() or _placeholder_text(image)
                except Exception as exc:
                    log.warning(
                        "vision describe_image failed, using placeholder",
                        extra={
                            "paper_id": paper_id,
                            "image_path": image.image_path,
                            "error": str(exc),
                        },
                    )
                    text = _placeholder_text(image)
            else:
                text = _placeholder_text(image)

            cid = chunk_id(paper_id, ordinal)
            chunks.append(
                VisionInsightChunk(
                    id=cid,
                    paper_id=paper_id,
                    text=text,
                    page=image.page,
                    bbox=image.bbox,
                    image_path=image.image_path,
                    source_ref=image.source_ref
                    or SourceRef(
                        section_label=image.figure_label or "Vision",
                        page_label=f"page {image.page}",
                        page=image.page,
                    ),
                    ordinal=ordinal,
                    token_count=max(1, len(text) // 4),
                    figure_label=image.figure_label,
                    caption=image.caption,
                )
            )
            ordinal += 1

        return chunks
