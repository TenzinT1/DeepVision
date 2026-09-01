"""OCR processor — stage 4 (OCR).

Runs pytesseract over raster regions (tables/scanned pages) to recover text as
:class:`OCRChunk` objects with confidence.
"""

from __future__ import annotations

import abc

from deepvision.ingestion.paths import to_abspath
from deepvision.models.chunks import ImageChunk, OCRChunk, SourceRef
from deepvision.models.settings import OCRLanguage
from deepvision.utils import get_logger
from deepvision.utils.ids import chunk_id

__all__ = ["OCRProcessor", "OCRProcessorService"]

log = get_logger(__name__)


class OCRProcessor(abc.ABC):
    """Recovers text from raster images via OCR."""

    @abc.abstractmethod
    def process(
        self,
        images: list[ImageChunk],
        paper_id: str,
        *,
        language: OCRLanguage = OCRLanguage.ENGLISH,
    ) -> list[OCRChunk]:
        """OCR each image region; return :class:`OCRChunk`s with confidence."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# Concrete implementation
# --------------------------------------------------------------------------

#: Ordinal namespace for OCR chunks — kept disjoint from text/image/vision.
OCR_ORDINAL_BASE = 200_000

_MIN_TEXT_LEN = 5

# tesseract has no true "auto" language mode; map it to English, which is a
# safe, always-installed default (see AppSettings.ocr_language docstring).
_LANG_MAP = {
    OCRLanguage.ENGLISH: "eng",
    OCRLanguage.ENGLISH_FRENCH: "eng+fra",
    OCRLanguage.CHINESE_SIMPLIFIED: "chi_sim",
    OCRLanguage.JAPANESE: "jpn",
    OCRLanguage.AUTO: "eng",
}


class OCRProcessorService(OCRProcessor):
    """pytesseract-based OCR over figure/table crops.

    Degrades gracefully (logs and returns an empty list) whenever Pillow or
    pytesseract aren't installed, or the ``tesseract`` binary itself is
    missing — OCR is an enhancement stage, never a pipeline blocker.
    """

    def process(
        self,
        images: list[ImageChunk],
        paper_id: str,
        *,
        language: OCRLanguage = OCRLanguage.ENGLISH,
    ) -> list[OCRChunk]:
        if not images:
            return []

        try:
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
        except ImportError as exc:
            log.warning(
                "OCR skipped: Pillow/pytesseract not installed",
                extra={"paper_id": paper_id, "error": str(exc)},
            )
            return []

        lang = _LANG_MAP.get(language, "eng")
        chunks: list[OCRChunk] = []
        ordinal = OCR_ORDINAL_BASE
        tesseract_missing = False

        for image in images:
            if tesseract_missing:
                break
            if not image.image_path:
                continue
            abs_path = to_abspath(image.image_path)
            if not abs_path.exists():
                continue
            try:
                with Image.open(abs_path) as pil_image:
                    data = pytesseract.image_to_data(
                        pil_image, lang=lang, output_type=pytesseract.Output.DICT
                    )
            except pytesseract.pytesseract.TesseractNotFoundError as exc:
                log.warning(
                    "OCR skipped: tesseract binary not found",
                    extra={"paper_id": paper_id, "error": str(exc)},
                )
                tesseract_missing = True
                break
            except Exception as exc:  # pragma: no cover - defensive
                log.warning(
                    "OCR failed for image, skipping",
                    extra={
                        "paper_id": paper_id,
                        "image_path": image.image_path,
                        "error": str(exc),
                    },
                )
                continue

            words: list[str] = []
            confidences: list[float] = []
            for word, conf in zip(data.get("text", []), data.get("conf", [])):
                word = word.strip()
                if not word:
                    continue
                try:
                    conf_val = float(conf)
                except (TypeError, ValueError):
                    continue
                if conf_val < 0:
                    continue
                words.append(word)
                confidences.append(conf_val)

            text = " ".join(words).strip()
            if len(text) < _MIN_TEXT_LEN:
                continue
            mean_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else None

            cid = chunk_id(paper_id, ordinal)
            chunks.append(
                OCRChunk(
                    id=cid,
                    paper_id=paper_id,
                    text=text,
                    page=image.page,
                    bbox=image.bbox,
                    image_path=image.image_path,
                    source_ref=image.source_ref
                    or SourceRef(
                        section_label=image.figure_label or "OCR",
                        page_label=f"page {image.page}",
                        page=image.page,
                    ),
                    ordinal=ordinal,
                    token_count=max(1, len(text) // 4),
                    ocr_confidence=mean_conf,
                    language=lang,
                )
            )
            ordinal += 1

        return chunks
