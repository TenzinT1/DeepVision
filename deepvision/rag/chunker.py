"""Chunker — split extracted content into embedding-sized chunks.

Applies the settings' ``chunk_size`` / ``chunk_overlap`` to normalize text,
OCR, and vision content into retrievable chunks.
"""

from __future__ import annotations

import abc
import re
from itertools import groupby
from typing import Optional, Sequence

from deepvision.models.chunks import (
    AnyChunk,
    BBox,
    Chunk,
    ImageChunk,
    OCRChunk,
    TextChunk,
    VisionInsightChunk,
)
from deepvision.models.settings import AppSettings
from deepvision.utils.ids import chunk_id
from deepvision.utils.logger import get_logger

__all__ = ["Chunker", "SlidingWindowChunker", "count_tokens"]

log = get_logger(__name__)

# A dependency-free, whitespace-based token approximation. ``tiktoken`` is not
# part of requirements.txt, so we approximate "tokens" as whitespace-delimited
# words; this keeps chunk_size/chunk_overlap meaningful (and swappable for a
# real BPE tokenizer later) without adding a hard dependency.
_WORD_RE = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    """Approximate token count of ``text`` (whitespace-delimited words)."""
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping windows of ~``size`` tokens each."""
    tokens = _tokenize(text)
    if not tokens:
        return [""]
    if len(tokens) <= size:
        return [text]
    step = max(size - overlap, 1)
    parts: list[str] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + size, n)
        parts.append(" ".join(tokens[start:end]))
        if end >= n:
            break
        start += step
    return parts


def _union_bbox(bboxes: Sequence[Optional[BBox]]) -> Optional[BBox]:
    boxes = [b for b in bboxes if b is not None]
    if not boxes:
        return None
    return BBox(
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )


class Chunker(abc.ABC):
    """Re-chunks raw extracted content to the configured token window."""

    @abc.abstractmethod
    def chunk(
        self, raw_chunks: Sequence[AnyChunk], settings: AppSettings
    ) -> list[AnyChunk]:
        """Return chunks sized per ``settings.chunk_size`` / ``chunk_overlap``.

        Deterministic ids should come from :func:`deepvision.utils.ids.chunk_id`.
        """
        raise NotImplementedError


class SlidingWindowChunker(Chunker):
    """Token-aware chunker over the unified multimodal extraction stream.

    Input is a heterogeneous, unordered sequence of ``TextChunk`` / ``OCRChunk``
    / ``VisionInsightChunk`` / ``ImageChunk`` produced by the ingestion stages
    (the ingestion layer). This chunker:

    1. Orders content by ``(page, ordinal)``.
    2. Greedily merges *adjacent* chunks that share the same concrete type,
       page, and provenance into a single buffer (up to ``chunk_size``
       tokens), so short paragraphs/OCR lines aren't embedded one-at-a-time.
    3. Splits any resulting text that still exceeds ``chunk_size`` tokens into
       overlapping windows (``chunk_overlap`` tokens of overlap).
    4. Leaves ``ImageChunk`` entries as atomic, unsplit units — their
       "text" for embedding purposes is their caption (see
       :mod:`deepvision.rag.embedding_pipeline`), but the crop itself is never
       token-windowed.

    Provenance, page, and a best-effort bounding box (union of merged pieces)
    are preserved on every output chunk. Output chunk ids are assigned via
    :func:`deepvision.utils.ids.chunk_id` in final emission order so a rerun
    with the same content deterministically overwrites the same ids.
    """

    def chunk(
        self, raw_chunks: Sequence[AnyChunk], settings: AppSettings
    ) -> list[AnyChunk]:
        if not raw_chunks:
            return []

        size = settings.chunk_size
        overlap = min(settings.chunk_overlap, max(size - 1, 0))

        output: list[AnyChunk] = []
        # Group defensively by paper_id (chunk() is expected to be called with
        # a single paper's raw chunks per the ingestion pipeline, but this
        # keeps the method safe/correct if ever called with a mixed batch).
        by_paper = sorted(raw_chunks, key=lambda c: c.paper_id)
        for paper_id, paper_group in groupby(by_paper, key=lambda c: c.paper_id):
            ordered = sorted(paper_group, key=lambda c: (c.page, c.ordinal))
            groups = self._merge_adjacent(ordered, size)
            counter = 0
            for group in groups:
                for piece in self._finalize_group(group, size, overlap):
                    piece.id = chunk_id(paper_id, counter)
                    piece.ordinal = counter
                    counter += 1
                    output.append(piece)
            log.debug(
                "chunked paper",
                paper_id=paper_id,
                raw_count=len(ordered),
                out_count=counter,
            )
        return output

    # -- internals ---------------------------------------------------------

    def _merge_adjacent(
        self, ordered: Sequence[Chunk], size: int
    ) -> list[list[Chunk]]:
        groups: list[list[Chunk]] = []
        buffer: list[Chunk] = []
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if buffer:
                groups.append(buffer)
            buffer = []
            buffer_tokens = 0

        for ch in ordered:
            if isinstance(ch, ImageChunk) or not isinstance(ch, TextChunk) or ch.image_path:
                # Crop-bound content (ImageChunk, OCRChunk, VisionInsightChunk,
                # or anything else carrying its own image_path) is atomic: it
                # must stay a 1:1 unit with its own label/caption/bbox/image
                # rather than being merged with a sibling that happens to
                # share page/type/provenance (e.g. two figures on one page).
                # Only free-flowing TextChunk content is eligible for merging.
                flush()
                groups.append([ch])
                continue
            tok = count_tokens(ch.text)
            same_bucket = (
                buffer
                and type(buffer[-1]) is type(ch)
                and buffer[-1].page == ch.page
                and buffer[-1].provenance == ch.provenance
            )
            if same_bucket and buffer_tokens + tok <= size:
                buffer.append(ch)
                buffer_tokens += tok
            else:
                flush()
                buffer.append(ch)
                buffer_tokens = tok
        flush()
        return groups

    def _finalize_group(
        self, group: list[Chunk], size: int, overlap: int
    ) -> list[AnyChunk]:
        if len(group) == 1 and isinstance(group[0], ImageChunk):
            return [group[0].model_copy(deep=True)]

        cls = type(group[0])
        page = group[0].page
        bbox = _union_bbox([g.bbox for g in group])
        image_path = next((g.image_path for g in group if g.image_path), None)
        source_ref = group[0].source_ref
        paper_id = group[0].paper_id
        text = "\n\n".join(g.text for g in group if g.text)

        extra = self._subtype_extra(cls, group)
        pieces = _split_text(text, size, overlap)

        results: list[AnyChunk] = []
        for piece_text in pieces:
            kwargs: dict = dict(
                id="",
                paper_id=paper_id,
                page=page,
                bbox=bbox,
                image_path=image_path,
                source_ref=source_ref,
                text=piece_text,
                token_count=count_tokens(piece_text),
                ordinal=0,
            )
            kwargs.update(extra)
            results.append(cls(**kwargs))
        return results

    @staticmethod
    def _subtype_extra(cls: type, group: list[Chunk]) -> dict:
        if cls is OCRChunk:
            confs = [
                g.ocr_confidence
                for g in group
                if isinstance(g, OCRChunk) and g.ocr_confidence is not None
            ]
            lang = next(
                (g.language for g in group if isinstance(g, OCRChunk) and g.language),
                None,
            )
            return {
                "ocr_confidence": (sum(confs) / len(confs)) if confs else None,
                "language": lang,
            }
        if cls is VisionInsightChunk:
            label = next(
                (
                    g.figure_label
                    for g in group
                    if isinstance(g, VisionInsightChunk) and g.figure_label
                ),
                None,
            )
            caption = next(
                (
                    g.caption
                    for g in group
                    if isinstance(g, VisionInsightChunk) and g.caption
                ),
                None,
            )
            return {"figure_label": label, "caption": caption}
        if cls is TextChunk:
            return {}
        return {}
