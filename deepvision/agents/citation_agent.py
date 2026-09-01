"""Citation agent — resolves inline claims to source citations.

Given section/answer text and the paper's chunks, produces resolvable
:class:`Citation`s (source label, page, page image, exact snippet, bbox) and
wires the inline ``[n]`` markers.
Purely deterministic (no LLM involved) — grounding a marker in an exact chunk
snippet is a structural operation, not a generative one, so this agent works
identically regardless of which LLM provider is configured.
"""

from __future__ import annotations

import re
from typing import Sequence

from deepvision.agents.base import clean_snippet
from deepvision.models.chunks import AnyChunk
from deepvision.models.report import Citation
from deepvision.utils.ids import citation_id

__all__ = ["CitationAgent"]

_MARKER_RE = re.compile(r"\[(\d+)\]")


class CitationAgent:
    """Grounds textual claims in exact source snippets."""

    def cite(self, text: str, chunks: Sequence[AnyChunk]) -> list[Citation]:
        """Return ordered :class:`Citation`s for the ``[n]`` markers in ``text``.

        Marker ``n`` resolves to the ``n``-th chunk (1-indexed) in ``chunks`` —
        the same ordered list the caller used when drafting ``text`` with those
        markers. If ``text`` carries no markers at all (e.g. a degraded LLM
        response that dropped them), the first few ``chunks`` are cited anyway
        so the section is never left completely ungrounded. Out-of-range marker
        numbers wrap around the available chunks rather than being dropped, so a
        slightly-off marker from an LLM rewrite still resolves to *something*
        real instead of vanishing.
        """
        chunk_list = list(chunks)
        if not chunk_list:
            return []

        markers = sorted({int(m) for m in _MARKER_RE.findall(text or "")})
        if not markers:
            markers = list(range(1, min(len(chunk_list), 5) + 1))

        citations: list[Citation] = []
        for marker in markers:
            idx = marker - 1
            if idx < 0:
                continue
            chunk = chunk_list[idx] if idx < len(chunk_list) else chunk_list[idx % len(chunk_list)]
            citations.append(self._citation_from_chunk(chunk, marker))
        return citations

    @staticmethod
    def _citation_from_chunk(chunk: AnyChunk, marker: int) -> Citation:
        ref = chunk.source_ref
        source = ref.section_label if ref else f"Page {chunk.page}"
        page_label = ref.page_label if ref else f"page {chunk.page}"
        # Rendered page images follow the fixed pipeline convention
        # (`PageRenderer.render_all` / `render_page`): `<paper_id>/pages/p{n}.png`.
        page_image_path = f"{chunk.paper_id}/pages/p{chunk.page}.png"
        snippet = clean_snippet(chunk.text, max_chars=280) or "(no text available for this source)"
        return Citation(
            id=citation_id(),
            marker=marker,
            source=source,
            page=chunk.page,
            page_label=page_label,
            page_image_path=page_image_path,
            snippet=snippet,
            bbox=chunk.bbox,
            chunk_id=chunk.id,
        )
