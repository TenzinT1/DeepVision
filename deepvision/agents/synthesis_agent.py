"""Synthesis agent — final report assembly.

Assembles the final report (:meth:`SynthesisAgent.assemble_report`,
the "produce the final coherent Report" half of this domain's job — ordering
sections into ``SECTION_ORDER``, and computing stats/model metadata — which
:class:`~deepvision.agents.agent_orchestrator.AgentOrchestrator` calls once the
summarizer/media agents have produced the sections.
"""

from __future__ import annotations

from typing import Optional, Sequence

from deepvision.agents.base import clean_snippet, complete_with_fallback, load_paper_meta
from deepvision.models.chunks import AnyChunk
from deepvision.models.paper import PaperMeta
from deepvision.models.report import SECTION_ORDER, Report, ReportStats, Section, SectionName
from deepvision.models.settings import AppSettings
from deepvision.providers.base import LLMProvider
from deepvision.rag.retrieval import Retriever
from deepvision.utils.ids import report_id, section_id
from deepvision.utils.logger import get_logger

__all__ = ["SynthesisAgent"]

log = get_logger(__name__)

_K_PER_DIMENSION = 3


class SynthesisAgent:
    """Assembles the final report from the other agents' sections."""

    def __init__(self, llm: LLMProvider, retriever: Optional[Retriever] = None) -> None:
        self.llm = llm
        self.retriever = retriever

    # ---- cross-paper comparison ------------------------------------------

    def assemble_report(
        self,
        *,
        paper_id: str,
        settings: AppSettings,
        sections: Sequence[Section],
        paper: Optional[PaperMeta] = None,
    ) -> Report:
        """Assemble the final :class:`Report`: order sections, compute stats.

        ``sections`` need not already be in :data:`SECTION_ORDER` or cover every
        name — any missing section is filled with an empty placeholder so the
        returned report always has exactly one section per
        :data:`SECTION_ORDER` entry, in that fixed order.
        """
        by_name = {s.name: s for s in sections}
        ordered: list[Section] = []
        for name in SECTION_ORDER:
            section = by_name.get(name)
            if section is None:
                section = Section(
                    id=section_id(),
                    name=name,
                    body_markdown="",
                    provenance=[],
                    default_open=(name is SectionName.OVERVIEW),
                )
            ordered.append(section)

        stats = self._compute_stats(ordered, paper)
        return Report(
            id=report_id(),
            paper_id=paper_id,
            paper=paper,
            stats=stats,
            sections=ordered,
            model_used=getattr(self.llm, "model", None),
        )

    @staticmethod
    def _compute_stats(sections: Sequence[Section], paper: Optional[PaperMeta]) -> ReportStats:
        figures_section = next((s for s in sections if s.name is SectionName.FIGURES), None)
        figures = len(figures_section.media) if figures_section else 0
        citations_extracted = sum(len(s.citations) for s in sections)
        word_count = 0
        for s in sections:
            word_count += len((s.body_markdown or "").split())
            word_count += len((s.deep_dive_markdown or "").split())
        reading_time_min = max(1, round(word_count / 200)) if word_count else 0
        pages = paper.page_count if paper and paper.page_count else 0
        if not pages:
            # Fall back to the highest page number seen in citations/media when
            # paper metadata (or its page_count) isn't available yet.
            seen_pages = [c.page for s in sections for c in s.citations if c.page]
            seen_pages += [m.page for s in sections for m in s.media if m.page]
            pages = max(seen_pages) if seen_pages else 0
        return ReportStats(
            pages=pages or 0,
            figures=figures,
            citations_extracted=citations_extracted,
            reading_time_min=reading_time_min,
        )
