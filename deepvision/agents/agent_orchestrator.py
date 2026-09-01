"""Agent orchestrator — coordinates agents into a full Report.

Runs the fixed agent sequence Research -> Summarizer -> Media -> Study ->
Synthesis (citation grounding is embedded in the Research/Summarizer/Study
steps — see the module docstrings on those agents) and assembles the complete
:class:`Report`. The four stages between Research and Synthesis each own a
disjoint slice of :data:`SECTION_ORDER`: Summarizer writes the nine prose
sections (At a Glance, Overview, Background, Key Concepts, Methods, Key
Results, Limitations & Open Questions, Why It Matters, Key Takeaways), Media
writes Figures, Study writes Study Questions only (the former Glossary was
absorbed into Key Concepts). for
``POST /compare``.. The report layer persists/serves the
result.

Every stage is retried and, on exhausted retries, replaced with a deterministic
degraded fallback rather than raising — ingestion must never crash because a
model backend hiccuped. Providers are always built fresh per call from the
*effective* settings via the factory
(``build_llm(settings, strict=get_config().strict_providers)``), per §5 of
prefers the real local/API adapter and lets ``EchoLLM``-equivalent degradation
happen at the individual LLM-call site (see
``agents.base.complete_with_fallback``) rather than at construction time — but
it must be *read*, not hardcoded: ``DEEPVISION_STRICT_PROVIDERS=0`` is the
app-wide "no models installed / stay offline" switch (that is what
``scripts/smoke_check.py`` runs under), and hardcoding ``strict=True`` here
made report generation dial a real Ollama anyway and block for minutes on
socket timeouts. Every other call site already reads the config; this one now
matches.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, TypeVar

from deepvision.agents.base import AgentContext, load_paper_meta
from deepvision.agents.citation_agent import CitationAgent
from deepvision.agents.media_agent import MediaAgent
from deepvision.agents.research_agent import ResearchAgent, flatten_outline
from deepvision.agents.study_agent import StudyAgent
from deepvision.agents.summarizer_agent import SummarizerAgent
from deepvision.agents.synthesis_agent import SynthesisAgent
from deepvision.config import get_config
from deepvision.models.report import (
    SECTION_ORDER,
    SECTION_QUESTIONS,
    Report,
    ReportStats,
    Section,
    SectionName,
)
from deepvision.models.settings import AppSettings
from deepvision.providers.factory import build_llm
from deepvision.rag.retrieval import Retriever
from deepvision.utils.ids import report_id, section_id
from deepvision.utils.logger import get_logger

__all__ = ["AgentOrchestrator"]

log = get_logger(__name__)

T = TypeVar("T")

#: The nine prose sections SummarizerAgent owns, in generation order — mirrors
#: ``summarizer_agent._NARRATIVE_ORDER``; used only by the degraded fallback.
_NARRATIVE_NAMES: tuple[SectionName, ...] = (
    SectionName.AT_A_GLANCE,
    SectionName.OVERVIEW,
    SectionName.BACKGROUND,
    SectionName.KEY_CONCEPTS,
    SectionName.METHODS,
    SectionName.KEY_RESULTS,
    SectionName.LIMITATIONS,
    SectionName.WHY_IT_MATTERS,
    SectionName.KEY_TAKEAWAYS,
)

#: Sections that start expanded. ``report.interactive_sections`` re-applies this
#: authoritatively on normalize; duplicated here so the agents layer stays free
#: of a report-layer import. Mirrors ``summarizer_agent._DEFAULT_OPEN``.
_DEFAULT_OPEN: tuple[SectionName, ...] = (
    SectionName.AT_A_GLANCE,
    SectionName.OVERVIEW,
    SectionName.BACKGROUND,
)


def _chunk_in_range(chunk, page_range: tuple[int, int]) -> bool:
    """True if ``chunk``'s page lies inside the inclusive ``page_range``.

    A chunk with no usable page is treated as out of range — a scoped report
    must not be grounded in material whose location can't be verified.
    """
    try:
        page = int(getattr(chunk, "page", None))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return page_range[0] <= page <= page_range[1]


def _scope_outline(
    outline: dict[SectionName, list], page_range: tuple[int, int]
) -> dict[SectionName, list]:
    """Drop every out-of-range chunk from a planned outline.

    Retrieval is already scoped by
    :class:`~deepvision.rag.chapter_scope.PageScopedRetriever`; this is the
    second, independent guarantee — including for the Figures group, whose
    media chunks are merged in from *all* section queries and so must be
    re-checked before any figure card is built.
    """
    return {
        name: [c for c in chunks if _chunk_in_range(c, page_range)]
        for name, chunks in (outline or {}).items()
    }


def _thin_scope_note(name: SectionName, page_range: tuple[int, int]) -> str:
    """Honest placeholder body for a section with no in-chapter material."""
    return (
        f"There isn't enough material on pages {page_range[0]}-{page_range[1]} "
        f"to write **{name.value}** for this chapter. Nothing outside the "
        "chapter's pages was used, so this section is intentionally empty "
        "rather than filled with content from elsewhere in the paper."
    )


class AgentOrchestrator:
    """Runs the agent ensemble to produce a paper's report (and comparisons).

    Requires a :class:`~deepvision.rag.retrieval.Retriever` — the concrete
    implementation lives in the rag layer and is
    injected here rather than constructed internally, since this module must
    not import a Domain-B-owned concrete class name. Whoever wires the app
    (the report/chat route bodies in the report layer) constructs this
    with the real retriever, e.g.::

        orchestrator = AgentOrchestrator(retriever=my_concrete_retriever)
        report = orchestrator.build_report(paper_id, settings)
    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        citation_agent: Optional[CitationAgent] = None,
        media_agent: Optional[MediaAgent] = None,
        retries: int = 2,
    ) -> None:
        self.retriever = retriever
        self.citation_agent = citation_agent or CitationAgent()
        self.media_agent = media_agent or MediaAgent()
        self.retries = max(1, retries)

    def build_report(
        self,
        paper_id: str,
        settings: AppSettings,
        *,
        page_range: Optional[tuple[int, int]] = None,
    ) -> Report:
        """Assemble the full :class:`Report` for ``paper_id`` from its chunks.

        ``page_range`` is the optional, explicit chapter scope: an inclusive
        ``(page_start, page_end)`` window. It is **additive** — the default
        ``None`` runs the whole-paper pipeline exactly as before — and it is
        passed in, never read from global state or from ``settings``.

        The scoped path is the *same* pipeline, same agents, same
        :meth:`_stage` retry + degraded-fallback behaviour. The caller is
        expected to also hand this orchestrator a
        :class:`~deepvision.rag.chapter_scope.PageScopedRetriever` (that is what
        keeps retrieval itself inside the chapter); the range given here is the
        independent second guarantee applied to whatever the outline came back
        with, and it is what makes the Figures section chapter-only.
        """
        # Local import: `deepvision.report` (the package `__init__`) imports
        # `report_generator`, which imports THIS module (`AgentOrchestrator`)
        # at module scope — a top-level import here would be a circular
        # import at package-load time. Deferring it to call time is safe
        # because by the time `build_report` actually runs, both modules are
        # already fully imported.
        from deepvision.report.figure_links import attach_referenced_figures

        llm = build_llm(settings, strict=get_config().strict_providers)
        research_agent = ResearchAgent(
            llm, citation_agent=self.citation_agent, media_agent=self.media_agent
        )
        summarizer_agent = SummarizerAgent(llm, citation_agent=self.citation_agent)
        study_agent = StudyAgent(llm, citation_agent=self.citation_agent)
        synthesis_agent = SynthesisAgent(llm, retriever=self.retriever)

        outline = self._stage(
            "research",
            lambda: research_agent.plan_outline(paper_id, settings, self.retriever),
            default={},
        )
        if page_range is not None:
            outline = _scope_outline(outline, page_range)
        all_chunks = flatten_outline(outline)

        if page_range is not None and not all_chunks:
            log.warning(
                "no in-scope chunks for chapter page range; emitting an honest "
                "empty report rather than widening the scope",
                extra={
                    "paper_id": paper_id,
                    "page_start": page_range[0],
                    "page_end": page_range[1],
                },
            )
            return self._empty_scope_report(paper_id, page_range, llm)

        # `page_range` rides along in `extra` so agents that would otherwise
        # reach for *paper-level* material (the summarizer seeds the Overview
        # from the paper's abstract, which lives on page 1) can tell they are
        # writing about one chapter. Absent for the whole-paper path, so nothing
        # there changes.
        agent_extra: dict = {"outline": outline}
        if page_range is not None:
            agent_extra["page_range"] = page_range

        narrative_sections = self._stage(
            "summarizer",
            lambda: summarizer_agent.run(
                AgentContext(
                    paper_id=paper_id,
                    settings=settings,
                    chunks=all_chunks,
                    extra=dict(agent_extra),
                )
            ),
            default=None,
        )
        if narrative_sections is None:
            narrative_sections = self._degraded_sections(outline)

        figures_chunks = outline.get(SectionName.FIGURES, [])
        figures_section = self._stage(
            "media",
            lambda: self.media_agent.build_figures_section(figures_chunks),
            default=None,
        )
        if figures_section is None:
            figures_section = Section(
                id=section_id(), name=SectionName.FIGURES, body_markdown="",
                provenance=[], media=[], default_open=False,
            )

        study_sections = self._stage(
            "study",
            lambda: study_agent.run(
                AgentContext(
                    paper_id=paper_id,
                    settings=settings,
                    chunks=all_chunks,
                    extra=dict(agent_extra),
                )
            ),
            default=None,
        )
        if study_sections is None:
            study_sections = self._degraded_study_sections(outline)

        sections = list(narrative_sections) + [figures_section] + list(study_sections)

        # Deterministic figure auto-linking: this is the one seam where the
        # full media pool for the paper (`figures_section.media`, built by
        # MediaAgent above) and the complete, final set of prose sections
        # both exist at once, and it runs before the synthesis stage because
        # SynthesisAgent only reorders/repairs `sections` — it does not add
        # media, so running this after synthesis instead would just mean
        # threading the pool through an extra layer for no benefit. Chapter
        # reports call this same `build_report` (scoped via `page_range` /
        # `PageScopedRetriever`), so scoped figure linking falls out of this
        # one call site for free — no second call site needed.
        sections = attach_referenced_figures(sections, figures_section.media)

        paper_meta = load_paper_meta(paper_id)

        report = self._stage(
            "synthesis",
            lambda: synthesis_agent.assemble_report(
                paper_id=paper_id, settings=settings, sections=sections, paper=paper_meta
            ),
            default=None,
        )
        if report is None:
            report = self._minimal_report(
                paper_id, sections, paper_meta, llm, page_range=page_range
            )
        if page_range is not None:
            report.sections = self._honest_scoped_sections(report.sections, page_range)
        return report

    def _stage(self, name: str, fn: Callable[[], T], *, default: T) -> T:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.retries + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - top-level pipeline guard
                last_exc = exc
                log.warning(
                    "agent stage failed",
                    extra={"stage": name, "attempt": attempt, "error": str(exc)},
                )
        log.error(
            "agent stage exhausted retries; using degraded fallback",
            extra={"stage": name, "error": str(last_exc) if last_exc else ""},
        )
        return default

    def _degraded_sections(self, outline: dict[SectionName, list]) -> list[Section]:
        """Minimal, deterministic sections used only if the summarizer stage fails entirely."""
        return [
            self._degraded_section(name, outline, badge="SUMMARY")
            for name in _NARRATIVE_NAMES
        ]

    def _degraded_study_sections(self, outline: dict[SectionName, list]) -> list[Section]:
        """Deterministic Study Questions used only if StudyAgent fails entirely.

        StudyAgent already degrades internally (its drafts are what ship when
        the LLM is unavailable), so reaching this means the stage raised — keep
        the section present and grounded rather than dropping it. StudyAgent no
        longer writes a Glossary (absorbed into Key Concepts), so this must not
        produce one either.
        """
        return [
            self._degraded_section(
                SectionName.STUDY_QUESTIONS, outline, badge="SELF-CHECK"
            ),
        ]

    def _degraded_section(
        self, name: SectionName, outline: dict[SectionName, list], *, badge: str
    ) -> Section:
        chunks = list((outline or {}).get(name, []))[:4]
        body = " ".join(
            f"{(c.text or '').strip()[:200]} [{i}]"
            for i, c in enumerate(chunks, start=1)
            if (c.text or "").strip()
        ) or f"No grounded content is available yet for **{name.value}**."
        citations = self.citation_agent.cite(body, chunks) if chunks else []
        return Section(
            id=section_id(), name=name, question=SECTION_QUESTIONS.get(name),
            body_markdown=body, badge=badge,
            provenance=[], citations=citations, media=[], confidence=0.2,
            degraded=True,
            default_open=(name in _DEFAULT_OPEN),
        )

    def _honest_scoped_sections(
        self, sections: Sequence[Section], page_range: tuple[int, int]
    ) -> list[Section]:
        """Replace empty bodies in a scoped report with an honest explanation.

        A chapter is a small slice of a paper, so some of the eleven sections
        genuinely have nothing to say (a Methods chapter has no Conclusions).
        Saying so is the contract; shipping an empty string — or quietly
        borrowing text from outside the chapter — is not.
        """
        out: list[Section] = []
        for sec in sections:
            if not (sec.body_markdown or "").strip() and not sec.media:
                sec.body_markdown = _thin_scope_note(sec.name, page_range)
                sec.confidence = sec.confidence if sec.confidence is not None else 0.0
            out.append(sec)
        return out

    def _empty_scope_report(
        self, paper_id: str, page_range: tuple[int, int], llm
    ) -> Report:
        """A valid eleven-section Report for a chapter that yielded no chunks.

        Reached when page-filtered retrieval returned nothing at all (a cover
        page, a chapter of pure figures with no embedded text, or a paper whose
        vectors were never built). Returning a valid report with an honest body
        per section is required behaviour — crashing, or returning empty
        strings, is not.
        """
        paper_meta = load_paper_meta(paper_id)
        sections = [
            Section(
                id=section_id(),
                name=name,
                question=SECTION_QUESTIONS.get(name),
                body_markdown=_thin_scope_note(name, page_range),
                provenance=[],
                media=[],
                confidence=0.0,
                default_open=(name in _DEFAULT_OPEN),
            )
            for name in SECTION_ORDER
        ]
        return Report(
            id=report_id(),
            paper_id=paper_id,
            paper=paper_meta,
            stats=ReportStats(
                pages=max(0, page_range[1] - page_range[0] + 1),
                figures=0,
                citations_extracted=0,
                reading_time_min=0,
            ),
            sections=sections,
            model_used=getattr(llm, "model", None),
        )

    def _minimal_report(
        self,
        paper_id: str,
        sections: Sequence[Section],
        paper_meta,
        llm,
        *,
        page_range: Optional[tuple[int, int]] = None,
    ) -> Report:
        by_name = {s.name: s for s in sections}
        ordered = [
            by_name.get(name)
            or Section(
                id=section_id(),
                name=name,
                question=SECTION_QUESTIONS.get(name),
                body_markdown="",
                provenance=[],
            )
            for name in SECTION_ORDER
        ]
        figures = next((s for s in ordered if s.name is SectionName.FIGURES), None)
        # For a chapter report `pages` is the chapter's own span, never the
        # document's (the report layer recomputes this, but the fallback path
        # must not publish the wrong number either).
        if page_range is not None:
            pages = max(0, page_range[1] - page_range[0] + 1)
        else:
            pages = (paper_meta.page_count if paper_meta and paper_meta.page_count else 0) or 0
        stats = ReportStats(
            pages=pages,
            figures=len(figures.media) if figures else 0,
            citations_extracted=sum(len(s.citations) for s in ordered),
            reading_time_min=0,
        )
        return Report(
            id=report_id(), paper_id=paper_id, paper=paper_meta, stats=stats,
            sections=ordered, model_used=getattr(llm, "model", None),
        )

