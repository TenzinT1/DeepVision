"""Research agent — answers grounded chat questions over one paper, and plans
the report outline used by the summarizer/media agents.

Consumes retrieved chunks and a question, returns a grounded assistant
:class:`ChatMessage` (text + citations + figure refs).
Beyond the fixed :meth:`run` (chat) entry point, this module also implements
:meth:`ResearchAgent.plan_outline` — the "gather key retrieved chunks per
section" half of the report pipeline. It issues one targeted retrieval query
per fixed report section (every name in
:data:`deepvision.models.report.SECTION_ORDER`, all eleven of them) against the
:class:`~deepvision.rag.retrieval.Retriever` and returns the per-section chunk
groups the summarizer / media / study agents write from.

Retrieval breadth is the first of the two places ``report_detail`` has any
effect (the second is ``summarizer_agent._DETAIL_PARAMS``): see
:data:`_K_BY_DETAIL` and :data:`_K_FIGURES_BY_DETAIL`.

Every non-Figures section's hits also pass through
:func:`deepvision.rag.chunk_quality.narrative_chunks` — the generic defence
against retrieval surfacing figure captions and running-header/back-matter
junk as if it were paper prose (see that module's docstring for the real
failures this prevents). The Figures section keeps its raw hits, including
captions, since that is exactly what it needs.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from deepvision.agents.base import (
    Agent,
    AgentContext,
    clean_snippet,
    load_paper_meta,
    repair_citation_markers,
)
from deepvision.agents.chat_intent import Intent, classify
from deepvision.agents.citation_agent import CitationAgent
from deepvision.agents.media_agent import MediaAgent
from deepvision.models.chat import AnswerKind, ChatMessage, ChatRole
from deepvision.models.chunks import AnyChunk, Modality
from deepvision.models.paper import PaperMeta
from deepvision.models.report import SECTION_ORDER, MediaRef, SectionName
from deepvision.models.settings import AppSettings, ReportDetail
from deepvision.providers.base import LLMProvider
from deepvision.rag.chunk_quality import (
    diversify,
    narrative_chunks,
    strip_repeated_prefixes,
)
from deepvision.rag.retrieval import Retriever
from deepvision.utils.ids import message_id
from deepvision.utils.logger import get_logger

__all__ = ["ResearchAgent", "flatten_outline"]

log = get_logger(__name__)

#: One retrieval query per fixed section, tuned to surface the passages a human
#: skimming the paper would pull for that heading. Every name in
#: :data:`SECTION_ORDER` (all eleven) must have an entry — ``plan_outline``
#: iterates the full order, and a section with no query gets an empty chunk
#: list (and therefore an empty section in the report) plus a logged warning.
_SECTION_QUERIES: dict[SectionName, str] = {
    SectionName.AT_A_GLANCE: (
        "problem statement what is proposed what this paper is about what was "
        "evaluated on headline result main contribution in one sentence"
    ),
    SectionName.OVERVIEW: (
        "paper overview abstract summary motivation contribution problem statement"
    ),
    SectionName.BACKGROUND: (
        "background related work prior work previous approaches existing methods "
        "motivation why this is hard limitations of earlier work context history"
    ),
    SectionName.KEY_CONCEPTS: (
        "we define definition preliminaries notation terminology concept intuition "
        "refers to denotes is defined as formulation basic idea building block "
        "term acronym abbreviation stands for also known as we denote we call this"
    ),
    SectionName.METHODS: (
        "method approach model architecture algorithm framework implementation"
    ),
    SectionName.KEY_RESULTS: (
        "results experiments evaluation performance benchmark accuracy findings"
    ),
    SectionName.FIGURES: (
        "figure table chart diagram plot visualization illustration"
    ),
    SectionName.LIMITATIONS: (
        "limitations drawbacks weaknesses fails when does not work assumption "
        "threats to validity future work remains open not addressed untested"
    ),
    SectionName.WHY_IT_MATTERS: (
        "implications significance impact applications enables in practice "
        "broader why this matters what changes as a result who benefits"
    ),
    SectionName.KEY_TAKEAWAYS: (
        "key takeaway main finding most important result central claim what to "
        "remember bottom line significance impact why it matters we show we find"
    ),
    SectionName.STUDY_QUESTIONS: (
        "main contribution key idea why it works central claim evidence result "
        "limitation implication what this means takeaway"
    ),
}

_MEDIA_MODALITIES = (Modality.IMAGE, Modality.VISION, Modality.OCR)

#: How many chunks to retrieve per narrative section, by report-detail level —
#: more detail pulls more grounded material for the summarizer to write from.
#: Spread ~2x per step so ``concise`` and ``standard`` really are different
#: reports (see ``summarizer_agent._DETAIL_PARAMS``, the other half of this).
_K_BY_DETAIL = {
    ReportDetail.CONCISE: 5,
    ReportDetail.STANDARD: 10,
    ReportDetail.DETAILED: 20,
}

#: Same idea for the Figures section, which is retrieved separately and much
#: more broadly (media chunks are filtered hard afterwards). Without this the
#: Figures section was byte-identical at every detail level.
_K_FIGURES_BY_DETAIL = {
    ReportDetail.CONCISE: 8,
    ReportDetail.STANDARD: 16,
    ReportDetail.DETAILED: 28,
}

#: Back-compat default for callers that don't carry a detail level.
_K_FIGURES = _K_FIGURES_BY_DETAIL[ReportDetail.STANDARD]

#: Section labels whose chunks are page furniture / back-matter / figure
#: exhibits, not paper prose — kept out of the narrative sections so a summary
#: isn't grounded in the reference list, acknowledgements, or a figure's example
#: sentence (e.g. an Overview/Conclusion built from an attention-visualization
#: sample sentence).
_NONCONTENT_LABEL_RE = re.compile(
    r"\b(reference|bibliograph|acknowledg|appendix|author\s+contribution|"
    r"funding|conflict|supplementary|visuali[sz]ation)", re.IGNORECASE
)
#: Tokenizer / model special tokens that only appear in figure exhibits or raw
#: model I/O dumps, never in real paper prose.
_MODEL_TOKEN_RE = re.compile(r"<\s*/?\s*(eos|bos|pad|unk|s|mask|sep|cls)\s*>", re.IGNORECASE)


#: Maps a routing decision onto the wire value the UI switches on.
_ANSWER_KIND_BY_INTENT: dict[Intent, AnswerKind] = {
    Intent.CITATION: AnswerKind.CITATION,
    Intent.METADATA: AnswerKind.METADATA,
    Intent.STRUCTURE: AnswerKind.STRUCTURE,
    Intent.CONTENT: AnswerKind.CONTENT,
}

#: Chat's default grounding breadth when the caller doesn't say.
_DEFAULT_TOP_K = 6

#: Most figures ever shown beside one chat answer.
_MAX_CHAT_FIGURES = 3

#: "figure 3", "fig. 2", "table 4" inside a user's question.
_FIGURE_REF_RE = re.compile(r"\b(?:fig(?:ure)?s?|tables?)\.?\s*(\d{1,2})\b", re.IGNORECASE)

_WORD_RE = re.compile(r"[a-z0-9]+")
_TRAILING_INT_RE = re.compile(r"(\d{1,3})\s*$")

#: The anti-hallucination rule. It is stated as a positive instruction ("say
#: you can't") rather than only a prohibition, because a bare "don't make things
#: up" leaves the model with no sanctioned alternative and it fills the gap
#: anyway. Asked "what are the limitations of this work?", the old prompt
#: produced two limitations that appear nowhere in the paper.
_GROUNDING_RULE = (
    "If the passages below do not actually answer the question, say so plainly "
    "in one sentence — 'The passages I can see don't cover that' — and then, "
    "only if it helps, state what they DO cover. Never fill a gap with a "
    "plausible-sounding claim: a wrong answer that reads well is the worst "
    "possible outcome here. Every specific claim, number or name you write must "
    "appear in the material below."
)

#: The chat renderer supports the same tiny markdown subset as the report.
_FORMAT_RULE = (
    "FORMATTING: plain sentences, using **bold** for emphasis. Do NOT use "
    "headings, bullet or numbered list syntax, tables or code fences — they "
    "render as raw characters in this reader."
)


def _is_uploaded(meta: PaperMeta) -> bool:
    """True for a user-uploaded PDF, whose arXiv fields are placeholders."""
    return (meta.arxiv_id or "").startswith("upload:")


def _trailing_int(text: Optional[str]) -> Optional[int]:
    """The trailing integer of a media label ('Fig 2' -> 2), else ``None``."""
    match = _TRAILING_INT_RE.search(text or "")
    return int(match.group(1)) if match else None


def _paper_note(meta: Optional[PaperMeta]) -> str:
    """One line telling the model which paper it is looking at.

    Cheap, and it stops a whole class of vagueness: without it the model has no
    idea of the paper's title, field or age, so it hedges. It is framing only —
    the grounding rule still forbids stating anything the passages don't support.
    """
    if meta is None:
        return ""
    bits = [f"The paper is {meta.title!r}"]
    if meta.authors:
        first = meta.authors[0]
        bits.append(f"by {first}{' et al.' if len(meta.authors) > 1 else ''}")
    if meta.published:
        bits.append(f"({meta.published.year})")
    return " ".join(bits) + ". Use this only to frame your answer. "


def _is_noncontent(chunk: AnyChunk) -> bool:
    """True if a chunk is back-matter, a figure exhibit, or a raw-token dump."""
    ref = getattr(chunk, "source_ref", None)
    label = getattr(ref, "section_label", "") or ""
    if _NONCONTENT_LABEL_RE.search(label):
        return True
    if _MODEL_TOKEN_RE.search(chunk.text or ""):
        return True
    return False


def _dedupe(chunks: Sequence[AnyChunk]) -> list[AnyChunk]:
    seen: dict[str, AnyChunk] = {}
    for chunk in chunks:
        seen.setdefault(chunk.id, chunk)
    return list(seen.values())


def flatten_outline(outline: dict[SectionName, list[AnyChunk]]) -> list[AnyChunk]:
    """Flatten a per-section outline into one de-duplicated, ordered chunk list."""
    seen: dict[str, AnyChunk] = {}
    for name in SECTION_ORDER:
        for chunk in outline.get(name, []):
            seen.setdefault(chunk.id, chunk)
    return list(seen.values())


class ResearchAgent(Agent[ChatMessage]):
    """Produces a grounded answer to a single question about one paper."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        citation_agent: Optional[CitationAgent] = None,
        media_agent: Optional[MediaAgent] = None,
    ) -> None:
        super().__init__(llm)
        self._citation_agent = citation_agent or CitationAgent()
        self._media_agent = media_agent or MediaAgent()

    def run(self, context: AgentContext) -> ChatMessage:
        """Answer ``context.extra['question']``, routed by what was actually asked.

        Not every question is a retrieval question, and treating them as if they
        all were is what made this chat useless. :func:`chat_intent.classify`
        splits them four ways:

        - ``CITATION`` / ``METADATA`` / ``STRUCTURE`` — answered
          **deterministically from the paper's own record**, with no retrieval
          and no model call. These are facts the body text does not contain, so
          retrieval could only ever produce a confident guess (it did: asked for
          an APA citation, the old path replied "based on the [n] markers, it
          appears that the citations are as follows: [1], [2], [3]").
        - ``CONTENT`` — the genuine RAG path, which now gets filtered,
          diversified chunks plus the paper's metadata for framing.

        Returns a fully populated :class:`ChatMessage` (text + ``answer_kind`` +
        citations + figures) so the chat route needs one call.
        """
        question = str(context.extra.get("question", "")).strip()
        chunks = list(context.chunks)

        if not question:
            return ChatMessage(
                id=message_id(),
                paper_id=context.paper_id,
                role=ChatRole.ASSISTANT,
                text="I didn't receive a question to answer.",
                answer_kind=AnswerKind.CONTENT,
                citations=[],
                figures=[],
            )

        meta = load_paper_meta(context.paper_id)
        intent = classify(question)

        if intent is not Intent.CONTENT and meta is not None:
            text = self._answer_from_record(intent, question, meta)
            if text:
                # No citations and no figures on purpose: this answer came from
                # the paper's record, not from a retrieved passage, and
                # attaching arbitrary figures to it would imply otherwise.
                return ChatMessage(
                    id=message_id(),
                    paper_id=context.paper_id,
                    role=ChatRole.ASSISTANT,
                    text=text,
                    answer_kind=_ANSWER_KIND_BY_INTENT[intent],
                    citations=[],
                    figures=[],
                )

        return self._answer_from_content(context, question, chunks, meta)

    # ---- Deterministic answers (no retrieval, no model) ---------------

    def _answer_from_record(
        self, intent: "Intent", question: str, meta: PaperMeta
    ) -> str:
        """Answer a citation/metadata/structure question from ``meta``.

        Returns ``""`` when the record genuinely cannot answer it, which sends
        the question back down the normal RAG path rather than inventing.
        """
        if intent is Intent.CITATION:
            # Deferred import: `deepvision.report.__init__` pulls in
            # report_generator -> agent_orchestrator -> this module, so importing
            # the formatter at module scope is a cycle. Same dodge study_agent
            # uses for chunk_quality. Only the citation path needs it.
            from deepvision.report.citation_styles import (
                CITATION_STYLE_LABELS,
                CitationStyle,
                detect_styles,
                format_citation,
            )

            wanted = detect_styles(question) or list(CitationStyle)
            lines = [
                f"**{CITATION_STYLE_LABELS[style]}** — {format_citation(meta, style)}"
                for style in wanted
            ]
            if not lines:
                return ""
            lead = (
                "Here is this paper cited in the style you asked for:"
                if len(lines) == 1
                else "Here is this paper in each citation style:"
            )
            return f"{lead}\n\n" + "\n\n".join(lines)

        if intent is Intent.METADATA:
            return self._metadata_answer(question, meta)

        if intent is Intent.STRUCTURE:
            return self._structure_answer(meta)

        return ""

    @staticmethod
    def _metadata_answer(question: str, meta: PaperMeta) -> str:
        """A plain-language answer about the paper's identity."""
        parts: list[str] = []
        low = question.lower()
        # A question can ask for more than one field ("who wrote this and
        # when"), so these are independent tests, not a chain — and "when" has
        # to be in the date list or that exact question silently answers only
        # half of itself, which it did.
        _DATE_WORDS = ("year", "publish", "date", "when", "written", "released")
        _AUTHOR_WORDS = ("author", "wrote", "who")
        wants_all = not any(
            k in low
            for k in _AUTHOR_WORDS + _DATE_WORDS + ("title", "arxiv", "categor", "subject")
        )

        if wants_all or any(k in low for k in _AUTHOR_WORDS):
            if meta.authors:
                parts.append(f"**Authors** — {', '.join(meta.authors)}")
            else:
                parts.append("**Authors** — not recorded for this paper.")
        if wants_all or any(k in low for k in _DATE_WORDS):
            published = meta.published.isoformat() if meta.published else None
            parts.append(
                f"**Published** — {published}" if published
                else "**Published** — no publication date is recorded."
            )
        if wants_all or "title" in low:
            parts.append(f"**Title** — {meta.title}")
        if wants_all or "arxiv" in low:
            if _is_uploaded(meta):
                parts.append(
                    "**Source** — an uploaded PDF, so it has no arXiv identifier."
                )
            else:
                link = meta.abs_url or f"https://arxiv.org/abs/{meta.arxiv_id}"
                parts.append(f"**arXiv** — {meta.arxiv_label or meta.arxiv_id} ({link})")
        if (wants_all or any(k in low for k in ("categor", "subject"))) and meta.categories:
            parts.append(f"**Categories** — {', '.join(meta.categories)}")

        return "\n\n".join(parts)

    @staticmethod
    def _structure_answer(meta: PaperMeta) -> str:
        """A plain-language answer about the document as an artifact."""
        parts: list[str] = []
        if meta.page_count:
            parts.append(f"**Pages** — {meta.page_count}")
        if meta.figure_count is not None:
            parts.append(f"**Figures and tables extracted** — {meta.figure_count}")
        if not parts:
            return ""
        parts.append(
            "Counts come from the ingested document itself, not from the text, so "
            "they reflect what DeepVision actually extracted."
        )
        return "\n\n".join(parts)

    # ---- The grounded RAG path ----------------------------------------

    def _answer_from_content(
        self,
        context: AgentContext,
        question: str,
        chunks: Sequence[AnyChunk],
        meta: Optional[PaperMeta],
    ) -> ChatMessage:
        """The genuine retrieval-grounded answer."""
        # The caller over-fetches (see `chat.py::_OVERFETCH`) precisely so there
        # is something to select from here: filter the furniture out, then keep
        # only chunks that say *different* things. Without the diversify step two
        # unrelated questions were measured sharing 4 of their 6 chunks, which is
        # why every answer read the same.
        top_k = int(context.extra.get("top_k") or _DEFAULT_TOP_K)
        prose = diversify(
            narrative_chunks(list(chunks), allow_fallback=False) or list(chunks),
            top_k,
        )

        draft = self._draft_answer(prose)
        history_note = self._history_note(context.extra.get("history"))
        system = (
            "You are DeepVision's research assistant. You answer questions about "
            "ONE paper, strictly from the passages supplied below. "
            f"{_paper_note(meta)}"
            f"{history_note}"
            f"The user asked: {question!r}. Rewrite the material below into a "
            "direct, specific answer to that exact question, in 3-6 sentences of "
            "plain language a reader with no background in the field can follow. "
            "Lead with the answer rather than with context. "
            "Preserve every '[n]' citation marker exactly where it grounds a "
            "claim — never add, remove or renumber one. "
            f"{_GROUNDING_RULE} {_FORMAT_RULE}"
        )
        answer = self._complete(system, draft, temperature=0.3, max_tokens=900)
        if not answer.strip():
            answer = draft
        answer = repair_citation_markers(answer, len(prose))

        citations = self._citation_agent.cite(answer, prose)
        # Figures relevant to THIS question, not simply the first four that came
        # back — the old `build_media(chunks)[:4]` is why the same figures sat
        # under every answer regardless of what was asked.
        figures = self._relevant_figures(question, chunks, prose)

        return ChatMessage(
            id=message_id(),
            paper_id=context.paper_id,
            role=ChatRole.ASSISTANT,
            text=answer,
            answer_kind=AnswerKind.CONTENT,
            citations=citations,
            figures=figures,
        )

    def _relevant_figures(
        self,
        question: str,
        all_chunks: Sequence[AnyChunk],
        used_chunks: Sequence[AnyChunk],
    ) -> list[MediaRef]:
        """Figures worth showing beside this answer, best first.

        Ranked, not truncated-in-arrival-order. A figure scores for being
        referenced by number in the question ("what does figure 3 show?"), for
        sitting on a page the answer is actually grounded in, and for caption
        overlap with the question's content words. A figure that scores nothing
        is not shown at all — an unrelated figure under an answer is worse than
        no figure, because it implies a connection that isn't there.
        """
        media = self._media_agent.build_media(all_chunks)
        if not media:
            return []

        asked_numbers = {int(n) for n in _FIGURE_REF_RE.findall(question)}
        used_pages = {
            getattr(c, "page", None) for c in used_chunks
            if getattr(c, "page", None) is not None
        }
        q_tokens = {
            t for t in _WORD_RE.findall(question.lower())
            if len(t) > 3
        }

        scored: list[tuple[int, int, MediaRef]] = []
        for index, ref in enumerate(media):
            score = 0
            label_number = _trailing_int(ref.label)
            if label_number is not None and label_number in asked_numbers:
                score += 10
            if ref.page in used_pages:
                score += 3
            caption_tokens = {
                t for t in _WORD_RE.findall((ref.caption or "").lower()) if len(t) > 3
            }
            score += min(3, len(q_tokens & caption_tokens))
            if score > 0:
                scored.append((score, -index, ref))

        scored.sort(key=lambda row: (-row[0], -row[1]))
        return [ref for _score, _order, ref in scored[:_MAX_CHAT_FIGURES]]

    def plan_outline(
        self,
        paper_id: str,
        settings: AppSettings,
        retriever: Retriever,
    ) -> dict[SectionName, list[AnyChunk]]:
        """Retrieve and group key chunks per fixed report section.

        Runs one targeted query per :data:`SECTION_ORDER` entry against
        ``retriever``. The Figures query result is filtered down to image/
        vision/OCR-modality chunks; any such chunks incidentally surfaced by
        the *other* section queries are folded in too, so figures aren't
        missed just because the figures-specific query scored them lower.
        Never raises: a failing retrieval query yields an empty group for that
        section instead of aborting the whole outline.

        **Chapter scoping needs no code here.** ``retriever`` is the single
        retrieval entry point of the whole report pipeline, so a caller that
        passes a
        :class:`~deepvision.rag.chapter_scope.PageScopedRetriever` gets an
        outline — including the Figures group and its media chunks — drawn
        exclusively from that chapter's pages. The scoped retriever over-fetches
        internally, so the ``k`` values below stay the number of chunks this
        method actually wants back.
        """
        outline: dict[SectionName, list[AnyChunk]] = {}
        overall_seen: dict[str, AnyChunk] = {}

        detail = getattr(settings, "report_detail", ReportDetail.STANDARD)
        per_section_k = _K_BY_DETAIL.get(detail, _K_BY_DETAIL[ReportDetail.STANDARD])
        figures_k = _K_FIGURES_BY_DETAIL.get(detail, _K_FIGURES)

        for name in SECTION_ORDER:
            query = _SECTION_QUERIES.get(name)
            if not query:
                log.warning(
                    "no retrieval query for section; leaving it empty",
                    extra={"paper_id": paper_id, "section": name.value},
                )
                outline[name] = []
                continue
            k = figures_k if name is SectionName.FIGURES else per_section_k
            try:
                hits = list(retriever.retrieve(query, paper_id, k=k))
            except Exception:
                log.warning(
                    "outline retrieval failed",
                    extra={"paper_id": paper_id, "section": name.value},
                    exc_info=True,
                )
                hits = []
            if name is SectionName.FIGURES:
                hits = [c for c in hits if c.modality in _MEDIA_MODALITIES]
            else:
                # Keep references/acknowledgements/appendix out of the narrative
                # sections; fall back to the unfiltered hits only if filtering
                # would empty the section entirely.
                content_hits = [c for c in hits if not _is_noncontent(c)]
                hits = content_hits or hits
                # Then strip running headers, drop captions/boilerplate, and
                # de-dupe near-identical chunks (e.g. the same caption pulled
                # twice) — never at the cost of emptying the section, see
                # narrative_chunks' fallback-to-original guarantee.
                hits = narrative_chunks(hits)
            outline[name] = _dedupe(hits)
            for chunk in hits:
                overall_seen.setdefault(chunk.id, chunk)

        extra_media = [c for c in overall_seen.values() if c.modality in _MEDIA_MODALITIES]
        if extra_media:
            merged = list(outline.get(SectionName.FIGURES, []))
            existing_ids = {c.id for c in merged}
            for chunk in extra_media:
                if chunk.id not in existing_ids:
                    merged.append(chunk)
                    existing_ids.add(chunk.id)
            outline[SectionName.FIGURES] = merged

        # Running-header stripping is a *corpus-wide* judgement: a prefix only
        # counts as page furniture if it recurs on 3+ distinct pages, and a
        # single section's 10-chunk group rarely contains three. Run once over
        # the union of everything retrieved, then push the cleaned text back
        # into each section — otherwise "(IJCSIS) International Journal of
        # Computer Science and Information Security, Vol. 7, No. 1, 2010" stays
        # glued to the front of the Methods section's only sentence, which is
        # exactly what shipped.
        cleaned = {c.id: c for c in strip_repeated_prefixes(list(overall_seen.values()))}
        if cleaned:
            for name, chunks in outline.items():
                # A chunk absent from `cleaned` was dropped as empty-after-strip.
                outline[name] = [cleaned[c.id] for c in chunks if c.id in cleaned]

        return outline

    @staticmethod
    def _draft_answer(chunks: Sequence[AnyChunk]) -> str:
        if not chunks:
            return (
                "I couldn't find grounded passages in this paper for that question "
                "yet. Try rephrasing, or check that the paper has finished ingesting."
            )
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            snippet = clean_snippet(chunk.text, max_chars=320)
            if snippet:
                parts.append(f"{snippet} [{i}]")
        return " ".join(parts) if parts else (
            "The retrieved passages did not contain readable text to answer this "
            "question."
        )

    @staticmethod
    def _history_note(history: object) -> str:
        if not history:
            return ""
        try:
            recent = list(history)[-4:]
        except TypeError:
            return ""
        lines = []
        for msg in recent:
            if isinstance(msg, dict):
                role_val = msg.get("role")
                text = msg.get("text", "")
            else:
                role = getattr(msg, "role", None)
                role_val = getattr(role, "value", role)
                text = getattr(msg, "text", "")
            if text:
                lines.append(f"{role_val}: {text}")
        if not lines:
            return ""
        return "Conversation so far:\n" + "\n".join(lines) + "\n\n"
