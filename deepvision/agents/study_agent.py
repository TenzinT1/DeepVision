"""Study agent — writes the Study Questions section.

Study Questions is one of the report's eleven fixed sections (see
:class:`deepvision.models.report.SectionName`): it exists so a reader with no
background in the topic can verify they actually understood the material they
just read.
**The Glossary section is gone.** A real generated report showed Glossary and
Key Concepts emitting near-identical term/definition pairs for the same
vocabulary — exactly the kind of duplication
:data:`~deepvision.models.report.SECTION_QUESTIONS` now forbids project-wide
(one section, one question, and no other section may answer it). Rather than
maintain two nearly-identical extraction paths, the sections were merged into
Key Concepts and the Glossary section was deleted. :func:`extract_terms` — the
deterministic term miner that used to anchor the Glossary draft — is the part
that survives: it now anchors the **Key Concepts** draft built in
``summarizer_agent.py``, which imports it from this module. This module keeps
:func:`extract_terms` exported with its exact original signature so that
import keeps working; :class:`StudyAgent` itself no longer calls it or builds
a Glossary section.

Like every other agent here, the Study Questions section is built as a
deterministic, citation-marked draft first and only then handed to the LLM for
polish through :func:`deepvision.agents.base.complete_with_fallback`. That is a
hard project rule, and it is what makes the section safe: with ``EchoLLM``, a
missing model, or a provider error, the draft itself is shipped — a real,
grounded question set, not a crash and not an empty section.

One deliberate design choice:

- **Questions are templated, answers are grounded.** The question ladder in
  :data:`_QUESTION_PLAN` escalates from recall to synthesis and is fixed, so the
  numbering in ``body_markdown`` can never drift from the numbering in
  ``deep_dive_markdown``. Only the *answers* are drawn from (and cited to) the
  paper's retrieved chunks, and only the answers are LLM-polished.

Per the Section contract, the questions live in ``body_markdown`` and their
answers in ``deep_dive_markdown``, so the reader attempts them first and
reveals the answers with the existing "+ Deep dive" toggle.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

from deepvision.agents.base import (
    Agent,
    AgentContext,
    clean_snippet,
    load_paper_meta,
    provenance_list,
)
from deepvision.agents.citation_agent import CitationAgent
from deepvision.models.chunks import AnyChunk, Modality
from deepvision.models.report import Section, SectionName
from deepvision.models.settings import ReportDetail
from deepvision.providers.base import LLMProvider
from deepvision.utils.ids import section_id
from deepvision.utils.logger import get_logger

__all__ = ["StudyAgent", "extract_terms"]

log = get_logger(__name__)

#: How many study questions each detail level aims for, and how much source
#: material feeds them. Kept in step with ``summarizer_agent._DETAIL_PARAMS``
#: and with the promise published in ``models.settings.DETAIL_LEVEL_INFO``.
_STUDY_PARAMS: dict[ReportDetail, dict] = {
    ReportDetail.CONCISE: {"max_chunks": 8, "questions": 5, "max_tokens": 1200},
    ReportDetail.STANDARD: {"max_chunks": 14, "questions": 6, "max_tokens": 1800},
    ReportDetail.DETAILED: {"max_chunks": 22, "questions": 8, "max_tokens": 3000},
}

#: The question ladder, in order, escalating recall -> comprehension ->
#: analysis -> synthesis, and shaped to walk the report's own eleven-section
#: arc: problem -> approach -> method -> evidence -> why prior work fell short
#: -> figures -> limitations -> significance. Each entry is
#: ``(question, preferred source sections)`` — the answer is drafted from the
#: first preferred section that actually has usable retrieved text, falling
#: back to the whole pool. The limitations question sources from the report's
#: own ``Limitations & Open Questions`` section now that one exists, and the
#: closing "why it matters" question sources from ``Why It Matters`` — both
#: used to fall back to the old catch-all ``Conclusions`` section, which no
#: longer exists.
_QUESTION_PLAN: list[tuple[str, tuple[SectionName, ...]]] = [
    (
        "In one or two sentences, what problem does this paper set out to solve?",
        (SectionName.BACKGROUND, SectionName.OVERVIEW),
    ),
    (
        "What do the authors propose, and what makes it different from what came before?",
        (SectionName.OVERVIEW, SectionName.METHODS),
    ),
    (
        "Walk through the method: what are its main components, and what does each one do?",
        (SectionName.METHODS,),
    ),
    (
        "Which results are offered as evidence, and how large is the improvement over the baselines?",
        (SectionName.KEY_RESULTS,),
    ),
    (
        "Why did the earlier approaches fall short where this one succeeds?",
        (SectionName.BACKGROUND, SectionName.WHY_IT_MATTERS),
    ),
    (
        "What do the paper's figures and tables show that the running text does not?",
        (SectionName.FIGURES, SectionName.KEY_RESULTS),
    ),
    (
        "What limitations or open questions does the paper leave behind?",
        (SectionName.LIMITATIONS,),
    ),
    (
        "How would you explain this work — and why it matters — to someone new to the field?",
        (SectionName.WHY_IT_MATTERS, SectionName.OVERVIEW, SectionName.KEY_RESULTS),
    ),
]

#: Formatting rule shared with the summarizer: the reader renders **bold**,
#: *italic* and ``[n]`` markers only.
_FORMAT_RULE = (
    "FORMATTING: plain paragraphs separated by blank lines, using **bold** only. "
    "Do NOT use headings (#), bullet or numbered list syntax (-, *, 1.), tables, "
    "links or code fences — they render as raw characters in this reader."
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
#: Leading article swept up by the (greedy) expansion capture, e.g. the "The" in
#: "The Byte Pair Encoding (BPE)".
_ARTICLE_RE = re.compile(r"^(?:the|a|an|our|this|these|its)\s+", re.IGNORECASE)

#: ``Full Name (ACR)`` — the paper defining its own acronym. Highest-value hit.
_EXPANSION_RE = re.compile(
    r"\b((?:[A-Za-z][\w\-]*\s+){1,5}[A-Za-z][\w\-]*)\s*\(\s*([A-Z][A-Za-z0-9\-]{1,9})s?\s*\)"
)
#: A bare acronym / CamelCase model name, e.g. ``BLEU``, ``GPU``, ``ResNet``.
_ACRONYM_RE = re.compile(r"\b([A-Z][A-Za-z0-9\-]*[A-Z0-9])\b")
#: A repeated Title-Case technical phrase, e.g. ``Scaled Dot-Product Attention``.
_PHRASE_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:[ \-][A-Z][a-z]{2,}){1,3})\b")

#: Capitalized tokens that are never domain vocabulary.
_STOP_TERMS: frozenset[str] = frozenset(
    {
        "THE", "AND", "FOR", "BUT", "NOT", "ALL", "ANY", "ONE", "TWO", "WE", "US",
        "IT", "IS", "ARE", "WAS", "THIS", "THAT", "THESE", "THOSE", "WITH", "FROM",
        "INTO", "OUR", "ITS", "CAN", "MAY", "USE", "USED", "NEW", "SEE", "FIG",
        "FIGURE", "TABLE", "SECTION", "APPENDIX", "EQ", "PDF", "URL", "HTTP",
        "HTTPS", "ARXIV", "DOI", "ISBN", "IEEE", "ACM", "ET", "AL", "II", "III",
        "IV", "VI", "VII", "IX", "XI", "A", "I", "AN", "OF", "IN", "ON", "TO", "BY",
        "AS", "AT", "OR", "IF", "SO", "BE", "DO", "NO", "UP",
    }
)

_MIN_DEFINITION_CHARS = 40
_MAX_DEFINITION_CHARS = 260

#: A run of lowercase letters — present in real prose, absent from ALL-CAPS
#: running headers, acronym strings and other non-sentence junk.
_LOWERCASE_RUN_RE = re.compile(r"[a-z]{3,}")
#: A short, deliberately loose whitelist of common copulas/auxiliaries/modals —
#: these show up in almost any predicate regardless of the paper's domain
#: vocabulary, so they are cheap high-confidence evidence of a real sentence.
_AUX_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|do|does|did|can|could|"
    r"will|would|shall|should|may|might|must)\b",
    re.IGNORECASE,
)
#: Any lowercase word ending in a common verb inflection (-s/-ed/-ing). Not a
#: real parser — a plain-tense verb like "propose" slips past both this and
#: the auxiliary list above, but between the two, one nearly always matches
#: in a genuine sentence ("...which merges pairs...", "...was trained on...",
#: "...is evaluated using..."), while a name/affiliation/journal-name
#: fragment ("AUTHORS PROFILE", "ISSN ... International Journal of ...")
#: typically matches neither.
_VERB_SUFFIX_RE = re.compile(r"\b[a-z]{3,}(?:s|ed|ing)\b")


def _looks_like_sentence(text: str) -> bool:
    """Cheap, deliberately simple check that ``text`` reads like a sentence.

    Journal boilerplate ("AUTHORS PROFILE", an ISSN/journal-name line, a
    running header) tends to be a title-cased or all-caps fragment with no
    real predicate — it is not a sentence, even though it can contain a
    capitalized token that looks like a term. Two signals catch most of it
    without an LLM: a run of lowercase letters (rules out ALL-CAPS fragments)
    and at least one verb-ish token — either a common auxiliary/modal or a
    word carrying a verb inflection (rules out noun-phrase fragments such as
    affiliations, journal names or bare citations, which tend to have
    neither).
    """
    text = text or ""
    if not _LOWERCASE_RUN_RE.search(text):
        return False
    return bool(_AUX_VERB_RE.search(text)) or bool(_VERB_SUFFIX_RE.search(text))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]


def _is_usable_term(term: str) -> bool:
    upper = term.upper()
    if upper in _STOP_TERMS:
        return False
    if len(term) < 2 or len(term) > 48:
        return False
    if not any(ch.isalpha() for ch in term):
        return False
    # A single all-caps word of one letter, or something that is all digits with
    # a stray capital, is noise rather than vocabulary.
    return not term.isdigit()


def extract_terms(
    chunks: Sequence[AnyChunk], *, limit: int
) -> list[tuple[str, str, int]]:
    """Mine ``chunks`` for the paper's vocabulary.

    Returns up to ``limit`` ``(term, defining_sentence, marker)`` triples, where
    ``marker`` is the 1-based index of the source chunk in ``chunks`` — i.e. the
    number to write as ``[marker]`` so
    :class:`~deepvision.agents.citation_agent.CitationAgent` resolves it to the
    right passage.

    Terms are ranked by how strong the evidence is that they are real domain
    vocabulary: an explicit ``Full Name (ACR)`` expansion beats a repeated
    acronym, which beats a repeated Title-Case phrase. Each term is paired with
    the sentence in the corpus that mentions it and looks most explanatory,
    which is the closest thing to a definition available without an LLM. Where
    the paper spells an acronym out, the two are merged into a single
    ``ACR (Full Name)`` entry rather than listed twice.

    Three precision filters keep journal boilerplate out (real observed junk:
    "**Computer Science** — AUTHORS PROFILE She has completed her Bachelor of
    engineering...", "**ISSN** — ...", "**IJCSIS** — (IJCSIS) International
    Journal of Computer Science and Information Security, Vol."):

    1. a candidate's whole defining sentence is dropped if
       :func:`~deepvision.rag.chunk_quality.is_boilerplate` flags it;
    2. a *bare* acronym (never seen with a spelled-out expansion anywhere in
       the paper, and never also seen as a Title-Case phrase) needs to appear
       in at least two distinct chunks before it earns a slot — a one-off
       all-caps token is almost always a journal name or a stray heading, not
       recurring vocabulary;
    3. a candidate is dropped if its defining sentence doesn't look like a
       real sentence (see :func:`_looks_like_sentence`) — catches ALL-CAPS
       running headers and noun-phrase fragments (affiliations, bare
       citations) that slipped past filter 1.

    Purely deterministic — no LLM, so it cannot fail or hallucinate.
    """
    if limit <= 0:
        return []

    # Deferred import: deepvision.rag.chunk_quality is being added by another
    # agent in this same build wave. Importing it lazily (the same pattern
    # build_deck() uses for study.card_queries, to dodge an import cycle) means
    # `import deepvision.agents.study_agent` never depends on build order —
    # only an actual call to extract_terms does, and by the time this ships
    # that module exists.
    from deepvision.rag.chunk_quality import is_boilerplate, prose_score

    # lowercased term -> [score, display term, defining sentence, marker]
    found: dict[str, list] = {}
    # lowercased acronym -> the paper's own spelled-out form for it
    expansions: dict[str, str] = {}
    # lowercased term -> distinct chunk markers it was seen in *only* as a
    # bare acronym (never via an expansion or a Title-Case phrase) — feeds the
    # two-distinct-chunks rule below.
    bare_acronym_markers: dict[str, set[int]] = {}
    # lowercased terms also seen via a non-acronym route (expansion or
    # phrase), which exempts them from that rule.
    non_acronym_keys: set[str] = set()

    def offer(
        term: str,
        score: int,
        sentence: str,
        marker: int,
        *,
        via_acronym: bool = False,
    ) -> None:
        term = term.strip(" -—:;,.").strip()
        if not _is_usable_term(term):
            return
        # Filter 3: reject a defining sentence that doesn't look like prose
        # before it can seed (or extend) an entry.
        if not _looks_like_sentence(sentence):
            return
        key = term.lower()
        if via_acronym:
            bare_acronym_markers.setdefault(key, set()).add(marker)
        else:
            non_acronym_keys.add(key)
        cleaned = clean_snippet(sentence, max_chars=_MAX_DEFINITION_CHARS)
        if not cleaned:
            return
        existing = found.get(key)
        if existing is None:
            found[key] = [score, term, cleaned, marker]
            return
        existing[0] += score
        # Prefer a longer (more explanatory) definition sentence, but only up to
        # the point where it stops being a one-liner.
        if len(cleaned) > len(existing[2]) and len(cleaned) >= _MIN_DEFINITION_CHARS:
            existing[2] = cleaned
            existing[3] = marker

    for marker, chunk in enumerate(chunks, start=1):
        text = (chunk.text or "").strip()
        if not text:
            continue
        for sentence in _sentences(text):
            # Filter 1: a boilerplate sentence contributes no terms at all —
            # not even one matched correctly, since its "definition" would be
            # lifted from that same boilerplate.
            if is_boilerplate(sentence):
                continue
            # Filter 1b: nor does non-prose. Text scraped out of a UI screenshot
            # is full of Title-Case tokens and so is a magnet for `_PHRASE_RE`:
            # "jeye blinks lei es File Help Scale Factor xan fiz.e=] yaxs
            # Refresh Preview Show Eyes…" produced the real glossary entries
            # "**File Help Scale Factor**" and "**Refresh Preview**". A term is
            # only as good as the sentence that defines it, and that sentence
            # has to read like a sentence.
            if prose_score(sentence) < 0.15:
                continue
            for expansion, acronym in _EXPANSION_RE.findall(sentence):
                expansion = _ARTICLE_RE.sub("", expansion.strip(" -—:;,.")).strip()
                acronym = acronym.strip()
                if _is_usable_term(acronym) and _is_usable_term(expansion):
                    expansions.setdefault(acronym.lower(), expansion)
                offer(acronym, 6, sentence, marker)
            for acronym in _ACRONYM_RE.findall(sentence):
                offer(acronym, 2, sentence, marker, via_acronym=True)
            for phrase in _PHRASE_RE.findall(sentence):
                offer(phrase, 1, sentence, marker)

    # Filter 2: a bare acronym with no expansion anywhere in the paper and no
    # sighting as a Title-Case phrase needs two independent chunk sightings.
    for key, markers in bare_acronym_markers.items():
        if key in non_acronym_keys or key in expansions:
            continue
        if len(markers) < 2:
            found.pop(key, None)

    # Drop a spelled-out form that is already carried by its own acronym entry,
    # so "RNN" and "Recurrent Neural Networks" don't both take a slot.
    redundant = {exp.lower() for exp in expansions.values()}
    ranked = sorted(
        (row for row in found.values() if row[1].lower() not in redundant),
        key=lambda row: (-row[0], len(row[1]), row[1].lower()),
    )

    picked: list[tuple[str, str, int]] = []
    for _score, term, sentence, marker in ranked[:limit]:
        expansion = expansions.get(term.lower())
        display = f"{term} ({expansion})" if expansion else term
        picked.append((display, sentence, marker))
    picked.sort(key=lambda row: row[0].lower())
    return picked


def _textual(chunks: Iterable[AnyChunk]) -> list[AnyChunk]:
    """Chunks that carry readable prose (text/OCR/vision captions)."""
    out: list[AnyChunk] = []
    for chunk in chunks:
        if chunk.modality is Modality.IMAGE:
            continue
        if (chunk.text or "").strip():
            out.append(chunk)
    return out


class StudyAgent(Agent[list[Section]]):
    """Writes the Study Questions section."""

    def __init__(self, llm: LLMProvider, *, citation_agent: Optional[CitationAgent] = None) -> None:
        super().__init__(llm)
        self._citation_agent = citation_agent or CitationAgent()

    def run(self, context: AgentContext) -> list[Section]:
        """Return ``[Study Questions]`` for this paper.

        Reads ``context.extra['outline']`` (the ``SectionName -> chunks`` map
        from ``ResearchAgent.plan_outline``) when present so answers can be
        drawn from the section that actually discusses them; otherwise falls
        back to ``context.chunks`` for everything.
        """
        outline: dict = context.extra.get("outline") or {}
        pool = _textual(context.chunks)
        detail = getattr(context.settings, "report_detail", ReportDetail.STANDARD)
        params = _STUDY_PARAMS.get(detail, _STUDY_PARAMS[ReportDetail.STANDARD])

        title = ""
        meta = load_paper_meta(context.paper_id)
        if meta is not None:
            title = (meta.title or "").strip()

        return [self._questions_section(outline, pool, params, title)]

    # ---- Study Questions ---------------------------------------------

    def _questions_section(
        self, outline: dict, pool: Sequence[AnyChunk], params: dict, title: str
    ) -> Section:
        wanted = params["questions"]
        plan = _QUESTION_PLAN[:wanted]

        answer_chunks: list[AnyChunk] = []
        marker_by_chunk: dict[str, int] = {}
        used_ids: set[str] = set()

        def take(preferred: Sequence[SectionName]) -> Optional[AnyChunk]:
            """Pick the best not-yet-used chunk for a question's answer."""
            candidates: list[AnyChunk] = []
            for name in preferred:
                candidates.extend(_textual(outline.get(name) or []))
            candidates.extend(pool)
            for chunk in candidates:
                if chunk.id not in used_ids:
                    used_ids.add(chunk.id)
                    return chunk
            return candidates[0] if candidates else None

        # The title-flavoured phrasing belongs to the ladder's own closing
        # "explain it to a newcomer" question, not to whatever question happens
        # to land last once the plan is truncated for a lower detail level —
        # otherwise Concise silently rewrites "What limitations or open
        # questions does the paper leave behind?" into the explain question
        # while still sourcing its answer from Limitations.
        explain_question = _QUESTION_PLAN[-1][0]

        qa_parts: list[str] = []
        question_parts: list[str] = []
        for number, (question, preferred) in enumerate(plan, start=1):
            text = question
            if question == explain_question and title:
                text = (
                    f"How would you explain **{title}** — and why it matters — to "
                    "someone new to the field?"
                )
            question_parts.append(f"**{number}. {text}**")

            chunk = take(preferred)
            if chunk is None:
                qa_parts.append(
                    f"**{number}. {text}**\n\n"
                    "No grounded passage was retrieved to answer this yet."
                )
                continue
            marker = marker_by_chunk.get(chunk.id)
            if marker is None:
                answer_chunks.append(chunk)
                marker = len(answer_chunks)
                marker_by_chunk[chunk.id] = marker
            snippet = clean_snippet(chunk.text, max_chars=420)
            qa_parts.append(f"**{number}. {text}**\n\n{snippet} [{marker}]")

        body = "\n\n".join(question_parts)
        answers_draft = "\n\n".join(qa_parts)

        system = (
            "You are writing the answer key for the Study Questions of a study "
            "guide, for a reader with NO background in the field. Below, each "
            "bold numbered line is a question, followed by the raw passage from "
            "the paper that answers it. Rewrite ONLY the passage under each "
            "question into a direct 2-4 sentence answer in plain language that "
            "actually answers that question. Keep every bold numbered question "
            "line exactly as written, in the same order, and keep each answer's "
            "'[n]' citation marker. Never state anything the passage does not "
            f"support. {_FORMAT_RULE}"
        )
        deep_dive = self._complete(
            system, answers_draft, temperature=0.3, max_tokens=params["max_tokens"]
        )

        citations = (
            self._citation_agent.cite(deep_dive, answer_chunks) if answer_chunks else []
        )
        return Section(
            id=section_id(),
            name=SectionName.STUDY_QUESTIONS,
            body_markdown=body,
            deep_dive_markdown=deep_dive or None,
            badge="SELF-CHECK",
            provenance=provenance_list(answer_chunks),
            citations=citations,
            media=[],
            confidence=0.6 if answer_chunks else 0.2,
            default_open=False,
        )
