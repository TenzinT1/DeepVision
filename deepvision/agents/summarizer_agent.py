"""Summarizer agent — writes nine of the report's eleven sections.

Turns retrieved/extracted chunks into the report's narrative sections (At a
Glance, Overview, Background, Key Concepts, Methods, Key Results, Limitations &
Open Questions, Why It Matters, Key Takeaways) with body + deep-dive markdown,
provenance, citations and a per-section ``degraded`` flag.
(AGENTS). Figures are :class:`~deepvision.agents.media_agent.MediaAgent`'s job;
Study Questions are :class:`~deepvision.agents.study_agent.StudyAgent`'s.

Each section is built as a deterministic, citation-marked extractive draft from
its chunks first (see :mod:`deepvision.agents.base`), then optionally polished
by the LLM — which degrades to the same draft under ``EchoLLM``, a provider
error, or a timeout. **The draft is therefore a shipping artifact, not
scaffolding**: it must read as a finished section on its own, and it must never
contain a word addressed to the model. (It used to: the Overview draft opened
with the literal label ``"Paper abstract (context, not to be cited): "``, which
shipped verbatim to readers whenever that call fell back. The instruction now
lives in the *system* prompt and the draft carries the bare abstract text.)
Citations are grounded via
:class:`~deepvision.agents.citation_agent.CitationAgent` against the exact chunk
list the draft's ``[n]`` markers refer to.

**Design premise: the reader has no background in the topic.** Every section
prompt below is written for a motivated beginner — plain language, jargon
defined on first use, no assumed familiarity with the field.

Three mechanisms in this module are worth understanding before editing it.

*Generation order is not display order.* :data:`SECTION_ORDER`
(``models/report.py``) is what the reader sees; :data:`_GENERATION_ORDER` is the
order these nine are *written*, and the difference is load-bearing. ``At a
Glance`` is written first because it is a self-contained fixed-schema fact card
that must not be contaminated by whatever prose came before it. ``Key Takeaways``
is written **last** — it displays late (after the evidence) and it is now a
*recap*, so it must be able to see everything already written this run. The
report layer (``report/interactive_sections.normalize_sections``) reorders the
finished sections into display order, so nothing downstream depends on the order
they were produced in.

*Anti-redundancy.* The previous section set said the same thing four times: Key
Concepts and Glossary emitted near-identical term/definition pairs, and Key
Results / Conclusions / Key Takeaways all answered "what did they find". Two
prompt-level mechanisms now prevent that, at the cost of input tokens only —
never an extra model call. (1) :data:`~deepvision.models.report.SECTION_QUESTIONS`
gives each section the one question it owns; that question opens its system
prompt, and every *other* section's question is listed as "answered elsewhere,
do not answer it here". (2) A **running digest** — ``(section name, first ~200
characters of its body)`` appended after each section completes, capped at
:data:`_DIGEST_MAX` entries — is fed into every subsequent section's system
prompt as "already covered, add what is new".

*Chunk quality.* Retrieval happily returns figure captions and journal running
headers, and an extractive draft turns them straight into body prose — a real
generated report had an entire Background section consisting of the same figure
caption repeated four times. Every section's chunk list is therefore passed
through :func:`~deepvision.rag.chunk_quality.narrative_chunks` before drafting,
and sentence-level drafts (At a Glance, Key Concepts, Key Takeaways,
Limitations) additionally drop individual sentences that
:func:`~deepvision.rag.chunk_quality.is_caption` or
:func:`~deepvision.rag.chunk_quality.is_boilerplate` rejects.

Two things control output length, and only these two: :data:`_DETAIL_PARAMS`
(how much grounded material is fed in, and the length target/hard constraint
handed to the LLM) and ``research_agent._K_BY_DETAIL`` (how much is retrieved in
the first place). They are deliberately spread ~3x apart per level so that
``concise`` and ``standard`` are visibly different documents, matching the
promise published to the UI in
:data:`deepvision.models.settings.DETAIL_LEVEL_INFO`.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from deepvision.agents.base import (
    Agent,
    AgentContext,
    clean_snippet,
    llm_fallback_count,
    load_paper_meta,
    provenance_list,
    repair_citation_markers,
)
from deepvision.agents.citation_agent import CitationAgent
from deepvision.agents.study_agent import extract_terms
from deepvision.models.chunks import AnyChunk, Modality
from deepvision.models.report import (
    AT_A_GLANCE_FIELDS,
    SECTION_QUESTIONS,
    Section,
    SectionName,
)
from deepvision.models.settings import ReportDetail
from deepvision.providers.base import LLMProvider
from deepvision.rag.chunk_quality import (
    demote_low_prose,
    is_boilerplate,
    is_caption,
    narrative_chunks,
)
from deepvision.utils.ids import section_id
from deepvision.utils.logger import get_logger

__all__ = ["SummarizerAgent"]

log = get_logger(__name__)

#: The nine sections this agent writes, in **generation** order — which is
#: deliberately NOT the display order in ``models.report.SECTION_ORDER``.
#:
#: - ``At a Glance`` is first: it is a standalone fixed-schema fact card, so it
#:   is written before any digest exists to bias it.
#: - ``Key Takeaways`` is last: it displays after the evidence and is now a
#:   *recap*, so it must be able to see every other section's digest entry.
#:
#: ``Figures`` belongs to ``MediaAgent``; ``Study Questions`` to ``StudyAgent``.
_GENERATION_ORDER: list[SectionName] = [
    SectionName.AT_A_GLANCE,
    SectionName.OVERVIEW,
    SectionName.BACKGROUND,
    SectionName.KEY_CONCEPTS,
    SectionName.METHODS,
    SectionName.KEY_RESULTS,
    SectionName.LIMITATIONS,
    SectionName.WHY_IT_MATTERS,
    SectionName.KEY_TAKEAWAYS,
]

#: Sections that start expanded. Mirrors
#: ``report.interactive_sections.DEFAULT_OPEN_SECTIONS`` (which re-applies it
#: authoritatively on both generate and load); duplicated as a plain tuple here
#: to keep the agents layer free of a report-layer import. The three sections
#: that *orient* a reader with no background are open on arrival: the fact card,
#: what this paper is, and what came before it. Key Takeaways is deliberately
#: NOT here any more — as a recap it only lands after the reader has seen the
#: material, and opening more than three buries the section list that is the
#: reader's map.
_DEFAULT_OPEN: tuple[SectionName, ...] = (
    SectionName.AT_A_GLANCE,
    SectionName.OVERVIEW,
    SectionName.BACKGROUND,
)

#: Fallback keyword buckets used only when no pre-built outline is supplied
#: (e.g. the agent is exercised standalone against raw retrieved chunks). One
#: entry per name in :data:`_GENERATION_ORDER`.
_SECTION_KEYWORDS: dict[SectionName, tuple[str, ...]] = {
    SectionName.AT_A_GLANCE: (
        "problem", "propose", "present", "we introduce", "dataset", "experiment",
        "accuracy", "achiev", "result", "application",
    ),
    SectionName.OVERVIEW: (
        "overview", "abstract", "introduc", "motivat", "propose", "contribut", "problem",
    ),
    SectionName.BACKGROUND: (
        "background", "prior work", "related work", "previously", "existing", "traditional",
        "motivat", "limitation", "challenge", "difficult", "problem",
    ),
    SectionName.KEY_CONCEPTS: (
        "we define", "is defined", "definition", "notation", "preliminar", "concept",
        "intuition", "refers to", "denotes", "terminolog", "formulat",
    ),
    SectionName.METHODS: (
        "method", "approach", "architectur", "model", "algorithm", "framework", "design",
        "implement",
    ),
    SectionName.KEY_RESULTS: (
        "result", "experiment", "evaluat", "performance", "accuracy", "benchmark",
        "outperform", "table", "score",
    ),
    SectionName.LIMITATIONS: (
        "limitation", "however", "future work", "does not", "cannot", "fail",
        "assum", "constraint", "restrict", "remains", "open question", "drawback",
    ),
    SectionName.WHY_IT_MATTERS: (
        "impact", "practical", "application", "enable", "useful", "deploy",
        "real-world", "real world", "significan", "implication", "benefit",
    ),
    SectionName.KEY_TAKEAWAYS: (
        "we show", "we find", "notably", "importantly", "in short", "key finding",
        "main contribution", "significant", "outperform", "state of the art", "novel",
    ),
}

#: Upper bound used only by the standalone keyword-fallback bucket.
_MAX_CHUNKS_PER_SECTION = 18

#: Sections that mine *sentences* out of the whole retrieved set rather than
#: rewriting a handful of passages, and therefore benefit from a wider net than
#: the level's ``max_chunks``. At a Glance has to answer five different
#: questions from one pool; Key Concepts is now doing two jobs (it absorbed the
#: Glossary) and its vocabulary miner wants breadth.
_WIDE_NET_BONUS: dict[SectionName, int] = {
    SectionName.AT_A_GLANCE: 6,
    SectionName.KEY_CONCEPTS: 6,
}

#: Per-detail-level knobs: how many grounded snippets feed the body vs. the deep
#: dive, and the length target + hard sentence-count constraint handed to the
#: LLM. The three levels are spread roughly 3x apart so the difference is
#: obvious in the rendered report — ``target``/``length_rule`` here must stay in
#: agreement with ``DETAIL_LEVEL_INFO`` in ``deepvision/models/settings.py``,
#: which is what the Settings screen promises the user. Those four length
#: strings are unchanged by the eleven-section redesign precisely so that
#: promise still holds: the report grew by two sections, but the per-section
#: length contract did not move.
#:
#: ``concept_count`` was raised (4/6/8 -> 5/8/12) because Key Concepts absorbed
#: the deleted Glossary and now carries the paper's whole vocabulary load.
#: ``takeaway_count`` is unchanged: Key Takeaways became a recap, not a longer
#: list. ``At a Glance`` has no count knob — it is always exactly the five
#: fields of :data:`~deepvision.models.report.AT_A_GLANCE_FIELDS`.
_DETAIL_PARAMS: dict[ReportDetail, dict] = {
    ReportDetail.CONCISE: {
        "max_chunks": 5, "body_parts": 2, "deep_parts": 3,
        "target": "a single tight paragraph of 2-3 sentences",
        "length_rule": (
            "HARD LENGTH LIMIT: exactly one paragraph, 2-3 sentences total. "
            "Do not write a fourth sentence. Do not write a second paragraph."
        ),
        "deep_target": "roughly 2 extra sentences",
        "deep_length_rule": "HARD LENGTH LIMIT: 2 sentences total, one paragraph.",
        "concept_count": 5,
        "takeaway_count": 4,
        "max_tokens": 400,
    },
    ReportDetail.STANDARD: {
        "max_chunks": 10, "body_parts": 6, "deep_parts": 6,
        "target": "two clear paragraphs totalling 6-9 sentences",
        "length_rule": (
            "HARD LENGTH LIMIT: exactly two paragraphs, 6-9 sentences in total. "
            "Fewer than 6 sentences is too short; more than 9 is too long."
        ),
        "deep_target": "roughly 5 sentences of additional depth",
        "deep_length_rule": "HARD LENGTH LIMIT: about 5 sentences total.",
        "concept_count": 8,
        "takeaway_count": 6,
        "max_tokens": 1100,
    },
    ReportDetail.DETAILED: {
        "max_chunks": 18, "body_parts": 14, "deep_parts": 10,
        "target": "four to five thorough paragraphs totalling 16-22 sentences",
        "length_rule": (
            "HARD LENGTH LIMIT: 4-5 paragraphs, 16-22 sentences in total. "
            "Anything under 16 sentences is too short — develop the explanation "
            "fully rather than summarising it."
        ),
        "deep_target": "roughly 10 sentences of additional depth",
        "deep_length_rule": "HARD LENGTH LIMIT: about 10 sentences total.",
        "concept_count": 12,
        "takeaway_count": 8,
        "max_tokens": 2600,
    },
}

#: Per-section instruction, prepended to the shared length/citation rules. Each
#: one states the section's job for a reader with zero background — and, for the
#: four sections that used to blur into one another, states explicitly what
#: belongs to a *different* section. Read these next to
#: :data:`~deepvision.models.report.SECTION_QUESTIONS`: the question says what
#: the section answers, the brief says how to answer it.
_SECTION_BRIEF: dict[SectionName, str] = {
    SectionName.AT_A_GLANCE: (
        "This is a five-line fact card, not prose and not a summary — it is the "
        "first thing the reader sees and it must be scannable in ten seconds. "
        "Each line answers its own bold label and nothing else. No opening "
        "sentence, no closing sentence, no connective text between lines."
    ),
    SectionName.OVERVIEW: (
        "This is the reader's first contact with the paper in prose. Say what "
        "the paper is, what it set out to do, and why it exists — in plain "
        "language a curious non-specialist can follow. No unexplained jargon. "
        "Do not list the findings or the numbers; Key Results owns those."
    ),
    SectionName.BACKGROUND: (
        "Explain the situation BEFORE this paper existed: what problem the field "
        "was stuck on, what people were doing about it, and why those earlier "
        "approaches fell short. Assume the reader knows nothing about the area — "
        "use everyday language and analogies where they help, keep technical "
        "terms to a minimum, and define any you must use. Do not describe this "
        "paper's own contribution here; that belongs to Overview and Methods."
    ),
    SectionName.KEY_CONCEPTS: (
        "This is the reader's vocabulary section — it absorbed what used to be a "
        "separate glossary, so it is the ONLY place terms get defined. Write "
        "each concept as its own short paragraph that STARTS with the term in "
        "bold — for example: **Attention** — a way for a model to weigh… — "
        "followed by a plain-language definition and then one sentence on why "
        "that idea matters in THIS paper specifically. Never assume prior "
        "knowledge, and never define a term using other undefined jargon: if "
        "the definition needs a second unexplained term, rewrite it in ordinary "
        "words instead. If an entry's source sentence is a journal header, an "
        "ISSN or volume line, a publisher name, or an author biography, DROP "
        "that entry entirely rather than defining it — it is page furniture, "
        "not vocabulary."
    ),
    SectionName.METHODS: (
        "Explain what the authors actually did, step by step and in order. "
        "Prefer a plain description of the mechanism over restating the paper's "
        "notation; define every technical term the first time it appears. Do "
        "not report how well it worked — that is Key Results."
    ),
    SectionName.KEY_RESULTS: (
        "Findings ONLY. Report what was measured and what came out: the concrete "
        "numbers, the baselines they are measured against, the size of the "
        "improvement, and how strong the evidence is (how much data, how many "
        "trials, what conditions). Say what each number means in practice so "
        "the reader can judge whether an improvement is big. Do NOT discuss why "
        "the findings are significant, what they enable, or who benefits — that "
        "belongs to Why It Matters and writing it here is the exact duplication "
        "this section set was redesigned to remove."
    ),
    SectionName.LIMITATIONS: (
        "Cover ONLY what this work does not establish: limitations the authors "
        "state, conditions they did not test, assumptions they rely on, threats "
        "to the validity of the results, and future work they explicitly leave "
        "open. This section is grounded reporting, not critique: every caveat "
        "you write must be visible in the material below. If the material does "
        "not state the work's limitations, say so plainly in your first "
        "sentence and then name only what is VISIBLY untested in what you were "
        "given — for example that results come from a single dataset, a single "
        "camera or capture setup, one language, or that no ablation or "
        "comparison against an alternative is shown. Never invent a limitation, "
        "never guess at one that would be plausible for this kind of work, and "
        "never attach a citation marker to a caveat the material does not "
        "support."
    ),
    SectionName.WHY_IT_MATTERS: (
        "Consequences only: what changes because of this work, who is affected, "
        "what it makes possible that was not possible before, and what a "
        "practitioner or researcher should do differently. Argue for "
        "significance in plain language. Do NOT restate the numbers or the "
        "measurements — Key Results already reported them, and repeating them "
        "here is the duplication this section set was redesigned to remove."
    ),
    SectionName.KEY_TAKEAWAYS: (
        "This is the RECAP, and the reader has just read everything above it — "
        "the fact card, the background, the concepts, the methods, the results, "
        "the caveats and the significance. Produce the handful of claims worth "
        "remembering a month from now. Each one must be a short bold lead-in "
        "phrase followed by an em dash and ONE complete self-contained sentence "
        "that still makes sense quoted on its own. Prefer a concrete number, a "
        "named result, or a specific mechanism over a generality like 'the "
        "method works well', and make each takeaway about something different. "
        "This must NOT be a paraphrase of the At a Glance card: that card told "
        "the reader what the paper is before they read it, while these are the "
        "claims that survived reading it."
    ),
}

#: How many bold-term paragraphs the Key Concepts section aims for, by level.
_CONCEPT_INSTRUCTION = (
    "Produce {count} concepts, each as its own paragraph beginning with the term "
    "in bold followed by an em dash, then a plain-language definition and one "
    "sentence on why the idea matters in this paper."
)

#: How many standalone claims the Key Takeaways recap aims for, by level.
#: Count scales with report_detail (not per-item length) — see _DETAIL_PARAMS.
_TAKEAWAY_INSTRUCTION = (
    "Produce {count} standalone takeaways — the recap a reader should still "
    "remember a month from now — each as its own paragraph beginning with a "
    "short bold lead-in phrase (3-6 words) followed by an em dash and ONE "
    "complete sentence that stands entirely on its own."
)

#: The At a Glance rewrite instruction. The five bold labels are already in the
#: draft (see :func:`_glance_draft_parts`), so the model's only job is the text
#: after each em dash — which is what keeps the card correctly shaped even on a
#: fully degraded run.
_GLANCE_INSTRUCTION = (
    "the same five labelled lines, rewriting ONLY the text after each em dash "
    "into one short line (at most about 25 words) that directly answers that "
    "label"
)
_GLANCE_LENGTH_RULE = (
    "HARD SHAPE RULE: output exactly five paragraphs, one per line of the "
    "material, in the same order. Keep each line's bold label and the em dash "
    "after it exactly as given — never merge, drop, add, reword or reorder the "
    "labels. Each line is a STATEMENT, not a question: never restate the label "
    "as a question, and never write a question mark. Keep each line's bracketed "
    "citation number exactly as it appears in the material, in square brackets "
    "at the end of that same line. A line reading 'Not stated in the retrieved "
    "passages.' must be left exactly as it is: do not invent a value for it and "
    "do not give it a citation marker."
)

#: Header pill per section. Mirrors ``report.interactive_sections.SECTION_BADGES``
#: for the nine sections this agent owns; ``None`` means "no badge" and the
#: report layer leaves it alone. These two tables must agree exactly.
_SECTION_BADGE: dict[SectionName, Optional[str]] = {
    SectionName.AT_A_GLANCE: "FACT CARD",
    SectionName.OVERVIEW: "SUMMARY",
    SectionName.BACKGROUND: "CONTEXT",
    SectionName.KEY_CONCEPTS: "CONCEPTS",
    SectionName.METHODS: None,
    SectionName.KEY_RESULTS: "KEY RESULTS",
    SectionName.LIMITATIONS: "CAVEATS",
    SectionName.WHY_IT_MATTERS: "IMPACT",
    SectionName.KEY_TAKEAWAYS: "RECAP",
}

#: Max characters of the paper's own abstract used to seed the Overview lead.
_ABSTRACT_LEAD_CHARS = 900

#: Anti-redundancy digest: how much of each finished section is remembered, and
#: how many entries are carried. Eight is the whole run (nine sections, the last
#: of which never reads its own entry), so the cap is a guard against future
#: growth rather than a live truncation.
_DIGEST_CHARS = 200
_DIGEST_MAX = 8

#: Formatting rule shared by every section prompt. The frontend renderer
#: supports **bold**, *italic* and ``[n]`` markers only — headings, tables,
#: links and bullet syntax would show up as raw characters.
_FORMAT_RULE = (
    "FORMATTING: plain paragraphs separated by blank lines, using **bold** for "
    "emphasis or a leading term. Do NOT use headings (#), bullet or numbered "
    "list syntax (-, *, 1.), tables, links or code fences — they render as raw "
    "characters in this reader."
)

#: Told to the model only when an abstract lead is present (Overview, whole-paper
#: reports). The *draft* carries the bare abstract text with no label in front of
#: it, because the draft is what ships when the call falls back.
_ABSTRACT_RULE = (
    "The first paragraph of the material is the paper's own abstract; use it to "
    "frame the section and never attach a citation marker to it."
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

#: Sentence length window for the sentence-level miners. Below the floor a
#: "sentence" is a fragment (a header, a page number, half a caption); above the
#: ceiling it is usually two run-together columns of a two-column PDF.
_MIN_SENTENCE_CHARS = 40
_MAX_SENTENCE_CHARS = 240

#: Keyword sets that pick each :data:`~deepvision.models.report.AT_A_GLANCE_FIELDS`
#: line out of the retrieved sentences, keyed by the field's label. Purely
#: deterministic, so the card is grounded even with no model available. Multi-word
#: phrases are weighted double because they are far less likely to fire by accident.
_GLANCE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Problem": (
        "problem", "challenge", "challenging", "limitation", "difficult",
        "bottleneck", "suffers from", "fails to", "hard to", "drawback",
        "unable to", "issue", "shortcoming", "not accurate", "error-prone",
    ),
    "Approach": (
        "we propose", "we present", "we introduce", "we design", "we develop",
        "this paper proposes", "this paper presents", "propose", "present a",
        "introduce", "our approach", "our method", "our system", "we use",
        "based on", "framework", "algorithm",
        # Papers state their contribution in the passive at least as often as
        # in the first person — an abstract reading "The system presented here
        # is a vision-based system for…" scored zero on the list above and left
        # the Approach line empty on a paper that plainly states its approach.
        "presented here", "designed here", "is designed", "we build",
        "we implement", "implemented", "system for", "method for", "approach to",
        "vision-based", "-based system", "a new way", "novel",
    ),
    "Evidence": (
        "dataset", "database", "benchmark", "evaluated on", "experiments on",
        "we evaluate", "test set", "training set", "corpus", "samples",
        "subjects", "images", "experimental setup", "we conducted", "trials",
        "camera", "cameras", "webcam", "hardware", "resolution",
    ),
    "Headline result": (
        "achieves", "achieved", "accuracy", "outperform", "reduces", "reduced",
        "improves by", "improvement of", "percent", "%", "precision", "recall",
        "f1", "error rate", "speedup", "state-of-the-art", "success rate",
        # Throughput IS the headline number for a real-time system. These used
        # to live under Evidence, where they made the setup line steal the
        # paper's actual result ("runs at 30 frames per second").
        "frames per second", "fps", "runs at", "operates at", "real time",
    ),
    "Read this if": (
        "useful", "practical", "application", "applications", "can be used",
        "enables", "suitable for", "real-time", "real world", "real-world",
        "deploy", "developers", "researchers", "practitioners", "industry",
        "human computer interface", "assistive",
    ),
}

#: Sentences that signal the authors talking about what their work does NOT do.
#: Drives the deterministic Limitations draft — see :func:`_limitation_draft_parts`.
#:
#: These are matched as substrings, so every entry has to be *specific*: a bare
#: "constrained" fires on "unconstrained environments" and turns the paper's
#: opening problem statement into a fabricated caveat, which is exactly the
#: failure this section must not have. Prefer multi-word phrases.
_LIMITATION_MARKERS: tuple[str, ...] = (
    "limitation", "however,", "future work", "in future", "future research",
    "does not", "do not handle", "cannot", "can not", "fails when", "fail to",
    "we assume", "assumption", "is constrained by", "restricted to",
    "only works", "remains an open", "open question", "drawback",
    "not addressed", "beyond the scope", "left for future", "yet to be",
)

#: Sentences that describe the *scope* of what was tested. Used only when no
#: limitation marker fires: they let the section state honestly that no
#: limitations were retrieved and then point at what was actually covered, which
#: is the only grounded way to name what is visibly untested.
_SCOPE_MARKERS: tuple[str, ...] = (
    "dataset", "database", "images", "evaluated", "experiment", "tested on",
    "we test", "samples", "subjects", "environment", "condition", "camera",
    "resolution", "illumination", "lighting", "we use", "benchmark",
)

#: Shipped verbatim (and it must read as shipped prose, because on a degraded
#: run it IS the section) when the retrieved material states no limitations.
_NO_LIMITATIONS_LEAD = (
    "The retrieved passages do not state the work's limitations directly. "
    "What they do show is the scope of what was actually tested:"
)
_NO_LIMITATIONS_ONLY = (
    "The retrieved passages do not state the work's limitations directly, and "
    "they do not describe the evaluation in enough detail to say what was left "
    "untested."
)

#: Emitted for an At a Glance field with no grounded candidate. Deliberately
#: carries no ``[n]`` marker — there is nothing to cite, and inventing a value
#: here would be the single most damaging hallucination in the report.
_GLANCE_UNKNOWN = "Not stated in the retrieved passages."


# --------------------------------------------------------------------------
# Deterministic draft helpers
#
# Everything below produces text that SHIPS as-is when the LLM is unavailable.
# Nothing here may emit an instruction, a label addressed to the model, or a
# claim the chunks do not support.
# --------------------------------------------------------------------------
def _sentences(text: str) -> list[str]:
    """Split ``text`` into candidate sentences (whitespace-normalized)."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]


def _sentence_candidates(chunks: Sequence[AnyChunk]) -> list[tuple[str, int]]:
    """``(sentence, marker)`` pairs mined from ``chunks``, quality-filtered.

    ``marker`` is the 1-based position of the source chunk in ``chunks``, i.e.
    exactly the number to write as ``[marker]`` so
    :class:`~deepvision.agents.citation_agent.CitationAgent` resolves it against
    the same list passed to :meth:`SummarizerAgent._write`.

    Caption text and journal boilerplate are dropped at *sentence* granularity
    here, on top of the chunk-level :func:`narrative_chunks` filter applied in
    :meth:`SummarizerAgent.run` — a mostly-prose chunk can still carry a running
    header or an "AUTHORS PROFILE" line in the middle of it.
    """
    out: list[tuple[str, int]] = []
    for marker, chunk in enumerate(chunks, start=1):
        for raw in _sentences((chunk.text or "").strip()):
            sentence = clean_snippet(raw, max_chars=_MAX_SENTENCE_CHARS)
            if len(sentence) < _MIN_SENTENCE_CHARS:
                continue
            if is_caption(sentence) or is_boilerplate(sentence):
                continue
            out.append((sentence, marker))
    return out


def _score(sentence: str, keywords: Sequence[str], *, prefer_digit: bool = False) -> int:
    """Keyword score for ``sentence``; multi-word phrases count double."""
    low = sentence.lower()
    total = 0
    for keyword in keywords:
        hits = low.count(keyword)
        if hits:
            total += hits * (2 if " " in keyword else 1)
    if prefer_digit and total and any(ch.isdigit() for ch in sentence):
        total += 3
    return total


def _glance_draft_parts(
    chunks: Sequence[AnyChunk], *, abstract: Optional[str] = None
) -> list[str]:
    """The five ``**Label** — sentence [n]`` lines of the At a Glance card.

    One line per entry of :data:`~deepvision.models.report.AT_A_GLANCE_FIELDS`,
    in that order, always all five. Each field takes the highest-scoring
    *unused* sentence for its own keyword set (:data:`_GLANCE_KEYWORDS`), so two
    fields never share a sentence while another candidate exists; "Headline
    result" additionally prefers a sentence containing a digit. A field with no
    scoring candidate gets :data:`_GLANCE_UNKNOWN` and no citation marker —
    never an invented value.

    ``abstract``, when given (whole-paper reports only — a chapter report must
    not reach for page 1), contributes its sentences as extra candidates. That
    matters a lot: the abstract is the paper's own one-paragraph statement of
    problem, approach, setup and result, which is four of this card's five
    fields. Abstract sentences carry **no citation marker** — they are the
    paper's framing rather than a retrieved passage, exactly as in the Overview
    — so they are stored with marker ``0`` and rendered without ``[n]``.

    Because the labels themselves are deterministic, a fully degraded run still
    ships a correctly shaped fact card.
    """
    candidates = _sentence_candidates(chunks)
    if abstract:
        for raw in _sentences(abstract):
            sentence = clean_snippet(raw, max_chars=_MAX_SENTENCE_CHARS)
            if len(sentence) >= _MIN_SENTENCE_CHARS:
                candidates.append((sentence, 0))

    # Assign strongest-match-first, NOT in display order. One sentence often
    # scores for two fields, and whichever field is asked first wins it under a
    # naive in-order loop — which is how "The system … runs at 30 frames per
    # second" landed on Evidence (one weak keyword hit) while Headline result,
    # the field it actually answers with a much higher score, printed "Not
    # stated". Highest confidence claims the sentence.
    best_by_label: dict[str, tuple[int, int]] = {}
    for label, _brief in AT_A_GLANCE_FIELDS:
        keywords = _GLANCE_KEYWORDS.get(label, ())
        prefer_digit = label == "Headline result"
        scored = [
            (_score(sentence, keywords, prefer_digit=prefer_digit), index)
            for index, (sentence, _marker) in enumerate(candidates)
        ]
        top = max(scored, default=(0, -1))
        if top[0] > 0:
            best_by_label[label] = top

    chosen: dict[str, int] = {}
    used: set[int] = set()
    for label in sorted(best_by_label, key=lambda k: -best_by_label[k][0]):
        keywords = _GLANCE_KEYWORDS.get(label, ())
        prefer_digit = label == "Headline result"
        best_index, best_score = -1, 0
        for index, (sentence, _marker) in enumerate(candidates):
            if index in used:
                continue
            score = _score(sentence, keywords, prefer_digit=prefer_digit)
            if score > best_score:
                best_score, best_index = score, index
        if best_index >= 0:
            chosen[label] = best_index
            used.add(best_index)

    parts: list[str] = []
    for label, _brief in AT_A_GLANCE_FIELDS:
        index = chosen.get(label)
        if index is None:
            parts.append(f"**{label}** — {_GLANCE_UNKNOWN}")
            continue
        sentence, marker = candidates[index]
        # marker 0 == an abstract sentence: framing, never cited.
        suffix = f" [{marker}]" if marker else ""
        parts.append(f"**{label}** — {sentence}{suffix}")
    return parts


def _concept_draft_parts(chunks: Sequence[AnyChunk], count: int) -> list[str]:
    """Deterministic ``**Term** — explanation [n]`` paragraphs for Key Concepts.

    :func:`~deepvision.agents.study_agent.extract_terms` is the vocabulary miner
    that used to feed the deleted Glossary section; Key Concepts owns it now.
    The result is the *draft* the LLM polishes, and it matters on its own
    because the draft is exactly what ships when the model is unavailable
    (``complete_with_fallback``), so the section keeps its promised term-led
    shape even fully degraded.

    Entries whose defining sentence is journal furniture ("AUTHORS PROFILE She
    has completed her Bachelor of engineering…", ISSN/IJCSIS running headers) or
    a figure caption are dropped rather than defined — that is where the old
    Glossary's worst entries came from.
    """
    parts: list[str] = []
    kept: list[str] = []
    for term, sentence, idx in extract_terms(chunks, limit=count * 3):
        if len(parts) >= count:
            break
        if is_caption(sentence) or is_boilerplate(sentence):
            continue
        # The miner can surface both an acronym entry and its spelled-out form
        # ("HCI (Human Computer Interface)" and "Human Computer Interface"), and
        # defining the same idea twice is precisely the Key Concepts/Glossary
        # duplication this redesign removes.
        low = term.lower()
        if any(low in other or other in low for other in kept):
            continue
        kept.append(low)
        parts.append(f"**{term}** — {sentence} [{idx}]")
    return parts


def _lead_phrase(sentence: str, max_words: int = 6) -> str:
    """First few words of ``sentence``, used as a bold lead-in for a takeaway."""
    words = (sentence or "").split()
    lead = " ".join(words[:max_words]).strip(" ,;:.-—")
    return lead or (sentence or "")[:40]


def _takeaway_draft_parts(chunks: Sequence[AnyChunk], count: int) -> list[str]:
    """Deterministic ``**Lead-in** — claim [n]`` paragraphs for Key Takeaways.

    Same role as :func:`_concept_draft_parts`: this is exactly what ships when
    the LLM is unavailable, so even fully degraded the section already reads as
    a list of standalone bold-led claims rather than a dump of raw snippets.
    Sentences that carry a number are preferred, because a recap made of
    concrete claims is the one worth remembering.
    """
    candidates = _sentence_candidates(chunks)
    # Stable sort: keeps retrieval order among equally-numeric sentences, so the
    # draft is deterministic for a given chunk list.
    ranked = sorted(
        candidates, key=lambda pair: 0 if any(c.isdigit() for c in pair[0]) else 1
    )
    parts: list[str] = []
    seen: set[str] = set()
    for sentence, marker in ranked:
        if len(parts) >= count:
            break
        key = sentence.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"**{_lead_phrase(sentence)}** — {sentence} [{marker}]")
    return parts


def _limitation_draft_parts(chunks: Sequence[AnyChunk], count: int) -> list[str]:
    """Deterministic draft for Limitations & Open Questions.

    This is the section most likely to hallucinate, so the draft is built only
    from sentences that actually contain a limitation marker
    (:data:`_LIMITATION_MARKERS`). When none do, the draft says so in plain
    words and then lists the *scope* sentences (:data:`_SCOPE_MARKERS`) — what
    was tested — which is the only grounded basis for naming what is visibly
    untested. When even that is missing, the section is one honest sentence.
    Nothing here ever asserts a limitation the material does not contain.
    """
    candidates = _sentence_candidates(chunks)
    stated = [
        (sentence, marker)
        for sentence, marker in candidates
        if _score(sentence, _LIMITATION_MARKERS)
    ]
    if stated:
        parts: list[str] = []
        seen: set[str] = set()
        for sentence, marker in stated:
            if len(parts) >= count:
                break
            key = sentence.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            parts.append(f"{sentence} [{marker}]")
        return parts

    scope = [
        (sentence, marker)
        for sentence, marker in candidates
        if _score(sentence, _SCOPE_MARKERS)
    ]
    if not scope:
        return [_NO_LIMITATIONS_ONLY]
    parts = [_NO_LIMITATIONS_LEAD]
    for sentence, marker in scope[: max(1, count - 1)]:
        parts.append(f"{sentence} [{marker}]")
    return parts


def _confidence_for(chunks: Sequence[AnyChunk]) -> float:
    if not chunks:
        return 0.2
    text_like = sum(1 for c in chunks if c.modality == Modality.TEXT)
    ocr_like = sum(1 for c in chunks if c.modality == Modality.OCR)
    base = 0.55 + 0.05 * min(len(chunks), 6)
    base += 0.1 * (text_like / len(chunks))
    base -= 0.05 * (ocr_like / len(chunks))
    ocr_confidences = [
        getattr(c, "ocr_confidence", None) for c in chunks
        if getattr(c, "ocr_confidence", None) is not None
    ]
    if ocr_confidences:
        base = 0.6 * base + 0.4 * (sum(ocr_confidences) / len(ocr_confidences))
    return round(max(0.1, min(base, 0.97)), 2)


# --------------------------------------------------------------------------
# Anti-redundancy prompt blocks
# --------------------------------------------------------------------------
def _own_question_rule(name: SectionName) -> str:
    """The section's own question, as the opening line of its system prompt."""
    question = SECTION_QUESTIONS.get(name)
    if not question:
        return ""
    return (
        f'This section answers exactly one question: "{question}" Everything '
        "you write must serve that question and nothing else. "
    )


def _elsewhere_rule(name: SectionName) -> str:
    """The other ten sections' questions, as a "handled elsewhere" list.

    Built from all eleven names in
    :data:`~deepvision.models.report.SECTION_QUESTIONS`, not just the nine this
    agent writes — Figures and Study Questions are just as capable of being
    duplicated into a prose section as any other.
    """
    others = [
        f"\"{question}\" belongs to '{other.value}'"
        for other, question in SECTION_QUESTIONS.items()
        if other is not name
    ]
    if not others:
        return ""
    return (
        "ANSWERED ELSEWHERE — do not answer these here: " + "; ".join(others) + ". "
    )


def _digest_rule(digest: Sequence[tuple[SectionName, str]]) -> str:
    """What earlier sections in this run already said, for the system prompt.

    Costs input tokens only — no extra model call. Capped at
    :data:`_DIGEST_MAX` entries so it cannot grow without bound if the section
    set ever does.
    """
    if not digest:
        return ""
    lines = "\n".join(
        f"- {name.value}: {text}" for name, text in list(digest)[-_DIGEST_MAX:]
    )
    return (
        "ALREADY COVERED IN EARLIER SECTIONS — do not repeat these points; add "
        f"what is new:\n{lines}\n"
    )


class SummarizerAgent(Agent[list[Section]]):
    """Generates the nine narrative report sections for a paper."""

    def __init__(self, llm: LLMProvider, *, citation_agent: Optional[CitationAgent] = None) -> None:
        super().__init__(llm)
        self._citation_agent = citation_agent or CitationAgent()

    def run(self, context: AgentContext) -> list[Section]:
        """Return the nine prose :class:`Section`s, in :data:`_GENERATION_ORDER`.

        Figures belong to ``MediaAgent``; Study Questions to ``StudyAgent``.
        Prefers ``context.extra["outline"]`` (a ``SectionName -> chunks`` map,
        typically from ``ResearchAgent.plan_outline``); if absent, falls back to
        a lightweight keyword bucketing of ``context.chunks`` so this agent
        still produces sensible output standalone.

        The returned list is in *generation* order, which is not display order —
        ``report.interactive_sections.normalize_sections`` reorders it into
        ``SECTION_ORDER`` and backfills the two sections this agent does not
        own. Nothing downstream may assume this list's order.
        """
        outline = context.extra.get("outline") or {}
        # Chunk-level quality gate, applied once to the standalone pool and
        # again per section below: captions and journal furniture must never
        # reach a draft, because the draft is what ships when the LLM falls back.
        #
        # `allow_fallback=False` is the important half. With the default, a
        # section whose retrieved chunks are ALL captions gets those captions
        # handed back — which is how a real report shipped a `Why It Matters`
        # section reading, in full, "Fig 9 — vision analysis unavailable;
        # showing the extracted crop only." An empty list is the honest answer,
        # and `_draft` turns it into a plain "nothing usable was retrieved" note.
        pool = narrative_chunks(list(context.chunks), allow_fallback=False)
        sections: list[Section] = []

        detail = getattr(context.settings, "report_detail", ReportDetail.STANDARD)
        params = _DETAIL_PARAMS.get(detail, _DETAIL_PARAMS[ReportDetail.STANDARD])

        # The paper's own abstract is the most accurate seed for the Overview —
        # but ONLY for a whole-paper report. When `extra["page_range"]` is set
        # the caller is building a per-chapter report, and the abstract lives on
        # page 1, outside almost every chapter: framing "3 Methods" with the
        # paper's abstract is exactly the out-of-chapter text that per-chapter
        # scoping promises never to substitute. Scoped runs use only the
        # retrieved in-chapter chunks.
        page_range = context.extra.get("page_range")
        if page_range is None:
            meta = load_paper_meta(context.paper_id)
            abstract = clean_snippet(
                getattr(meta, "abstract", "") or "", _ABSTRACT_LEAD_CHARS
            )
        else:
            abstract = ""

        # The running anti-redundancy digest: (section, first ~200 chars of its
        # body), appended as each section finishes and fed into every later
        # section's system prompt. This is why generation order matters — Key
        # Takeaways is last so its digest covers the whole report.
        digest: list[tuple[SectionName, str]] = []

        for name in _GENERATION_ORDER:
            section_chunks = narrative_chunks(
                list(outline.get(name) or []), allow_fallback=False
            )
            # Always top up from the cleaned pool rather than only when the
            # section's own group is empty. A group can come back non-empty and
            # still be useless — Background on 1002-2191 retrieved exactly two
            # chunks, both OCR noise off a rotated axis label, and with a
            # "top up only if empty" rule that noise WAS the section. Appending
            # (deduplicated, after the section's own hits, so relevance order is
            # preserved) lets `demote_low_prose` sink the noise below real prose
            # and the per-level cap then cuts it off.
            existing = {c.id for c in section_chunks}
            section_chunks = section_chunks + [
                c for c in self._fallback_bucket(name, pool) if c.id not in existing
            ]
            section_chunks = demote_low_prose(section_chunks)
            cap = params["max_chunks"] + _WIDE_NET_BONUS.get(name, 0)
            section_chunks = section_chunks[:cap]

            # Overview is framed by the abstract, and so is At a Glance: the
            # abstract is by construction the paper's own one-paragraph answer
            # to Problem / Approach / Evidence / Headline result, which is four
            # of the card's five fields. Without it the card degrades to five
            # "Not stated in the retrieved passages." lines on a paper whose
            # abstract says all of it — observed on 1002-2191.
            lead = (
                abstract
                if name in (SectionName.OVERVIEW, SectionName.AT_A_GLANCE)
                else None
            )
            body, deep_dive, degraded = self._write(
                name, section_chunks, params, lead=lead, digest=digest
            )
            combined = body + ("\n\n" + deep_dive if deep_dive else "")
            citations = self._citation_agent.cite(combined, section_chunks) if section_chunks else []

            sections.append(
                Section(
                    id=section_id(),
                    name=name,
                    question=SECTION_QUESTIONS.get(name),
                    body_markdown=body,
                    deep_dive_markdown=deep_dive,
                    badge=_SECTION_BADGE.get(name),
                    provenance=provenance_list(section_chunks),
                    citations=citations,
                    media=[],
                    confidence=_confidence_for(section_chunks),
                    degraded=degraded,
                    default_open=(name in _DEFAULT_OPEN),
                )
            )
            digest.append((name, clean_snippet(body, max_chars=_DIGEST_CHARS)))

        return sections

    @staticmethod
    def _fallback_bucket(name: SectionName, chunks: Sequence[AnyChunk]) -> list[AnyChunk]:
        keywords = _SECTION_KEYWORDS.get(name, ())
        scored: list[tuple[int, AnyChunk]] = []
        for chunk in chunks:
            if chunk.modality not in (Modality.TEXT, Modality.OCR):
                continue
            text_lower = (chunk.text or "").lower()
            score = sum(text_lower.count(k) for k in keywords)
            if score:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        picked = [c for _, c in scored[:_MAX_CHUNKS_PER_SECTION]]
        if not picked:
            picked = [c for c in chunks if c.modality in (Modality.TEXT, Modality.OCR)]
            picked = picked[:_MAX_CHUNKS_PER_SECTION]
        return picked

    # ---- Drafting ----------------------------------------------------

    @staticmethod
    def _draft(
        name: SectionName,
        chunks: Sequence[AnyChunk],
        params: dict,
        *,
        lead: Optional[str],
    ) -> tuple[str, Optional[str]]:
        """Return ``(body_draft, deep_draft)`` — both shippable prose as-is.

        Four sections replace the generic "snippet [n]" draft with a shaped one
        so their contract survives full degradation: At a Glance (five labelled
        lines), Key Concepts (term-led paragraphs), Key Takeaways (bold-led
        standalone claims) and Limitations (grounded caveats, or an honest
        statement that none were retrieved).
        """
        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            snippet = clean_snippet(chunk.text, max_chars=340)
            if snippet:
                parts.append(f"{snippet} [{i}]")

        if not parts and not lead:
            return "", None

        body_parts = parts[: params["body_parts"]]
        deep_parts = parts[params["body_parts"]: params["body_parts"] + params["deep_parts"]]

        if name is SectionName.AT_A_GLANCE:
            # Fixed five-line schema, no deep dive: a fact card has nothing to
            # defer behind a toggle, and a sixth line would break the shape.
            body_parts = _glance_draft_parts(chunks, abstract=lead)
            deep_parts = []
            # The abstract is a *source* for this card, not a lead paragraph to
            # print above it — clear it so the generic lead handling below does
            # not prepend the whole abstract to a five-line fact card.
            lead = None
        elif name is SectionName.KEY_CONCEPTS:
            # Anchor the draft on real mined terms so the LLM-less fallback
            # already has the "**Term** — explanation" shape the section
            # contract promises. Concepts beyond the level's target become the
            # deep dive, which keeps that toggle coherent with the section
            # instead of dumping unrelated passages into it.
            count = params["concept_count"]
            concepts = _concept_draft_parts(chunks, count * 2)
            if concepts:
                body_parts = concepts[:count]
                deep_parts = concepts[count:]
        elif name is SectionName.KEY_TAKEAWAYS:
            # Same reasoning as Key Concepts. No deep dive — this section is the
            # recap, so there is nothing left to hold back.
            body_parts = _takeaway_draft_parts(chunks, params["takeaway_count"])
            deep_parts = []
        elif name is SectionName.LIMITATIONS:
            # No deep dive: everything grounded that bears on "what this does
            # not establish" belongs in the body, and padding the toggle with
            # unrelated passages is how a caveats section starts inventing.
            body_parts = _limitation_draft_parts(chunks, params["body_parts"])
            deep_parts = []

        draft_pieces: list[str] = []
        if lead:
            # Bare abstract text, deliberately unlabelled. The "this is the
            # abstract, do not cite it" instruction lives in the system prompt
            # (_ABSTRACT_RULE) — putting it here once shipped the literal string
            # "Paper abstract (context, not to be cited):" to readers.
            draft_pieces.append(lead)
        draft_pieces.extend(body_parts)

        body_draft = "\n\n".join(p for p in draft_pieces if p)
        deep_draft = "\n\n".join(deep_parts) if deep_parts else None
        return body_draft, deep_draft

    # ---- Prompts -----------------------------------------------------

    @staticmethod
    def _body_system(
        name: SectionName,
        params: dict,
        *,
        has_lead: bool,
        digest: Sequence[tuple[SectionName, str]],
    ) -> str:
        """System prompt for a section body.

        Order matters: the section's own question first (what to write), then
        the brief (how), then the anti-redundancy blocks (what NOT to write),
        then the shared citation/format/length rules.
        """
        if name is SectionName.AT_A_GLANCE:
            target = _GLANCE_INSTRUCTION
            length_rule = _GLANCE_LENGTH_RULE
        elif name is SectionName.KEY_CONCEPTS:
            target = _CONCEPT_INSTRUCTION.format(count=params["concept_count"])
            length_rule = (
                f"HARD COUNT LIMIT: exactly {params['concept_count']} concepts, "
                "no more and no fewer, each its own bold-term paragraph."
            )
        elif name is SectionName.KEY_TAKEAWAYS:
            target = _TAKEAWAY_INSTRUCTION.format(count=params["takeaway_count"])
            length_rule = (
                f"HARD COUNT LIMIT: exactly {params['takeaway_count']} takeaways, "
                "no more and no fewer, each its own bold-led standalone sentence."
            )
        else:
            target = params["target"]
            length_rule = params["length_rule"]

        abstract_rule = f"{_ABSTRACT_RULE} " if has_lead else ""
        return (
            f"You are writing the '{name.value}' section of a study guide for a "
            "research paper. The reader is intelligent but has NO background in "
            f"this field. {_own_question_rule(name)}"
            f"{_SECTION_BRIEF.get(name, '')} "
            f"{_elsewhere_rule(name)}"
            f"{_digest_rule(digest)}"
            f"Rewrite the material below into {target}. "
            "Preserve every '[n]' citation marker exactly where it grounds a "
            "claim — never add, remove, or renumber markers. "
            f"{abstract_rule}"
            "Never state anything the material does not support. "
            f"{_FORMAT_RULE} {length_rule}"
        )

    @staticmethod
    def _deep_system(name: SectionName, params: dict) -> str:
        """System prompt for a section's "deep dive" expansion."""
        if name is SectionName.KEY_CONCEPTS:
            deep_target = (
                "further concepts, each as its own paragraph beginning with the "
                "term in bold followed by an em dash, a plain-language "
                "definition and one sentence on why it matters in this paper, "
                "exactly like the main section"
            )
            deep_length_rule = (
                "HARD RULE: keep one paragraph per concept and do not merge or "
                "drop any of them."
            )
        else:
            deep_target = params["deep_target"]
            deep_length_rule = params["deep_length_rule"]
        return (
            f"You are writing the extended 'deep dive' detail for the "
            f"'{name.value}' section of a study guide, for a reader with no "
            f"background in the field. {_own_question_rule(name)}"
            f"{_elsewhere_rule(name)}"
            f"Rewrite the material below into {deep_target}, adding depth the "
            "main section left out, preserving every '[n]' citation marker "
            f"exactly and inventing nothing. {_FORMAT_RULE} {deep_length_rule}"
        )

    # ---- Generation --------------------------------------------------

    def _write(
        self,
        name: SectionName,
        chunks: Sequence[AnyChunk],
        params: dict,
        *,
        lead: Optional[str] = None,
        digest: Sequence[tuple[SectionName, str]] = (),
    ) -> tuple[str, Optional[str], bool]:
        """Return ``(body, deep_dive, degraded)`` for one section.

        ``degraded`` is measured, not guessed: ``agents.base`` keeps a
        thread-local tally of LLM calls that silently fell back to their
        deterministic draft, so snapshotting :func:`llm_fallback_count` around
        *this section's* calls tells us whether this section's text is model
        prose or raw PDF extract. That is what
        :attr:`~deepvision.models.report.Section.degraded` shows the reader, and
        it is strictly finer-grained than the whole-run counter the report stage
        reports.
        """
        body_draft, deep_draft = self._draft(name, chunks, params, lead=lead)
        if not body_draft:
            # Honest, user-facing, and not a scaffold string: nothing usable was
            # retrieved. Reached either because retrieval returned nothing or
            # because everything it returned was figure captions / page
            # furniture, which `narrative_chunks(allow_fallback=False)` removes
            # rather than hand back. Saying so is the contract; padding the
            # section with captions is the bug this replaced. Not degraded — no
            # LLM call was attempted, so there was no fallback to report.
            return (
                f"No usable prose was retrieved for **{name.value}**. The "
                "passages found for this section were figures, captions or page "
                "furniture rather than text, so nothing is shown here instead of "
                "filling the section with material that does not answer it.",
                None,
                False,
            )

        before = llm_fallback_count()

        body = self._complete(
            self._body_system(name, params, has_lead=bool(lead), digest=digest),
            body_draft,
            temperature=0.3,
            max_tokens=params["max_tokens"],
        )

        deep_dive = None
        if deep_draft:
            deep_dive = self._complete(
                self._deep_system(name, params),
                deep_draft,
                temperature=0.3,
                max_tokens=params["max_tokens"],
            )

        # A marker the model re-styled ("[n] 3", "**(1)**") stops resolving to
        # its citation and the popover silently never opens, so normalise before
        # CitationAgent is asked to ground the text against these same chunks.
        markers = len(chunks)
        body = repair_citation_markers(body, markers)
        if deep_dive:
            deep_dive = repair_citation_markers(deep_dive, markers)

        return body, deep_dive, llm_fallback_count() > before
