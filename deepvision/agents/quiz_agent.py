"""Quiz agent — builds a mixed, cited, difficulty-tiered quiz from a report.

Owned by the QUIZ builder. The agent's primary input is the paper's **persisted
report sections** (whole-paper, or one chapter's report when the quiz is
chapter-scoped) — the report is already grounded, already carries resolved
:class:`~deepvision.models.report.Citation` objects with page numbers and chunk
ids, and is the text the user actually studied.

It also optionally takes ``extra_chunks``: the paper's whole-chunk **source
pool** (:func:`~deepvision.study.source_pool.pool_for_paper`), which supplies
the roughly two thirds of every paper the report never quotes. Without it the
generator is exactly as before — report-only, so an empty/omitted pool is a
strict degrade, never a failure. With it, :func:`collect_sources` mines
citable claim sentences straight out of the paper's own text (page/chunk id
taken off the chunk, section guessed heuristically since a raw chunk carries
none), filtered through
:func:`~deepvision.rag.chunk_quality.is_boilerplate`/:func:`~deepvision.rag.chunk_quality.prose_score`
so reference-list and page-furniture text never becomes a question. This is
what makes two consecutive quizzes on the same paper actually differ instead of
reshuffling the same handful of report-summarised claims — see
``avoid_prompts`` below and :mod:`deepvision.study.source_pool`.

Shape of the output (all enforced, never assumed):

- a **mix of kinds** — ``multiple_choice`` (exactly 4 options, exactly one
  correct), ``true_false`` (the fixed ``true``/``false`` pair) and
  ``short_answer``;
- a **mix of difficulty tiers** — ``recall`` / ``understand`` / ``apply``;
- a non-empty ``explanation`` on every question saying *why* the right answer is
  right and, for multiple choice, why the tempting distractor is wrong;
- a :class:`~deepvision.models.study.QuizCitation` on every question, pointing at
  the section (and page, when the report knew one) the answer comes from. A
  question that cannot be cited is **dropped**, never emitted with a null
  citation.

How generation degrades — the project's hard rule, applied here:

1. A deterministic draft quiz is built first, straight from the report text
   (glossary/key-concept definitions, and claim sentences containing a number).
   Distractors are *this paper's other definitions*, so they are plausible by
   construction rather than filler; the deterministic set never contains "all of
   the above"/"none of the above" and the generated set is rejected if it does.
2. That draft is handed to the LLM through
   :func:`~deepvision.agents.base.complete_with_fallback` as the last user
   message, with the source material in the system prompt. A real model returns
   a better, wider quiz; ``EchoLLM`` echoes the draft back verbatim; a provider
   error returns the draft unchanged. All three parse.
3. Whatever comes back is validated **question by question** (see
   :func:`validate_draft`) and anything that fails is dropped, then the shortfall
   is topped up from the deterministic draft. So a hallucinating model can shrink
   the quiz but cannot corrupt it, and a dead model still yields a real quiz.

``avoid_prompts`` — the prompt text of this paper's earlier quizzes — is applied
at draft-selection time (:func:`_select_drafts`), before the LLM ever sees the
draft: a candidate whose prompt was already asked is skipped while unseen
material remains, and only reused once the pool genuinely runs dry, so a
regenerated quiz never comes back short because everything was excluded.

Nothing in this module raises: :meth:`QuizAgent.generate` returns a
:class:`~deepvision.models.study.Quiz` — possibly an empty one, if the report
itself has no usable text — and the caller turns that into an honest job result.

**Answers never leave here in a client-facing shape.** This module produces
:class:`~deepvision.models.study.QuizQuestion` (answer key included); the router
is responsible for projecting it to ``QuizQuestionPublic``.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from deepvision.agents.base import (
    Agent,
    AgentContext,
    clean_snippet,
    complete_with_fallback,
    safe_json,
)
from deepvision.models.chunks import AnyChunk
from deepvision.models.report import Citation, Section, SectionName
from deepvision.models.settings import AppSettings
from deepvision.models.study import (
    Difficulty,
    QuestionKind,
    Quiz,
    QuizCitation,
    QuizOption,
    QuizQuestion,
)
from deepvision.providers.base import LLMProvider
from deepvision.rag.chunk_quality import (
    is_boilerplate,
    is_caption_sentence,
    is_ordinal_number,
    prose_score,
)
from deepvision.study.grading import punctuation_insensitive, token_overlap
from deepvision.utils.ids import new_id
from deepvision.utils.logger import get_logger

__all__ = [
    "QuizAgent",
    "SourceItem",
    "collect_sources",
    "validate_draft",
    "MCQ_OPTION_IDS",
    "TRUE_FALSE_OPTIONS",
]

log = get_logger(__name__)

#: Option ids for a multiple-choice question, in display order. Exactly four —
#: the contract in :class:`~deepvision.models.study.QuestionKind`.
MCQ_OPTION_IDS: tuple[str, ...] = ("a", "b", "c", "d")

#: The fixed true/false pair, in the fixed order the UI renders as two buttons.
TRUE_FALSE_OPTIONS: tuple[tuple[str, str], ...] = (("true", "True"), ("false", "False"))

#: Banned option texts — they turn a knowledge question into a test-taking
#: heuristic, and they make "exactly one correct option" ambiguous.
_BANNED_OPTION_RE = re.compile(
    r"^\s*(all|none|both|either|neither)\s+of\s+(the\s+)?(above|these|them)\b", re.I
)

#: ``**Term** — definition`` / ``**Term**: definition``, optionally inside a
#: markdown bullet or numbered item. This is the body convention the Glossary and
#: Key Concepts sections are written to (see ``models/report.py::Section``).
_DEF_RE = re.compile(
    r"^\s*(?:[-*+]\s+|\d+[.)]\s+)?"
    r"\*\*(?P<term>[^*\n]{2,60}?)\*\*"
    r"\s*(?:[—–\-:]|--)\s*"
    r"(?P<body>\S.*)$"
)
#: An inline citation marker, e.g. ``[3]``.
_MARKER_RE = re.compile(r"\[(\d{1,3})\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
#: A standalone number, optionally decimal, optionally a percentage.
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)(?![\w])")
#: ``Acronym (Spelled Out)`` or ``Spelled Out (Acronym)`` in a term label.
_PARENTHETICAL_RE = re.compile(r"^(?P<head>[^()]+?)\s*\((?P<tail>[^()]+)\)\s*$")
_WS_RE = re.compile(r"\s+")

#: Sections mined for term/definition pairs, in preference order. Key Concepts
#: is the sole vocabulary section now — it absorbed the former Glossary (see
#: ``deepvision.models.report.SectionName``), which produced near-identical
#: term/definition pairs for the same terms.
_DEFINITION_SECTIONS: tuple[SectionName, ...] = (SectionName.KEY_CONCEPTS,)
#: Sections mined for factual claim sentences, in preference order. Figures is
#: excluded (its body is caption scaffolding, not prose claims). The former
#: catch-all ``Conclusions`` section is gone, replaced here by its two
#: successors: ``Why It Matters`` (what the results mean) and
#: ``Limitations & Open Questions`` (what is unresolved) — so "what does it
#: mean" and "what is unresolved" claims are still mined, just from the
#: section that actually owns that question now.
_CLAIM_SECTIONS: tuple[SectionName, ...] = (
    SectionName.KEY_TAKEAWAYS,
    SectionName.KEY_RESULTS,
    SectionName.METHODS,
    SectionName.WHY_IT_MATTERS,
    SectionName.LIMITATIONS,
    SectionName.OVERVIEW,
    SectionName.BACKGROUND,
)

#: Hard caps so one enormous report/paper cannot blow up the prompt. Raised from
#: the report-only baseline (28) now that ``extra_chunks`` (the whole-paper
#: source pool) is a second contributor — a cap sized only for the report would
#: let report items fill every slot before a single pool sentence is ever seen.
_MAX_SOURCES = 60
_MAX_DEFINITIONS = 18
_MAX_CLAIMS = 16
#: Sentences mined out of the whole-paper source pool (``extra_chunks``), as
#: opposed to the report's own prose.
_MAX_POOL_CLAIMS = 30
#: ...and at most this many from any single chunk. Without a per-chunk cap the
#: global budget above is a prefix, not a sample: a chunk yields a dozen usable
#: sentences, so measured on this library **2 of 10 pool chunks filled the
#: entire 30-claim budget** and the other 80% of the whole-paper pool never
#: reached the quiz at all. That silently undid most of what `source_pool`
#: exists to provide, and it is why `_ordering_questions` looked dead: sequence
#: markers ("First, ... Then, ...") cluster in methods prose late in a paper,
#: and 11 such sentences in one paper's pool produced exactly 1 source.
_MAX_CLAIMS_PER_POOL_CHUNK = 3
_SNIPPET_CHARS = 240
_MIN_DEFINITION_CHARS = 30
_MIN_CLAIM_CHARS = 45
_MAX_CLAIM_CHARS = 320
#: A pool sentence below this :func:`~deepvision.rag.chunk_quality.prose_score`
#: reads as OCR noise or a mangled reference-list entry (the observed failure:
#: a quiz question that quoted "In Proceed- ings of the IE..." verbatim), not
#: paper prose worth asking a question about.
_MIN_POOL_PROSE_SCORE = 0.2

#: Heuristic keyword sets used to guess which report section a whole-paper pool
#: sentence belongs under, for citation/labelling purposes only — pool chunks
#: carry a page and chunk id but no section, unlike report-derived material. Not
#: precise, and it doesn't need to be: a slightly-off section label still points
#: the reader at genuine paper text via ``page``/``chunk_id``, and ``prose_score``
#: plus ``is_boilerplate`` (not this) are what keep the *content* honest.
_POOL_SECTION_HINTS: tuple[tuple[SectionName, frozenset[str]], ...] = (
    (
        SectionName.KEY_CONCEPTS,
        frozenset(
            {"define", "defined", "definition", "denote", "denotes", "term",
             "terminology", "notation", "refers", "called", "means"}
        ),
    ),
    (
        SectionName.METHODS,
        frozenset(
            {"method", "approach", "architecture", "algorithm", "propose",
             "proposed", "training", "trained", "dataset", "implementation",
             "procedure", "pipeline", "module", "layer", "step", "steps",
             "network", "model"}
        ),
    ),
    (
        SectionName.KEY_RESULTS,
        frozenset(
            {"result", "results", "accuracy", "performance", "score", "scores",
             "baseline", "outperform", "outperforms", "improve", "improvement",
             "improves", "evaluation", "benchmark", "bleu", "f1"}
        ),
    ),
    (
        SectionName.LIMITATIONS,
        frozenset(
            {"limitation", "limitations", "however", "fail", "fails",
             "assumption", "future", "drawback", "weakness", "threat",
             "unresolved"}
        ),
    ),
    (
        SectionName.WHY_IT_MATTERS,
        frozenset(
            {"conclude", "conclusion", "implication", "implications", "impact",
             "enables", "significance", "matters", "application",
             "applications", "apply"}
        ),
    ),
    (
        SectionName.BACKGROUND,
        frozenset(
            {"prior", "previous", "related", "existing", "traditionally",
             "literature", "earlier"}
        ),
    ),
    (
        SectionName.FIGURES,
        frozenset(
            {"figure", "table", "fig", "chart", "illustrates", "shown", "depicts"}
        ),
    ),
)
#: Fallback when no hint set scores above zero — orienting prose is the most
#: plausible default guess for an unclassified paper sentence.
_DEFAULT_POOL_SECTION = SectionName.OVERVIEW
_WORD_RE = re.compile(r"[a-z]+")


def _classify_pool_section(text: str) -> SectionName:
    """Best-effort guess at which report section a pool sentence belongs under.

    Deterministic keyword overlap, no LLM. Ties and no-hits fall back to
    :data:`_DEFAULT_POOL_SECTION`.
    """
    tokens = set(_WORD_RE.findall((text or "").lower()))
    if not tokens:
        return _DEFAULT_POOL_SECTION
    best_section = _DEFAULT_POOL_SECTION
    best_score = 0
    for section, hints in _POOL_SECTION_HINTS:
        score = len(tokens & hints)
        if score > best_score:
            best_score = score
            best_section = section
    return best_section


# ==========================================================================
# Source material
# ==========================================================================
@dataclass(frozen=True)
class SourceItem:
    """One citable piece of the report the quiz can be built from.

    ``id`` is a short handle (``s1``, ``s2``, …) that the LLM must quote back on
    every question it writes. That is how a generated question earns a citation:
    the citation is looked up from *this* table, never taken from the model's
    output, so a model cannot invent a page number or a section it never saw.
    """

    id: str
    section: SectionName
    page: Optional[int]
    chunk_id: Optional[str]
    snippet: str
    #: Set for a ``**Term** — definition`` pair; ``None`` for a plain claim.
    term: Optional[str]
    #: The definition body, or the claim sentence. Citation markers stripped.
    body: str

    @property
    def is_definition(self) -> bool:
        return bool(self.term)

    def citation(self) -> QuizCitation:
        return QuizCitation(
            section=self.section,
            page=self.page,
            chunk_id=self.chunk_id,
            snippet=self.snippet,
        )


def _strip_markers(text: str) -> str:
    return _WS_RE.sub(" ", _MARKER_RE.sub("", text or "")).strip()


def _first_marker(text: str) -> Optional[int]:
    match = _MARKER_RE.search(text or "")
    return int(match.group(1)) if match else None


#: How much of a line's vocabulary a citation's quoted snippet must account for
#: before that citation is treated as the line's actual source. Overlap is
#: measured against the *line*, so a long snippet is not rewarded for length.
_CITATION_MATCH_FLOOR = 0.4


def _best_matching_citation(
    line: str, candidates: Sequence[Citation]
) -> Optional[Citation]:
    """The citation whose quoted snippet actually overlaps ``line``, else None.

    Replaces "just use the section's first citation", which is how a whole quiz
    ended up attributing every question to one Figure 4 caption on p.14: report
    sections are written by the summarizer and their definition lines usually
    carry no ``[n]`` marker at all, so *every* marker-less line hit the same
    fallback. The citation shown under an answer is supposed to be the evidence
    for it; pointing all of them at one arbitrary passage makes it decoration.

    Returning ``None`` is a perfectly good answer. `emit` then quotes the line
    itself as the snippet and records no page, which is honest — an absent page
    beats a confidently wrong one.
    """
    words = set(_WORD_RE.findall((line or "").lower()))
    if not words:
        return None
    best: Optional[Citation] = None
    best_score = 0.0
    for candidate in candidates:
        snippet_words = set(_WORD_RE.findall((candidate.snippet or "").lower()))
        if not snippet_words:
            continue
        score = len(words & snippet_words) / len(words)
        if score > best_score:
            best, best_score = candidate, score
    return best if best_score >= _CITATION_MATCH_FLOOR else None


def _citation_for(
    line: str, by_marker: dict[int, Citation], candidates: Sequence[Citation]
) -> Optional[Citation]:
    """Resolve a line's citation: its own ``[n]`` marker, else a real overlap."""
    marker = _first_marker(line)
    if marker is not None and marker in by_marker:
        return by_marker[marker]
    return _best_matching_citation(line, candidates)


def collect_sources(
    sections: Sequence[Section], *, extra_chunks: Sequence[AnyChunk] = ()
) -> list[SourceItem]:
    """Mine citable definitions and claim sentences from ``sections`` and
    ``extra_chunks``.

    Purely deterministic — no LLM, so it cannot fail or hallucinate. Each item
    carries the section it came from plus provenance: for a report-derived item,
    whatever the section's ``[n]`` marker resolves to (a real
    :class:`~deepvision.models.report.Citation`, with its page, chunk id and
    exact quoted snippet); for a pool-derived item (``extra_chunks`` — see
    :mod:`deepvision.study.source_pool`), the page and chunk id come straight
    off the chunk it was mined from, and the section is a best-effort guess
    (:func:`_classify_pool_section`) since a raw chunk carries no section of its
    own. Either way every item is citable — :meth:`SourceItem.citation` never
    needs a null section.

    ``extra_chunks`` sentences are filtered with
    :func:`~deepvision.rag.chunk_quality.is_boilerplate` and
    :func:`~deepvision.rag.chunk_quality.prose_score` before becoming question
    material — the direct fix for a generated question that once quoted a
    reference-list entry verbatim ("In Proceed- ings of the IE..."):
    :func:`~deepvision.study.source_pool.pool_for_paper` filters at the *chunk*
    level, but a chunk that is mostly good prose can still carry one
    reference-shaped sentence, and it is sentences that become questions here.

    Sections/chunks with no usable prose contribute nothing rather than an
    empty-shell item.
    """
    by_name: dict[SectionName, Section] = {}
    for section in sections or []:
        # normalize_sections guarantees one section per name; if a duplicate
        # somehow appears, the first (richer) one wins.
        by_name.setdefault(section.name, section)

    items: list[SourceItem] = []
    counter = 0
    seen_bodies: set[str] = set()

    def emit(
        section_name: SectionName,
        *,
        term: Optional[str],
        body: str,
        page: Optional[int],
        chunk_id: Optional[str],
        snippet_source: Optional[str],
    ) -> None:
        nonlocal counter
        body = _strip_markers(body)
        if not body:
            return
        # Figure/table captions are not study material. Checked here rather
        # than at the three call sites so no future source path can bypass it.
        # A shipped quiz asked "Fill in the blank with the value this paper
        # reports: 'Table ____: Variations on the Transformer architecture'",
        # which tests nothing; captions also reach this from *persisted*
        # reports written before the report-side filters existed.
        if is_caption_sentence(body) or is_caption_sentence(term or ""):
            return
        key = punctuation_insensitive(f"{term or ''}|{body}")[:160]
        if key in seen_bodies:
            return
        seen_bodies.add(key)
        counter += 1
        snippet = clean_snippet(snippet_source or body, max_chars=_SNIPPET_CHARS)
        if not snippet:
            return
        items.append(
            SourceItem(
                id=f"s{counter}",
                section=section_name,
                page=page,
                chunk_id=chunk_id,
                snippet=snippet,
                term=term,
                body=clean_snippet(body, max_chars=_MAX_CLAIM_CHARS),
            )
        )

    # --- report: definitions ----------------------------------------------
    definitions = 0
    for name in _DEFINITION_SECTIONS:
        section = by_name.get(name)
        if section is None:
            continue
        by_marker = {c.marker: c for c in section.citations}
        candidates = section.citations or []
        for line in (section.body_markdown or "").splitlines():
            if definitions >= _MAX_DEFINITIONS:
                break
            match = _DEF_RE.match(line)
            if not match:
                continue
            term = _strip_markers(match.group("term"))
            body = _strip_markers(match.group("body"))
            if not term or len(body) < _MIN_DEFINITION_CHARS:
                continue
            citation = _citation_for(line, by_marker, candidates)
            emit(
                section.name,
                term=term,
                body=body,
                page=citation.page if citation else None,
                chunk_id=citation.chunk_id if citation else None,
                snippet_source=citation.snippet if citation and citation.snippet else None,
            )
            definitions += 1

    # --- report: claims ------------------------------------------------------
    claims = 0
    for name in _CLAIM_SECTIONS:
        section = by_name.get(name)
        if section is None:
            continue
        by_marker = {c.marker: c for c in section.citations}
        candidates = section.citations or []
        for line in (section.body_markdown or "").splitlines():
            if claims >= _MAX_CLAIMS:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "|", ">", "```")):
                continue
            citation = _citation_for(line, by_marker, candidates)
            # A bullet is already one claim; a paragraph needs splitting.
            for sentence in _SENTENCE_SPLIT_RE.split(_strip_markers(stripped)):
                if claims >= _MAX_CLAIMS:
                    break
                sentence = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", sentence).strip()
                if len(sentence) < _MIN_CLAIM_CHARS or len(sentence) > _MAX_CLAIM_CHARS:
                    continue
                if _DEF_RE.match(sentence):
                    continue  # already captured as a definition
                emit(
                    section.name,
                    term=None,
                    body=sentence,
                    page=citation.page if citation else None,
                    chunk_id=citation.chunk_id if citation else None,
                    snippet_source=citation.snippet if citation and citation.snippet else None,
                )
                claims += 1

    # --- whole-paper pool ------------------------------------------------
    pool_claims = 0
    for chunk in extra_chunks or ():
        if pool_claims >= _MAX_POOL_CLAIMS:
            break
        text = (chunk.text or "").strip()
        if not text:
            continue
        usable: list[str] = []
        for sentence in _SENTENCE_SPLIT_RE.split(_strip_markers(text)):
            sentence = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", sentence).strip()
            if len(sentence) < _MIN_CLAIM_CHARS or len(sentence) > _MAX_CLAIM_CHARS:
                continue
            # The direct fix for reference-list junk turning into study
            # material: a mangled bibliography entry reads as low-prose
            # boilerplate and never reaches ``emit``.
            if is_boilerplate(sentence) or prose_score(sentence) < _MIN_POOL_PROSE_SCORE:
                continue
            usable.append(sentence)

        # Which `_MAX_CLAIMS_PER_POOL_CHUNK` of them to keep. Taking a prefix is
        # arbitrary, so prefer sentences carrying the paper's own sequence
        # markers ("First, ... Then, ..."): they are concrete procedural claims,
        # and two from one chunk are what `_ordering_questions` needs. Measured
        # on this library, five chunks hold 2+ such sentences (one holds seven)
        # and a blind prefix surfaced almost none of them, which is the whole
        # reason that question shape had never once fired. `sorted` is stable,
        # so document order — the ground truth an ordering question tests — is
        # preserved within each group.
        usable.sort(key=lambda s: not _ORDINAL_MARKER_RE.match(s))

        from_this_chunk = 0
        for sentence in usable:
            if pool_claims >= _MAX_POOL_CLAIMS:
                break
            if from_this_chunk >= _MAX_CLAIMS_PER_POOL_CHUNK:
                break
            match = _DEF_RE.match(sentence)
            if match:
                term = _strip_markers(match.group("term"))
                body = _strip_markers(match.group("body"))
                if not term or len(body) < _MIN_DEFINITION_CHARS:
                    term, body = None, sentence
            else:
                term, body = None, sentence
            emit(
                _classify_pool_section(sentence),
                term=term,
                body=body,
                page=getattr(chunk, "page", None),
                chunk_id=chunk.id,
                snippet_source=None,
            )
            pool_claims += 1
            from_this_chunk += 1

    return items[:_MAX_SOURCES]


# ==========================================================================
# Small text helpers
# ==========================================================================
def _term_variants(term: str) -> list[str]:
    """``"BPE (Byte Pair Encoding)"`` → the label plus each of its two halves."""
    term = (term or "").strip()
    if not term:
        return []
    variants = [term]
    match = _PARENTHETICAL_RE.match(term)
    if match:
        variants.append(match.group("head").strip())
        variants.append(match.group("tail").strip())
    return [v for v in dict.fromkeys(variants) if len(v) >= 2]


def _mask_term(text: str, term: str) -> str:
    """Blank out ``term`` (and its variants) inside ``text``.

    A definition very often restates its own term, which would hand the answer
    to the reader for free — the option that mentions the term *is* the correct
    one. Masking is applied to **every** option in a question, not just the
    correct one, so the blank itself carries no signal.
    """
    masked = text or ""
    for variant in sorted(_term_variants(term), key=len, reverse=True):
        pattern = re.compile(
            rf"(?<!\w){re.escape(variant)}(?:'s|s)?(?!\w)", re.IGNORECASE
        )
        masked = pattern.sub("___", masked)
    return _WS_RE.sub(" ", masked).strip()


#: Two definitions this similar are treated as describing the same thing.
_NEAR_DUPLICATE_OVERLAP = 0.75


def _too_similar(left: str, right: str) -> bool:
    """True when two definitions say nearly the same thing.

    Papers routinely define near-synonyms ("self-attention" and
    "intra-attention"). Using one as a distractor for the other would give a
    multiple-choice question two defensible answers, and would make a
    "**A** means <definition of B>" true/false statement — built to be false —
    actually true. Cheaper to skip the pairing than to ship a question the user
    is right to argue with.
    """
    return max(token_overlap(left, right), token_overlap(right, left)) >= (
        _NEAR_DUPLICATE_OVERLAP
    )


def _lower_first(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    # Don't de-capitalize an acronym or a proper noun run ("BLEU scores …").
    if len(text) > 1 and text[1].isupper():
        return text
    return text[0].lower() + text[1:]


def _trim_period(text: str) -> str:
    return (text or "").strip().rstrip(" .")


#: A definition that opens by restating its subject — "This is a technique
#: that ...", "It refers to a method ...", "___ is a type of ...". The tail is
#: what the term actually *is*; the opener is a pronoun standing in for it.
#: An optional scene-setting clause before the subject — "In the context of
#: attention mechanisms, ___ refers to ...". Anchoring the copular pattern at
#: ``^`` alone missed every definition written this way, and the result shipped:
#: "**QKT** refers to in the context of attention mechanisms, ___ refers to a
#: mathematical operation ..." — a doubled "refers to", a leaked ``___`` mask
#: and a prepositional phrase where the predicate belongs, all at once.
#: Restricted to a short clause opening with a preposition or subordinator and
#: closing on a comma, so it cannot eat real subject matter.
_PREAMBLE = (
    r"(?:(?:in|on|at|for|with|within|under|by|from|as|when|where|during|"
    r"through|via|after|before|given|based\s+on)\b[^,]{0,70},\s*)?"
)
_COPULAR_OPENER_RE = re.compile(
    r"^\s*" + _PREAMBLE + r"(?:this|it|these|those|that|___)\s+"
    r"(?:is|are|was|were|refers?\s+to|denotes?|means?)\s+"
    r"(?:defined\s+as\s+)?",
    re.IGNORECASE,
)


def _as_predicate(text: str) -> str:
    """Reduce a definition to the phrase that can follow "X refers to ...".

    Without this the true/false builder stitched its template straight onto a
    definition that already had a subject, and shipped prompts reading "in this
    paper, **Multi-Head Attention** refers to this is a technique used in deep
    learning models ..." — visible in the UI, and the kind of thing that makes
    a study tool look untrustworthy even when the answer key is right.

    Dropping the opener also removes the most common place a ``___`` mask
    surfaced, since :func:`_mask_term` blanks the term wherever it appears and
    a definition usually leads with it.

    Falls back to the original text whenever trimming would leave too little
    to be a sentence — a wrong-looking prompt beats an empty one.
    """
    original = (text or "").strip()
    trimmed = _COPULAR_OPENER_RE.sub("", original, count=1).strip()
    if len(trimmed) < 12:
        return original
    return trimmed


# ==========================================================================
# Question wording
# ==========================================================================
# Measured on five real quizzes (49 questions): only 9 of 12 stems ever fired,
# the top three covered 61% of everything, and "True or false: in this paper,
# **X** refers to ..." alone was 30% of *every* quiz — because
# `_definition_true_false` had exactly one stem and definitions are most of
# what a report yields. Sitting a 10-question quiz meant reading the same
# opening line three times.
#
# Stems are chosen by hashing the question's own content, never with `rng`.
# That is not cosmetic: `avoid_prompts` de-duplicates across quizzes by
# comparing normalized prompt *text*, so a randomly-worded question would slip
# past it and the same material would come back wearing a different sentence.
# Deterministic wording keeps that guard working.


def _stem_rotate(index: int, options: Sequence[str]) -> str:
    """Pick a stem by position, so a quiz exhausts the pool before repeating one.

    Rotation rather than hashing, and the difference is visible in output.
    Hashing each item independently draws stems uniformly *and independently*,
    so with four stems and three same-kind questions a collision is more likely
    than not — the first cut of this shipped "Pick the description this paper
    gives for **X**." twice inside one ten-question quiz, which is the exact
    complaint this work exists to fix. Rotating by index cannot do that until
    the pool is used up.

    Still fully deterministic, which `avoid_prompts` relies on:
    :func:`collect_sources` is deterministic for a given report and pool, so an
    item keeps its index and therefore its wording across regenerations.
    """
    if not options:
        return ""
    return options[index % len(options)]


#: True/false over a term and its definition. ``{term}`` and ``{pred}``.
_TF_DEFINITION_STEMS: tuple[str, ...] = (
    "True or false: in this paper, **{term}** refers to {pred}.",
    "True or false — this paper uses **{term}** to mean {pred}.",
    "The paper defines **{term}** as {pred}. True or false?",
    "Is this right? In this paper, **{term}** is {pred}.",
    "True or false: for this paper, **{term}** means {pred}.",
)
#: Multiple choice, "which description is this term". ``{term}``.
_MC_DEFINITION_STEMS: tuple[str, ...] = (
    "As this paper uses it, which description best fits **{term}**?",
    "This paper uses **{term}** in a particular way. Which description is it?",
    "Pick the description this paper gives for **{term}**.",
    "Which of these is **{term}**, as this paper defines it?",
)
#: Short answer, "name the thing this describes". ``{masked}``.
#: The term is masked out of ``{masked}``, so the answer is never in the stem.
_SA_DEFINITION_STEMS: tuple[str, ...] = (
    "What does this paper call the following? “{masked}”",
    "The paper has a name for this. What is it? “{masked}”",
    "Name the idea this describes: “{masked}”",
    "Which term does this paper use for this? “{masked}”",
)
#: Fill in a reported number. ``{blanked}``.
_NUMERIC_BLANK_STEMS: tuple[str, ...] = (
    "Fill in the blank with the value this paper reports: “{blanked}”",
    "What value belongs in the blank? “{blanked}”",
    "The paper reports a specific figure here — what is it? “{blanked}”",
    "Recall the number: “{blanked}”",
)
#: Multiple choice, "the paper describes something — name it". ``{own}``.
_MC_INTRO_STEMS: tuple[str, ...] = (
    "The paper introduces an idea this way: “{own}”. Which of its terms is "
    "being described?",
    "This description comes from the paper: “{own}”. What is it describing?",
    "The paper says: “{own}”. Which of its ideas is that?",
)
#: Multiple choice, transfer framing. ``{own}``.
_MC_APPLY_STEMS: tuple[str, ...] = (
    "You are applying this work to a new problem and need the piece that "
    "provides the following: “{own}”. Which of the paper's ideas should you "
    "reach for?",
    "Your new system needs this: “{own}”. Which idea from the paper supplies it?",
    "To get this behaviour — “{own}” — which of the paper's components do you "
    "need?",
)
#: Short answer, transfer framing. ``{masked}``.
_SA_APPLY_STEMS: tuple[str, ...] = (
    "In a new project you need this capability: “{masked}”. Which idea from "
    "this paper would you use? Name it.",
    "You need this in a new build: “{masked}”. Name the idea from this paper "
    "that gives it to you.",
    "Which of this paper's ideas would you reach for to get this? “{masked}”",
)
#: "Which section would this be in". ``{excerpt}``.
_MC_LOCATOR_STEMS: tuple[str, ...] = (
    "If you wanted to find this in the paper, which section would you look "
    "in? “{excerpt}”",
    "Which section of the paper does this come from? “{excerpt}”",
    "You remember reading this. Where in the paper was it? “{excerpt}”",
    "Place this passage: which section is it from? “{excerpt}”",
)
#: True/false over a claim the paper states verbatim. ``{sentence}``.
_TF_CLAIM_STEMS: tuple[str, ...] = (
    "True or false: {sentence}.",
    "Does this paper say this? {sentence}.",
    "Is this what the paper reports? {sentence}.",
    "True or false, according to this paper: {sentence}.",
)


def _mutate_number(value: str, suffix: str) -> str:
    """Return a number that is clearly different from ``value``.

    Used to build a **reliably false** true/false statement out of a true one:
    the rest of the sentence is the paper's own wording, and only the quantity is
    wrong. A year is shifted by seven (doubling 2017 is absurd rather than
    plausible); anything else is doubled, with a +1 guard so 0 does not map to
    itself.
    """
    try:
        if "." in value:
            number = float(value)
            mutated = round(number * 2, 2) if number else 1.5
            text = f"{mutated:g}"
        else:
            number = int(value)
            if 1900 <= number <= 2100:
                mutated = number + 7
            else:
                mutated = number * 2 if number else 1
            if mutated == number:
                mutated = number + 1
            text = str(mutated)
    except (TypeError, ValueError):  # pragma: no cover - regex guarantees numeric
        return value + suffix
    return text + suffix


# ==========================================================================
# Deterministic draft questions
# ==========================================================================
#: A draft is a plain dict so it round-trips through JSON to the model and back
#: unchanged. Keys mirror the JSON contract in the system prompt.
Draft = dict[str, Any]


@dataclass
class _DraftPool:
    """Candidate drafts bucketed by ``(kind, difficulty)``."""

    buckets: dict[tuple[QuestionKind, Difficulty], list[Draft]] = field(
        default_factory=dict
    )

    def add(self, kind: QuestionKind, difficulty: Difficulty, draft: Draft) -> None:
        self.buckets.setdefault((kind, difficulty), []).append(draft)

    def take(self, kind: QuestionKind, difficulty: Difficulty) -> Optional[Draft]:
        bucket = self.buckets.get((kind, difficulty))
        return bucket.pop(0) if bucket else None

    def leftovers(self) -> list[Draft]:
        out: list[Draft] = []
        for bucket in self.buckets.values():
            out.extend(bucket)
        return out


def _draft(
    *,
    family: str,
    kind: QuestionKind,
    difficulty: Difficulty,
    prompt: str,
    source: SourceItem,
    explanation: str,
    options: Optional[list[dict[str, str]]] = None,
    correct_option_id: Optional[str] = None,
    correct_answer_text: Optional[str] = None,
    accepted_answers: Optional[list[str]] = None,
    anchor: Optional[str] = None,
) -> Draft:
    """Build one draft dict.

    ``anchor`` is internal bookkeeping — the term or fact the question is
    *about*, used by :func:`_select_drafts` to stop one glossary entry from
    supplying half the quiz. ``family`` is the question *shape* it came from
    ("definition", "numeric", "ordering", "locator"), used by the same function
    to stop one shape from doing so. Both are stripped before the drafts are
    shown to the model and ignored by :func:`validate_draft`.
    """
    return {
        "kind": kind.value,
        "difficulty": difficulty.value,
        "prompt": prompt.strip(),
        "options": options or [],
        "correct_option_id": correct_option_id,
        "correct_answer_text": correct_answer_text,
        "accepted_answers": accepted_answers or [],
        "explanation": explanation.strip(),
        "source_id": source.id,
        "_anchor": anchor or correct_answer_text or source.term or source.id,
        "_family": family,
    }


def _shuffled_options(
    texts: Sequence[str], correct_index: int, rng: random.Random
) -> tuple[list[dict[str, str]], str]:
    """Shuffle four option texts and return ``(options, correct_option_id)``.

    Shuffling matters: without it the correct answer is always ``a`` and the
    quiz is solvable without reading it.
    """
    order = list(range(len(texts)))
    rng.shuffle(order)
    options = [
        {"id": MCQ_OPTION_IDS[slot], "text": texts[src]}
        for slot, src in enumerate(order)
    ]
    correct_option_id = MCQ_OPTION_IDS[order.index(correct_index)]
    return options, correct_option_id


def _definition_mcqs(
    definitions: Sequence[SourceItem], rng: random.Random, pool: _DraftPool
) -> None:
    """Term↔definition multiple choice, in three framings (one per tier)."""
    if len(definitions) < 4:
        return
    for position, item in enumerate(definitions):
        others = [
            d
            for d in definitions
            if d.id != item.id and not _too_similar(d.body, item.body)
        ]
        if len(others) < 3:
            continue
        distractors = rng.sample(others, 3)
        term = item.term or ""
        own = _trim_period(_mask_term(item.body, term))
        other_texts = [_trim_period(_mask_term(d.body, d.term or "")) for d in distractors]
        other_terms = [d.term or "" for d in distractors]

        # recall — "which description belongs to this term"
        options, correct = _shuffled_options([own, *other_texts], 0, rng)
        pool.add(
            QuestionKind.MULTIPLE_CHOICE,
            Difficulty.RECALL,
            _draft(
                family="definition",
                kind=QuestionKind.MULTIPLE_CHOICE,
                difficulty=Difficulty.RECALL,
                prompt=_stem_rotate(position, _MC_DEFINITION_STEMS).format(term=term),
                source=item,
                options=options,
                correct_option_id=correct,
                explanation=(
                    f"The paper's {item.section.value} describes **{term}** as: "
                    f"{_trim_period(item.body)}. The other options are real "
                    f"descriptions from the same paper — of "
                    f"{', '.join(t for t in other_terms if t)} — which is what makes "
                    f"them tempting: they are on topic, but they answer a different "
                    f"question."
                ),
            ),
        )

        # understand — "which term does this description name"
        term_texts = [term, *other_terms]
        options, correct = _shuffled_options(term_texts, 0, rng)
        pool.add(
            QuestionKind.MULTIPLE_CHOICE,
            Difficulty.UNDERSTAND,
            _draft(
                family="definition",
                kind=QuestionKind.MULTIPLE_CHOICE,
                difficulty=Difficulty.UNDERSTAND,
                prompt=(
                    _stem_rotate(position, _MC_INTRO_STEMS).format(
                        own=_trim_period(own)
                    )
                ),
                source=item,
                options=options,
                correct_option_id=correct,
                explanation=(
                    f"That is this paper's definition of **{term}**: "
                    f"{_trim_period(item.body)}. "
                    f"**{other_terms[0]}** is the easiest one to pick by mistake — it "
                    f"appears in the same discussion, but the paper uses it for "
                    f"something else."
                ),
            ),
        )

        # apply — same content, transferred to a new situation
        if len(item.body) >= 60:
            options, correct = _shuffled_options(term_texts, 0, rng)
            pool.add(
                QuestionKind.MULTIPLE_CHOICE,
                Difficulty.APPLY,
                _draft(
                    family="definition",
                    kind=QuestionKind.MULTIPLE_CHOICE,
                    difficulty=Difficulty.APPLY,
                    prompt=(
                        _stem_rotate(position, _MC_APPLY_STEMS).format(
                            own=_trim_period(own)
                        )
                    ),
                    source=item,
                    options=options,
                    correct_option_id=correct,
                    explanation=(
                        f"**{term}** is the part that does this: "
                        f"{_trim_period(item.body)}. The other options are genuine "
                        f"ideas from the paper, but they solve a different part of "
                        f"the problem, so picking one would leave this requirement "
                        f"unmet."
                    ),
                ),
            )


def _definition_true_false(
    definitions: Sequence[SourceItem], rng: random.Random, pool: _DraftPool
) -> None:
    """Term/definition pairings, half of them deliberately mismatched.

    The false variant swaps in **another term's** definition from the same
    paper: reliably false (the two definitions differ) and genuinely tempting
    (both are real content the reader has seen).
    """
    if len(definitions) < 2:
        return
    for position, item in enumerate(definitions):
        term = item.term or ""
        if position % 2 == 0:
            statement = _mask_term(item.body, term)
            pool.add(
                QuestionKind.TRUE_FALSE,
                Difficulty.RECALL,
                _draft(
                    family="definition",
                    kind=QuestionKind.TRUE_FALSE,
                    difficulty=Difficulty.RECALL,
                    prompt=_stem_rotate(position, _TF_DEFINITION_STEMS).format(
                        term=term,
                        pred=_lower_first(_as_predicate(_trim_period(statement))),
                    ),
                    source=item,
                    options=[{"id": oid, "text": text} for oid, text in TRUE_FALSE_OPTIONS],
                    correct_option_id="true",
                    explanation=(
                        f"True. The {item.section.value} section defines **{term}** "
                        f"exactly this way: {_trim_period(item.body)}."
                    ),
                ),
            )
            continue

        others = [
            d
            for d in definitions
            if d.id != item.id and d.term and not _too_similar(d.body, item.body)
        ]
        if not others:
            continue
        other = rng.choice(others)
        statement = _mask_term(other.body, other.term or "")
        pool.add(
            QuestionKind.TRUE_FALSE,
            Difficulty.UNDERSTAND,
            _draft(
                family="definition",
                kind=QuestionKind.TRUE_FALSE,
                difficulty=Difficulty.UNDERSTAND,
                prompt=_stem_rotate(position, _TF_DEFINITION_STEMS).format(
                    term=term,
                    pred=_lower_first(_as_predicate(_trim_period(statement))),
                ),
                source=item,
                options=[{"id": oid, "text": text} for oid, text in TRUE_FALSE_OPTIONS],
                correct_option_id="false",
                explanation=(
                    f"False — that is the paper's description of **{other.term}**, "
                    f"not of **{term}**. The paper describes **{term}** as: "
                    f"{_trim_period(item.body)}."
                ),
            ),
        )


def _definition_short_answers(
    definitions: Sequence[SourceItem], pool: _DraftPool
) -> None:
    """Recall and apply framings of "name the term for this description"."""
    for position, item in enumerate(definitions):
        term = (item.term or "").strip()
        if not term:
            continue
        masked = _trim_period(_mask_term(item.body, term))
        if not masked or "___" == masked:
            continue
        accepted = _term_variants(term)
        pool.add(
            QuestionKind.SHORT_ANSWER,
            Difficulty.RECALL,
            _draft(
                family="definition",
                kind=QuestionKind.SHORT_ANSWER,
                difficulty=Difficulty.RECALL,
                prompt=_stem_rotate(position, _SA_DEFINITION_STEMS).format(masked=masked),
                source=item,
                correct_answer_text=term,
                accepted_answers=accepted,
                explanation=(
                    f"The paper's term is **{term}**: {_trim_period(item.body)}."
                ),
            ),
        )
        if len(item.body) >= 60:
            pool.add(
                QuestionKind.SHORT_ANSWER,
                Difficulty.APPLY,
                _draft(
                    family="definition",
                    kind=QuestionKind.SHORT_ANSWER,
                    difficulty=Difficulty.APPLY,
                    prompt=(
                        _stem_rotate(position, _SA_APPLY_STEMS).format(masked=masked)
                    ),
                    source=item,
                    correct_answer_text=term,
                    accepted_answers=accepted,
                    explanation=(
                        f"**{term}** is the idea to reach for — the paper describes "
                        f"it as: {_trim_period(item.body)}."
                    ),
                ),
            )


def _numeric_claims(claims: Sequence[SourceItem], pool: _DraftPool) -> None:
    """True/false and fill-in-the-blank questions built around a stated number.

    Only sentences with **exactly one** number are used: with two, mutating one
    of them can accidentally produce a statement that is true of the other, and
    the blank becomes ambiguous.
    """
    for position, item in enumerate(claims):
        numbers = _NUMBER_RE.findall(item.body)
        if len(numbers) != 1:
            continue
        # A lone number that is really a pointer — "as shown in Figure 1" — is
        # not a value the paper "reports", and blanking it produces a question
        # about a figure's numbering rather than about the work. Same rule the
        # flashcard cloze uses; see `chunk_quality.is_ordinal_number`.
        only = _NUMBER_RE.search(item.body)
        if only is not None and is_ordinal_number(item.body, only.start()):
            continue
        value, suffix = numbers[0]
        original = f"{value}{suffix}"
        sentence = _trim_period(item.body)

        if position % 2 == 0:
            pool.add(
                QuestionKind.TRUE_FALSE,
                Difficulty.UNDERSTAND,
                _draft(
                    family="numeric",
                    kind=QuestionKind.TRUE_FALSE,
                    difficulty=Difficulty.UNDERSTAND,
                    prompt=_stem_rotate(position, _TF_CLAIM_STEMS).format(
                        sentence=_lower_first(sentence)
                    ),
                    source=item,
                    options=[{"id": oid, "text": text} for oid, text in TRUE_FALSE_OPTIONS],
                    correct_option_id="true",
                    explanation=(
                        f"True — the paper's {item.section.value} states it: "
                        f"“{sentence}”, including the figure of "
                        f"{original}."
                    ),
                ),
            )
        else:
            mutated = _mutate_number(value, suffix)
            false_sentence = _NUMBER_RE.sub(
                lambda _m: mutated, sentence, count=1
            )
            pool.add(
                QuestionKind.TRUE_FALSE,
                Difficulty.UNDERSTAND,
                _draft(
                    family="numeric",
                    kind=QuestionKind.TRUE_FALSE,
                    difficulty=Difficulty.UNDERSTAND,
                    prompt=_stem_rotate(position, _TF_CLAIM_STEMS).format(
                        sentence=_lower_first(_trim_period(false_sentence))
                    ),
                    source=item,
                    options=[{"id": oid, "text": text} for oid, text in TRUE_FALSE_OPTIONS],
                    correct_option_id="false",
                    explanation=(
                        f"False. The paper gives {original}, not {mutated}: "
                        f"“{sentence}”."
                    ),
                ),
            )

        blanked = _NUMBER_RE.sub(lambda _m: "____", sentence, count=1)
        pool.add(
            QuestionKind.SHORT_ANSWER,
            Difficulty.UNDERSTAND,
            _draft(
                family="numeric",
                kind=QuestionKind.SHORT_ANSWER,
                difficulty=Difficulty.UNDERSTAND,
                prompt=_stem_rotate(position, _NUMERIC_BLANK_STEMS).format(
                    blanked=blanked
                ),
                source=item,
                correct_answer_text=original,
                accepted_answers=list(dict.fromkeys([original, value])),
                explanation=(
                    f"The paper states: “{sentence}” — so the missing "
                    f"value is **{original}**."
                ),
            ),
        )


#: A sentence opening with an explicit sequence word — "First, we tokenize...",
#: "Then the encoder...", "Finally, a classifier...". These are the paper's own
#: markers of order, never inferred, so an ordering question built from them has
#: ground truth exactly as strong as a definition or a stated number.
_ORDINAL_MARKER_RE = re.compile(
    r"^(first|second|third|fourth|fifth|initially|then|next|finally|"
    r"subsequently|afterwards?|after\s+that|lastly)\b[,:]?\s+",
    re.IGNORECASE,
)


def _strip_ordinal_marker(sentence: str) -> str:
    return _ORDINAL_MARKER_RE.sub("", sentence, count=1).strip()


def _ordering_questions(
    sources: Sequence[SourceItem], rng: random.Random, pool: _DraftPool
) -> None:
    """"Which step comes first" MCQ / true-false, built from the paper's own
    sequence markers.

    A shape the report-only source pool made impractical: three or more
    explicitly-ordered steps rarely all survived into the same section's
    summarised prose. Grouped by ``(section, chunk_id)`` so the order compared
    is always order *within one contiguous span of the paper's own text*, never
    an order inferred across unrelated chunks.
    """
    groups: dict[tuple[SectionName, Optional[str]], list[SourceItem]] = {}
    for item in sources:
        if item.is_definition:
            continue
        if not _ORDINAL_MARKER_RE.match(item.body):
            continue
        groups.setdefault((item.section, item.chunk_id), []).append(item)

    for (section, _chunk_id), steps in groups.items():
        if len(steps) < 2:
            continue
        steps = steps[:5]
        texts = [_trim_period(_strip_ordinal_marker(s.body)) for s in steps]
        if any(len(t) < 15 for t in texts):
            continue
        anchor_source = steps[0]
        anchor = f"order:{section.value}:{anchor_source.chunk_id or anchor_source.id}"

        # recall — which of these steps happens first. Multiple choice is a
        # closed contract of exactly 4 options (see validate_draft), so this
        # framing only fires with 4+ genuinely-ordered steps; fewer than that
        # still gets the true/false framing below.
        if len(texts) >= 4:
            mcq_texts = texts[:4]
            options, correct = _shuffled_options(mcq_texts, 0, rng)
            pool.add(
                QuestionKind.MULTIPLE_CHOICE,
                Difficulty.RECALL,
                _draft(
                    family="ordering",
                    kind=QuestionKind.MULTIPLE_CHOICE,
                    difficulty=Difficulty.RECALL,
                    prompt=(
                        f"According to this paper's {section.value}, which of "
                        "these steps happens first?"
                    ),
                    source=anchor_source,
                    options=options,
                    correct_option_id=correct,
                    explanation=(
                        "The paper lays these out in this order: "
                        + "; then ".join(mcq_texts)
                        + f". So the first step is: {mcq_texts[0]}."
                    ),
                    anchor=anchor,
                ),
            )

        # understand — true/false about the relative order of two adjacent steps
        i = rng.randrange(len(texts) - 1)
        true_first, true_second = texts[i], texts[i + 1]
        flip = rng.random() < 0.5
        shown_first, shown_second = (
            (true_second, true_first) if flip else (true_first, true_second)
        )
        pool.add(
            QuestionKind.TRUE_FALSE,
            Difficulty.UNDERSTAND,
            _draft(
                family="ordering",
                kind=QuestionKind.TRUE_FALSE,
                difficulty=Difficulty.UNDERSTAND,
                prompt=(
                    f"True or false: in this paper's {section.value}, "
                    f"“{_lower_first(shown_first)}” happens before "
                    f"“{_lower_first(shown_second)}”."
                ),
                source=anchor_source,
                options=[{"id": oid, "text": text} for oid, text in TRUE_FALSE_OPTIONS],
                correct_option_id="false" if flip else "true",
                explanation=(
                    f"False — the paper's actual order is the reverse."
                    if flip
                    else "True — that is the order the paper describes."
                )
                + f" It states: {true_first}; then {true_second}.",
                anchor=anchor,
            ),
        )


def _locator_questions(
    sources: Sequence[SourceItem], rng: random.Random, pool: _DraftPool
) -> None:
    """"Which section would you look in for X" — a locator/study-navigation MCQ.

    Only meaningfully answerable with material spread across several distinct
    sections, which is exactly what the whole-paper pool adds: report-only
    material tended to cluster in two or three sections (Key Concepts, Key
    Results), making every distractor section implausible by construction.
    """
    sections_present = sorted({item.section for item in sources}, key=lambda s: s.value)
    if len(sections_present) < 4:
        return

    for position, item in enumerate(sources):
        excerpt = _trim_period(item.body)
        if len(excerpt) < 20:
            continue
        others = [s for s in sections_present if s != item.section]
        if len(others) < 3:
            continue
        distractor_sections = rng.sample(others, 3)
        section_texts = [item.section.value, *[s.value for s in distractor_sections]]
        options, correct = _shuffled_options(section_texts, 0, rng)
        pool.add(
            QuestionKind.MULTIPLE_CHOICE,
            Difficulty.UNDERSTAND,
            _draft(
                family="locator",
                kind=QuestionKind.MULTIPLE_CHOICE,
                difficulty=Difficulty.UNDERSTAND,
                prompt=_stem_rotate(position, _MC_LOCATOR_STEMS).format(
                    excerpt=excerpt
                ),
                source=item,
                options=options,
                correct_option_id=correct,
                explanation=(
                    f"That is covered under **{item.section.value}**. "
                    f"{', '.join(s.value for s in distractor_sections)} cover "
                    "other parts of the paper, not this."
                ),
                anchor=f"locator:{item.id}",
            ),
        )


def _build_draft_pool(sources: Sequence[SourceItem], rng: random.Random) -> _DraftPool:
    pool = _DraftPool()
    definitions = [s for s in sources if s.is_definition]
    claims = [s for s in sources if not s.is_definition]
    _definition_mcqs(definitions, rng, pool)
    _definition_true_false(definitions, rng, pool)
    _definition_short_answers(definitions, pool)
    _numeric_claims(claims, pool)
    _ordering_questions(sources, rng, pool)
    _locator_questions(sources, rng, pool)
    return pool


def _rotation(
    kinds: Sequence[QuestionKind], difficulties: Sequence[Difficulty]
) -> list[tuple[QuestionKind, Difficulty]]:
    """Slot order that alternates kind *and* tier.

    Walking difficulty-major (all recall, then all understand, …) and kind-minor
    means a truncated quiz still contains every kind and starts easy — which is
    the shape a study quiz should have.
    """
    wanted_kinds = [k for k in QuestionKind if k in set(kinds)] or list(QuestionKind)
    wanted_tiers = [d for d in Difficulty if d in set(difficulties)] or list(Difficulty)
    return [(kind, tier) for tier in wanted_tiers for kind in wanted_kinds]


def _select_drafts(
    pool: _DraftPool,
    *,
    count: int,
    kinds: Sequence[QuestionKind],
    difficulties: Sequence[Difficulty],
    avoid_prompts: frozenset[str] = frozenset(),
) -> list[Draft]:
    """Round-robin over the requested ``(kind, tier)`` slots until ``count``.

    ``avoid_prompts`` is a set of :func:`~deepvision.study.grading.
    punctuation_insensitive`-normalized prompts already asked in a previous
    quiz for this paper. A draft whose prompt matches one is skipped *while
    unseen material remains* — it is set aside in ``avoided`` rather than
    discarded, and only spent if the pool truly runs dry, so a quiz never comes
    back short because everything candidate was something asked before.
    """
    rotation = _rotation(kinds, difficulties)
    if not rotation:
        return []

    picked: list[Draft] = []
    seen_prompts: set[str] = set()
    #: A term may anchor at most two questions, so a 10-question quiz is not
    #: four questions about one glossary entry.
    term_uses: dict[str, int] = {}
    #: Source items already spent. One question per item: the anchor cap alone
    #: permitted the same *definition* to be asked twice in two formats, which
    #: is what a real 9-question quiz did three times over (short-answer then
    #: multiple-choice on identical text), carrying about five distinct subjects.
    #: A term can still anchor two questions — but only from two different
    #: sources, i.e. only when there is genuinely different material behind them.
    used_sources: set[str] = set()
    #: Drafts set aside only because they match ``avoid_prompts`` — the reuse
    #: pool if nothing else is left.
    avoided: list[Draft] = []
    #: Drafts set aside only because their source was already used. Spent before
    #: the quiz is allowed to come up short, so this tightening can shrink
    #: *redundancy* but never the question count.
    deferred: list[Draft] = []
    #: How many questions each *shape* has contributed. Widening the stems fixed
    #: how a question is worded but not how many of one kind you get: a real
    #: 10-question quiz came back with **five** "which section is this from?"
    #: questions wearing four different wordings, because a paper with few crisp
    #: definitions leaves the locator generator to fill every MC/understand slot
    #: the rotation offers. Varied phrasing on a monotonous quiz is still a
    #: monotonous quiz.
    family_uses: dict[str, int] = {}

    #: No shape may exceed its fair share of the quiz while other shapes still
    #: have material. The share is computed from the families actually present,
    #: not a constant: a paper yielding only two shapes cannot fill ten
    #: questions at "one third each", and a fixed cap there would just push
    #: everything into the spillover pass and be no cap at all.
    available_families = {
        str(draft.get("_family") or "")
        for bucket in pool.buckets.values()
        for draft in bucket
    } - {""}
    family_cap = max(2, -(-count // max(len(available_families), 1)))

    def accept(
        draft: Optional[Draft],
        *,
        allow_avoided: bool = False,
        allow_reused_source: bool = False,
        allow_over_family: bool = False,
    ) -> bool:
        if draft is None:
            return False
        key = punctuation_insensitive(draft.get("prompt", ""))[:200]
        if not key or key in seen_prompts:
            return False
        if not allow_avoided and avoid_prompts and key in avoid_prompts:
            avoided.append(draft)
            return False
        source_id = str(draft.get("source_id") or "")
        if not allow_reused_source and source_id and source_id in used_sources:
            deferred.append(draft)
            return False
        family = str(draft.get("_family") or "")
        if (
            not allow_over_family
            and family
            and family_uses.get(family, 0) >= family_cap
        ):
            deferred.append(draft)
            return False
        anchor = punctuation_insensitive(str(draft.get("_anchor") or ""))
        if anchor and term_uses.get(anchor, 0) >= 2:
            return False
        seen_prompts.add(key)
        if source_id:
            used_sources.add(source_id)
        if family:
            family_uses[family] = family_uses.get(family, 0) + 1
        if anchor:
            term_uses[anchor] = term_uses.get(anchor, 0) + 1
        picked.append(draft)
        return True

    progressed = True
    while len(picked) < count and progressed:
        progressed = False
        for slot in rotation:
            if len(picked) >= count:
                break
            for _attempt in range(4):  # skip past duplicates in this bucket
                draft = pool.take(*slot)
                if draft is None:
                    break
                if accept(draft):
                    progressed = True
                    break

    if len(picked) < count:
        for draft in pool.leftovers():
            if len(picked) >= count:
                break
            accept(draft)

    # Still short: spend the second question on an already-used source before
    # spending a repeat of a previous quiz. Asking about the same passage twice
    # is a milder failure than asking the identical question two quizzes running.
    # Spill over in two stages, relaxing the cheaper constraint first. A draft
    # held back only for its family must not also be granted source reuse:
    # letting both go at once silently undid the one-question-per-source rule
    # whenever a family cap bound first.
    if len(picked) < count and deferred:
        for draft in list(deferred):
            if len(picked) >= count:
                break
            accept(draft, allow_over_family=True)
    if len(picked) < count and deferred:
        for draft in list(deferred):
            if len(picked) >= count:
                break
            accept(draft, allow_reused_source=True, allow_over_family=True)

    # The pool is exhausted (nothing unseen left) but the quiz is still short:
    # reuse material that was only set aside for repeating a previous quiz.
    if len(picked) < count and avoided:
        for draft in avoided:
            if len(picked) >= count:
                break
            accept(
                draft,
                allow_avoided=True,
                allow_reused_source=True,
                allow_over_family=True,
            )

    return picked


# ==========================================================================
# Validation — the gate everything (draft or model-written) must pass
# ==========================================================================
def validate_draft(
    raw: Any,
    *,
    sources: dict[str, SourceItem],
    quiz_id: str,
    paper_id: str,
    seen_prompts: set[str],
) -> Optional[QuizQuestion]:
    """Turn one raw draft dict into a :class:`QuizQuestion`, or return ``None``.

    Every rule here exists because breaking it ships a broken quiz:

    - the prompt and the explanation must be real (a question with no
      explanation teaches nothing, which is the entire point of the feature);
    - ``source_id`` must name a source item we actually extracted — that is what
      makes the citation real rather than model-supplied, and a question that
      cannot be cited is dropped rather than emitted with a null citation;
    - multiple choice needs exactly 4 distinct options, exactly one of them
      flagged correct, and no "all/none of the above";
    - true/false needs exactly the fixed ``true``/``false`` pair;
    - short answer needs a canonical answer, and ``accepted_answers`` always
      contains it;
    - the prompt must not contain its own answer (a model asked to rewrite
      questions loves to leak it), and no two questions may share a prompt.

    Never raises: malformed input is a dropped question, not a 500.
    """
    if not isinstance(raw, dict):
        return None
    try:
        source = sources.get(str(raw.get("source_id") or "").strip())
        if source is None:
            log.debug("quiz question dropped: unknown source_id", extra={"raw": str(raw)[:200]})
            return None

        prompt = _WS_RE.sub(" ", str(raw.get("prompt") or "")).strip()
        explanation = _WS_RE.sub(" ", str(raw.get("explanation") or "")).strip()
        if len(prompt) < 12 or len(explanation) < 20:
            return None

        # ``prompt_norm`` is the full text (used for answer-leak checks, which
        # must not be blinded by a truncation); ``prompt_key`` is the bounded
        # dedupe key.
        prompt_norm = punctuation_insensitive(prompt)
        prompt_key = prompt_norm[:200]
        if not prompt_key or prompt_key in seen_prompts:
            return None

        try:
            kind = QuestionKind(str(raw.get("kind") or "").strip())
            difficulty = Difficulty(str(raw.get("difficulty") or "").strip())
        except ValueError:
            return None

        options: list[QuizOption] = []
        correct_option_id: Optional[str] = None
        correct_answer_text: Optional[str] = None
        accepted_answers: list[str] = []

        if kind is QuestionKind.SHORT_ANSWER:
            correct_answer_text = str(raw.get("correct_answer_text") or "").strip()
            if not punctuation_insensitive(correct_answer_text):
                return None
            raw_accepted = raw.get("accepted_answers")
            candidates = [correct_answer_text]
            if isinstance(raw_accepted, list):
                candidates.extend(str(a).strip() for a in raw_accepted if str(a).strip())
            seen_answers: set[str] = set()
            for candidate in candidates:
                key = punctuation_insensitive(candidate)
                if key and key not in seen_answers:
                    seen_answers.add(key)
                    accepted_answers.append(candidate)
            # The answer must not be sitting in the question.
            if punctuation_insensitive(correct_answer_text) in prompt_norm:
                return None
        else:
            raw_options = raw.get("options")
            if not isinstance(raw_options, list):
                return None
            parsed: list[tuple[str, str]] = []
            for entry in raw_options:
                if not isinstance(entry, dict):
                    return None
                oid = str(entry.get("id") or "").strip().casefold()
                text = _WS_RE.sub(" ", str(entry.get("text") or "")).strip()
                if not oid or not text:
                    return None
                if _BANNED_OPTION_RE.match(text):
                    return None
                parsed.append((oid, text))

            correct_option_id = str(raw.get("correct_option_id") or "").strip().casefold()
            ids = [oid for oid, _ in parsed]
            texts = [punctuation_insensitive(text) for _, text in parsed]
            if len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
                return None
            if correct_option_id not in ids:
                return None

            if kind is QuestionKind.TRUE_FALSE:
                if [oid for oid, _ in parsed] != [oid for oid, _ in TRUE_FALSE_OPTIONS]:
                    return None
                parsed = list(TRUE_FALSE_OPTIONS)  # canonical labels, always
            else:
                if len(parsed) != len(MCQ_OPTION_IDS) or ids != list(MCQ_OPTION_IDS):
                    return None
                # A multiple-choice prompt that contains its winning option is
                # answerable without knowing anything.
                winner = next(text for oid, text in parsed if oid == correct_option_id)
                winner_key = punctuation_insensitive(winner)
                if len(winner_key) >= 12 and winner_key in prompt_norm:
                    return None

            options = [QuizOption(id=oid, text=text) for oid, text in parsed]

        seen_prompts.add(prompt_key)
        return QuizQuestion(
            id=new_id("qq"),
            quiz_id=quiz_id,
            paper_id=paper_id,
            index=0,  # assigned by the caller once the final order is known
            prompt=prompt,
            kind=kind,
            difficulty=difficulty,
            options=options,
            correct_option_id=correct_option_id,
            correct_answer_text=correct_answer_text or None,
            accepted_answers=accepted_answers,
            explanation=explanation,
            citation=source.citation(),
        )
    except Exception:  # noqa: BLE001 - a bad question is dropped, never fatal
        log.warning("quiz question failed validation", exc_info=True)
        return None


# ==========================================================================
# The agent
# ==========================================================================
_SYSTEM_HEADER = """\
You are writing a study quiz about ONE research paper, for a reader who has just \
read the study report below. Output MUST be a single JSON array of question \
objects and nothing else — no prose, no markdown, no code fence.

Each object has exactly these keys:
  "kind":                "multiple_choice" | "true_false" | "short_answer"
  "difficulty":          "recall" | "understand" | "apply"
  "prompt":              the question text (plain text; **bold** is allowed)
  "options":             [{"id": "a", "text": "..."}, ...]
                         exactly 4 with ids a,b,c,d for multiple_choice;
                         exactly [{"id":"true","text":"True"},{"id":"false","text":"False"}]
                         for true_false; [] for short_answer
  "correct_option_id":   the winning option id (multiple_choice / true_false), else null
  "correct_answer_text": the canonical answer (short_answer), else null
  "accepted_answers":    other acceptable short answers (synonyms, acronym and
                         spelled-out form); [] for the other kinds
  "explanation":         2-3 sentences saying WHY the correct answer is correct and,
                         for multiple choice, why the most tempting wrong option is
                         wrong. Never empty.
  "source_id":           the id of the SOURCE item below that supports the answer.

Hard rules:
- Every question MUST carry a "source_id" from the SOURCE list. If you cannot
  ground a question in one of those items, do not write it.
- Never state anything the SOURCE list does not support. Do not invent numbers,
  datasets, authors or results.
- Distractors must be drawn from this paper's own content and must be plausible.
  Never write "all of the above", "none of the above", or a joke option.
- Exactly one option is correct. The prompt must not contain its own answer.
- Mix the three kinds and the three difficulty tiers. "recall" = a fact stated in
  the paper; "understand" = connecting two facts or explaining why; "apply" =
  transferring the idea to a new situation.
- Keep prompts under 40 words and answers short.
"""


class QuizAgent(Agent[Quiz]):
    """Builds a :class:`~deepvision.models.study.Quiz` from persisted report sections."""

    def run(self, context: AgentContext) -> Quiz:
        """:class:`~deepvision.agents.base.Agent` entry point.

        Reads ``context.extra``: ``sections`` (required), ``quiz_id``,
        ``question_count``, ``kinds``, ``difficulties``, ``paper_title``.
        """
        return self.generate(
            paper_id=context.paper_id,
            sections=context.extra.get("sections") or [],
            quiz_id=context.extra.get("quiz_id") or new_id("quiz"),
            paper_title=context.extra.get("paper_title") or "",
            question_count=int(context.extra.get("question_count") or 10),
            kinds=context.extra.get("kinds") or list(QuestionKind),
            difficulties=context.extra.get("difficulties") or list(Difficulty),
            settings=context.settings,
        )

    def generate(
        self,
        *,
        paper_id: str,
        sections: Sequence[Section],
        quiz_id: Optional[str] = None,
        paper_title: str = "",
        question_count: int = 10,
        kinds: Sequence[QuestionKind] = tuple(QuestionKind),
        difficulties: Sequence[Difficulty] = tuple(Difficulty),
        settings: Optional[AppSettings] = None,
        on_progress: Optional[Callable[[int, str], None]] = None,
        extra_chunks: Sequence[AnyChunk] = (),
        avoid_prompts: frozenset[str] = frozenset(),
    ) -> Quiz:
        """Generate a validated quiz. **Never raises.**

        Returns a quiz with as many valid questions as the material supports —
        possibly zero, if the report has no usable prose yet. The caller decides
        what to tell the user about an empty quiz; this method does not guess.

        ``extra_chunks`` is the whole-paper source pool (see
        :func:`~deepvision.study.source_pool.pool_for_paper`) — optional, and an
        empty tuple degrades to the old report-only behaviour exactly.
        ``avoid_prompts`` is the raw prompt text of this paper's previous quiz
        questions; normalized here the same way :func:`_select_drafts` compares
        prompts, so a regenerated quiz stops re-asking what a past one already
        asked, as long as unseen material exists to replace it with.
        """
        quiz_id = quiz_id or new_id("quiz")
        question_count = max(1, min(int(question_count or 10), 30))
        rng = random.Random(f"{paper_id}:{quiz_id}")
        avoid_keys = frozenset(
            key
            for key in (punctuation_insensitive(p)[:200] for p in avoid_prompts)
            if key
        )

        def progress(percent: int, message: str) -> None:
            if on_progress is not None:
                try:
                    on_progress(percent, message)
                except Exception:  # noqa: BLE001 - progress is advisory
                    log.debug("quiz progress callback failed", exc_info=True)

        sources = collect_sources(sections, extra_chunks=extra_chunks)
        if not sources:
            log.warning(
                "no quiz source material in report", extra={"paper_id": paper_id}
            )
            return Quiz(
                id=quiz_id,
                paper_id=paper_id,
                title="Quiz · 0 questions",
                questions=[],
                model_used=self._model_name(settings),
            )
        progress(30, f"Found {len(sources)} citable passages")

        pool = _build_draft_pool(sources, rng)
        drafts = _select_drafts(
            pool,
            count=question_count,
            kinds=kinds,
            difficulties=difficulties,
            avoid_prompts=avoid_keys,
        )
        if not drafts:
            log.warning(
                "no draft quiz questions could be built",
                extra={"paper_id": paper_id, "sources": len(sources)},
            )
            return Quiz(
                id=quiz_id,
                paper_id=paper_id,
                title="Quiz · 0 questions",
                questions=[],
                model_used=self._model_name(settings),
            )

        progress(45, f"Drafted {len(drafts)} questions; asking the model to sharpen them")
        polished = self._polish(
            drafts,
            sources=sources,
            paper_title=paper_title,
            question_count=question_count,
            kinds=kinds,
            difficulties=difficulties,
        )

        by_id = {item.id: item for item in sources}
        seen_prompts: set[str] = set()
        questions: list[QuizQuestion] = []
        for raw in polished:
            if len(questions) >= question_count:
                break
            question = validate_draft(
                raw,
                sources=by_id,
                quiz_id=quiz_id,
                paper_id=paper_id,
                seen_prompts=seen_prompts,
            )
            if question is not None:
                questions.append(question)

        # Top up from the deterministic drafts if the model's output was thin or
        # largely rejected. The draft set is known-valid by construction, so this
        # is the floor under a bad generation.
        if len(questions) < question_count:
            for raw in drafts:
                if len(questions) >= question_count:
                    break
                question = validate_draft(
                    raw,
                    sources=by_id,
                    quiz_id=quiz_id,
                    paper_id=paper_id,
                    seen_prompts=seen_prompts,
                )
                if question is not None:
                    questions.append(question)

        for index, question in enumerate(questions):
            question.index = index

        difficulty_mix: dict[str, int] = {}
        kind_mix: dict[str, int] = {}
        for question in questions:
            difficulty_mix[question.difficulty.value] = (
                difficulty_mix.get(question.difficulty.value, 0) + 1
            )
            kind_mix[question.kind.value] = kind_mix.get(question.kind.value, 0) + 1

        progress(85, f"Validated {len(questions)} questions")
        return Quiz(
            id=quiz_id,
            paper_id=paper_id,
            title=f"Quiz · {len(questions)} question{'' if len(questions) == 1 else 's'}",
            questions=questions,
            difficulty_mix=difficulty_mix,
            kind_mix=kind_mix,
            model_used=self._model_name(settings),
        )

    # ---- internals ----------------------------------------------------
    @staticmethod
    def _model_name(settings: Optional[AppSettings]) -> Optional[str]:
        return getattr(settings, "llm_model", None) if settings else None

    def _polish(
        self,
        drafts: Sequence[Draft],
        *,
        sources: Sequence[SourceItem],
        paper_title: str,
        question_count: int,
        kinds: Sequence[QuestionKind],
        difficulties: Sequence[Difficulty],
    ) -> list[Draft]:
        """Ask the LLM to improve/extend ``drafts``; fall back to them verbatim.

        The deterministic drafts are the *last user message*, which is what
        :func:`~deepvision.agents.base.complete_with_fallback` returns when the
        provider errors — and what ``EchoLLM`` echoes — so the offline path and
        the failure path both round-trip a valid quiz through the same JSON
        parser as the online path. Anything unparseable falls back to the drafts.
        """
        source_lines = []
        for item in sources:
            label = f"{item.section.value}"
            if item.page:
                label += f", page {item.page}"
            prefix = f"{item.term}: " if item.term else ""
            source_lines.append(f'- {item.id} [{label}] {prefix}{item.body}')

        system = (
            _SYSTEM_HEADER
            + (f'\nPAPER: "{paper_title}"\n' if paper_title else "\n")
            + f"Write about {question_count} questions using these kinds "
            + f"({', '.join(k.value for k in kinds)}) and tiers "
            + f"({', '.join(d.value for d in difficulties)}).\n"
            + "\nSOURCE (the only material you may use):\n"
            + "\n".join(source_lines)
            + "\n\nThe user message is a mechanically generated draft of this quiz "
            "in the required JSON format. Rewrite it into better questions: clearer "
            "prompts, more plausible distractors, and explanations that actually "
            "teach. Keep the same JSON array format and the same number of "
            "questions, and keep each question's \"source_id\"."
        )
        # Internal bookkeeping keys (``_anchor``) are stripped: the model sees
        # exactly the schema the system prompt documents, nothing else.
        wire = [
            {k: v for k, v in draft.items() if not k.startswith("_")}
            for draft in drafts
        ]
        draft_json = json.dumps(wire, ensure_ascii=False, indent=1)
        max_tokens = max(1500, min(6000, 420 * max(1, question_count)))

        raw = complete_with_fallback(
            self.llm, system, draft_json, temperature=0.4, max_tokens=max_tokens
        )
        parsed = safe_json(raw, default=None)
        if isinstance(parsed, dict):
            # A model that wrapped the array in {"questions": [...]}.
            for key in ("questions", "quiz", "items", "data"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
        if not isinstance(parsed, list) or not parsed:
            log.info("quiz polish produced no usable JSON; using the deterministic draft")
            return list(drafts)
        return [item for item in parsed if isinstance(item, dict)] or list(drafts)
