"""Flashcard agent — turns a persisted report into a study deck.

Follows the same contract every other agent in this package does: build a
**deterministic, grounded draft first**, then hand it to the LLM only to
*polish*. That is what makes the deck safe. With ``EchoLLM``, a missing Ollama,
or a provider timeout, the draft ships — a real deck extracted from the paper's
own Key Concepts, Key Takeaways, Key Results, Methods and Limitations & Open
Questions — rather than a crash, a 500 or an empty screen. :func:`build_deck`
never raises.

**There is no more Glossary source.** The report's Glossary section was
deleted (it produced near-duplicate term/definition pairs against Key
Concepts — see ``deepvision.models.report.SectionName``), so the deck's
vocabulary cards now come from Key Concepts alone.
:class:`~deepvision.models.study.FlashcardOrigin` keeps its ``glossary`` member
— cards already persisted with that origin from before this change are real
data on the wire and must keep validating — but no section produces it any
more; new decks simply never mint that origin.

**Why the deck is not just term/definition pairs.** A deck of nothing but
those is a vocabulary list: you can ace it without having understood the
paper. So cards are drawn from several places and interleaved round-robin,
which guarantees a mix of
:class:`~deepvision.models.study.FlashcardOrigin` values even when one source
is far richer than the others:

===============  ===============================================================
``key_concept``  A Key Concepts entry, rephrased as a question.
``takeaway``     A Key Takeaways bullet, turned into a recall prompt.
``fact``         A number/dataset/baseline from Key Results, Methods or
                 Limitations & Open Questions, as a fill-in-the-blank cloze.
                 These are the cards that make the deck about *this paper*
                 rather than about the field.
===============  ===============================================================

**Provenance** comes free: section bodies carry inline ``[n]`` markers that the
section's own :class:`~deepvision.models.report.Citation` list resolves to a page
and a chunk. Each card records the section it came from plus that page/chunk, so
the UI can offer "show me where this came from". The markers themselves are
stripped from the card text — a flashcard has no citation list to resolve them
against, so they would render as literal ``[3]`` noise.

**The LLM pass** rewrites only the *backs*, in a strict ``n :: text`` line
format, and is discarded wholesale if the reply does not come back with exactly
the same line numbers. A model that decides to merge two cards, drop one, or
answer a different question cannot corrupt the deck: the deterministic draft is
already correct, and any parse mismatch keeps it.

**The report is no longer the whole world.** ``build_deck`` also accepts
``extra_chunks`` — a whole-paper pool built by
:func:`deepvision.study.source_pool.pool_for_paper` — and mines definition,
numeric-fact and general-recall cards straight out of the paper's own prose
(:func:`_chunk_definition_cards`, :func:`_chunk_fact_cards`,
:func:`_chunk_recall_cards`), each sentence checked with
:func:`~deepvision.rag.chunk_quality.is_boilerplate` and
:func:`~deepvision.rag.chunk_quality.prose_score` before it can become a card —
the direct fix for cards that used to front on reference-list junk like
``**AI**`` or ``**CVF**``. Report-derived cards are still preferred (they carry
a citation), but pool cards are interleaved alongside them rather than
appended after, so a deck is a mix rather than "report cards, then paper
cards". ``avoid_keys`` (the content keys a persisted deck already has) does
not drop repeats — it pushes them behind unseen material so a regeneration
surfaces new cards first without ever emptying the deck when everything has
technically been seen before.

**Template monotony is broken by hashing, not by rolling dice.** Card types
that used to have exactly one phrasing (``_recall_front``, `_term_cards``'s
question form, the numeric-cloze wrapper) now have 7-8 interchangeable ones,
picked by :func:`_stable_choice` — a deterministic hash of the card's own
content, never :mod:`random`. The same underlying claim always picks the same
phrasing, so its ``content_key`` — and therefore its SRS schedule — is stable
across regenerations.

**Why hashing and not rotation.** ``quiz_agent`` widened its stems the other
way, rotating by the item's index (``_stem_rotate``), which spreads strictly
better: rotation cannot repeat a stem until the pool is exhausted, while
independent hashing collides. A deck cannot do that. A card's index depends on
what the retrieval pool returned that run, so rotation would re-word a card
whenever its neighbours changed — and the wording is inside ``content_key``, so
re-wording orphans the SM-2 schedule. Hashing the card's *own content* is what
makes the phrasing survive a regeneration. The cost is the occasional repeated
stem within a deck, which is the right trade.

**New stems reach new cards only** (:func:`_widen_new_card_phrasing`). Adding a
template changes ``len(options)`` and therefore every existing card's
``_stable_choice`` result, which would re-key whole decks. So each front is
rendered from the original stem set first, and only widened if that front's key
is not already in the deck. Full contract in that function's docstring.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Iterable, Optional, Sequence

from deepvision.agents.base import Agent, AgentContext, complete_with_fallback
from deepvision.models.chunks import AnyChunk
from deepvision.models.report import Citation, Report, Section, SectionName
from deepvision.models.study import Flashcard, FlashcardOrigin
from deepvision.providers.base import LLMProvider
from deepvision.rag.chunk_quality import (
    is_boilerplate,
    is_caption_sentence,
    is_ordinal_number,
    prose_score,
)
from deepvision.utils.ids import new_id
from deepvision.utils.logger import get_logger

__all__ = [
    "FlashcardAgent",
    "build_deck",
    "content_key_for",
    "normalize_front",
]

log = get_logger(__name__)

#: Deck size guard rails. The request's ``count`` is the target; these bound it
#: so a caller cannot ask for a 3-card deck (not worth a review session) or a
#: 200-card one (nobody finishes it, and it is mostly padding by then).
MIN_DECK_CARDS = 8
MAX_DECK_CARDS = 60

#: A back shorter than this is not an answer, it is a fragment.
_MIN_BACK_CHARS = 20
#: Backs longer than this do not stick; the draft truncates at a sentence end.
_MAX_BACK_CHARS = 320
_MAX_FRONT_CHARS = 240

_WS_RE = re.compile(r"\s+")
#: Inline citation marker, e.g. ``[3]``. Stripped from card text, but read first
#: for the page/chunk provenance it points at.
_MARKER_RE = re.compile(r"\[(\d{1,3})\]")
#: ``**Term** — definition`` (em dash, en dash, hyphen or colon separator), with
#: an optional leading list bullet or ``1.`` ordinal.
_BOLD_PAIR_RE = re.compile(
    r"^\s*(?:[-*+•]\s+|\d+[.)]\s+)?\*\*(?P<term>[^*\n]{2,80})\*\*\s*(?:[—–:-]\s*)?(?P<body>.+)$",
    re.DOTALL,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*+•]\s+|\d+[.)]\s+)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
#: A number worth blanking out: 12, 3.4, 91.2%, 1.5x, 100k, $2M. The optional
#: space lives *inside* the unit group on purpose — pulled out of it, the regex
#: eats the space after a bare number ("41.8 BLEU" → "____BLEU").
_NUMBER_RE = re.compile(r"\$?\d+(?:[.,]\d+)?(?:\s?(?:%|×|x\b|k\b|M\b|B\b))?")
#: A whole bracketed citation group, single or multi: ``[3]``, ``[7, 24, 15]``,
#: ``[2; 9]``. Broader than :data:`_MARKER_RE`, which stays single-number
#: because :func:`_first_marker` resolves it to one citation for provenance.
_CITATION_GROUP_RE = re.compile(r"\[\s*\d{1,3}(?:\s*[,;]\s*\d{1,3})*\s*\]")
#: Any bracketed span at all — used only to veto blanking a number inside one.
_BRACKET_SPAN_RE = re.compile(r"\[[^\]\n]{0,80}\]")
#: A URL, a file name or a bare footnote link. Real cards shipped fronted
#: "Complete this sentence from the paper: ____http://www.collectspace.com/
#: images/news-091516d-lg.jpg" — a footnote marker glued to an image URL,
#: blanked as if it were a finding. Nothing worth studying is a link.
_URLISH_RE = re.compile(
    r"https?://|www\.\w|\.(?:com|org|net|edu|gov|io|jpg|jpeg|png|gif|pdf)\b",
    re.IGNORECASE,
)
#: Sentences that are pure boilerplate make terrible cards.
_BOILERPLATE_RE = re.compile(
    r"^(?:no |this section|the (?:report|section)|re-?generate|not enough|"
    r"insufficient|could not be)",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# Phrasing variety — deterministic, never `random` (content_key must be
# stable across regenerations, so the same input always picks the same
# template; see the module docstring).
# --------------------------------------------------------------------------
_TERM_QUESTION_TEMPLATES: tuple[str, ...] = (
    "What is **{term}**, and why does it matter in this paper?",
    "How would you explain **{term}** to someone new to this paper?",
    "In this paper's context, what does **{term}** mean?",
    "What role does **{term}** play in this paper?",
)
_BOLD_SUBJECT_TEMPLATES: tuple[str, ...] = (
    "What does the paper claim about **{subject}**?",
    "According to the paper, what is true of **{subject}**?",
    "What point does the paper make regarding **{subject}**?",
    "How does the paper characterize **{subject}**?",
)
_NUMBER_BLANK_TEMPLATES: tuple[str, ...] = (
    "Fill in the blank: {blanked}",
    "Complete this sentence from the paper: {blanked}",
    "What number completes this claim? {blanked}",
)
_LEAD_WORDS_TEMPLATES: tuple[str, ...] = (
    "Complete this {label} from the paper: “{lead} …”",
    "How does this {label} continue? “{lead} …”",
    "Finish the thought: “{lead} …”",
)

# --------------------------------------------------------------------------
# Wider phrasing — appended to the tuples above, but ONLY for cards the deck
# has not produced before. See `_widen_new_card_phrasing` for why that gate
# exists and why these cannot simply be merged into the tuples above.
#
# Measured need: across four real decks, numeric-cloze and lead-words fronts
# were 39% and 36% of every card — three quarters of a deck drawn from six
# stems between them — so those two families get the most new phrasings. The
# additions are different *framings* (motivation, counterfactual, imperative),
# not synonyms of the existing question, because swapping "what" for "which"
# does not make a review session feel less repetitive.
# --------------------------------------------------------------------------
_TERM_QUESTION_EXTRA: tuple[str, ...] = (
    "Why does this paper need **{term}**?",
    "Define **{term}** as this paper uses it.",
    "Where does **{term}** fit into the paper's approach?",
    "What would be missing from this paper without **{term}**?",
)
_BOLD_SUBJECT_EXTRA: tuple[str, ...] = (
    "What is this paper's position on **{subject}**?",
    "State what this paper establishes about **{subject}**.",
    "What is worth remembering about **{subject}**?",
    "Recall the paper's finding on **{subject}**.",
)
_NUMBER_BLANK_EXTRA: tuple[str, ...] = (
    "Supply the missing figure: {blanked}",
    "The paper reports a specific number here — what is it? {blanked}",
    "Recall the exact value: {blanked}",
    "From memory, fill this in: {blanked}",
)
_LEAD_WORDS_EXTRA: tuple[str, ...] = (
    "Continue this {label} in the paper's own terms: “{lead} …”",
    "What comes next? “{lead} …”",
    "Recall the rest of this {label}: “{lead} …”",
    "Pick up where this leaves off: “{lead} …”",
)

#: family -> (legacy stems, extra stems). The split is load-bearing: see
#: `_widen_new_card_phrasing`.
_STEM_FAMILIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "term": (_TERM_QUESTION_TEMPLATES, _TERM_QUESTION_EXTRA),
    "bold": (_BOLD_SUBJECT_TEMPLATES, _BOLD_SUBJECT_EXTRA),
    "number": (_NUMBER_BLANK_TEMPLATES, _NUMBER_BLANK_EXTRA),
    "lead": (_LEAD_WORDS_TEMPLATES, _LEAD_WORDS_EXTRA),
}


def _render_stem(
    family: str, seed: str, params: dict[str, str], *, widened: bool
) -> str:
    """Render one card front from ``family``, hashing ``seed`` to pick the stem.

    ``widened=False`` reproduces exactly what this card would have been called
    before the extra stems existed — that is what makes the legacy phrasing
    recoverable, and therefore what makes the whole scheme safe.
    """
    legacy, extra = _STEM_FAMILIES[family]
    options = legacy + extra if widened else legacy
    return _stable_choice(seed, options).format(**params)


def _stable_choice(seed: str, options: Sequence[str]) -> str:
    """Deterministically pick one of ``options`` from ``seed``.

    Hashing, not :mod:`random`: the same claim/term must always land on the
    same phrasing so its ``content_key`` (derived from the resulting front)
    never drifts between regenerations — a changing key would orphan the
    card's SM-2 schedule, which is the one thing this feature must not do.
    """
    if not options:
        return ""
    digest = hashlib.sha1((seed or "").casefold().encode("utf-8")).hexdigest()
    return options[int(digest, 16) % len(options)]


# --------------------------------------------------------------------------
# The whole-paper pool — material outside the report (see the module
# docstring and `deepvision.study.source_pool`).
# --------------------------------------------------------------------------
#: Mirrors `rag.chunk_quality`'s own internal prose-score floor (not
#: exported — it is a private tuning constant there). A sentence below this
#: reads like OCR noise or a reference-list entry, not paper prose.
_MIN_CHUNK_PROSE = 0.15
#: A bare pronoun is never a real definition subject ("This is defined as...").
_WEAK_DEFINITION_SUBJECTS = frozenset(
    {"this", "it", "that", "these", "those", "there", "which", "who", "here"}
)
#: ``Term is/are defined as ...`` / ``Term refers to ...`` / ``We define Term
#: as ...`` and near variants. Conservative on purpose — a false match just
#: means a card that misattributes its "term", so patterns stay narrow.
_DEFINITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?P<term>[A-Z][\w][\w\-\s]{1,60}?)\s+(?:is|are)\s+defined\s+as\s+(?P<body>.+)$"
    ),
    re.compile(r"^(?P<term>[A-Z][\w][\w\-\s]{1,60}?)\s+refers\s+to\s+(?P<body>.+)$"),
    re.compile(r"^(?P<term>[A-Z][\w][\w\-\s]{1,60}?)\s+denotes\s+(?P<body>.+)$"),
    re.compile(
        r"^We\s+(?:define|call|refer to)\s+(?P<term>.{2,60}?)\s+as\s+(?P<body>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<term>[A-Z][\w][\w\-\s]{1,60}?),?\s+"
        r"(?:also\s+known\s+as|i\.e\.,?)\s+(?P<body>.+)$",
        re.IGNORECASE,
    ),
)

#: Every section the deck may draw on, and the origin each contributes. The
#: Glossary row is gone with the section (see the module docstring); Limitations
#: & Open Questions is new — "what does this paper NOT show" is exactly the
#: kind of paper-specific claim the ``fact`` origin already exists for (Key
#: Results / Methods), so it reuses that origin and the same numeric-cloze
#: extraction (``_fact_cards``) rather than inventing a new one.
_SOURCES: list[tuple[SectionName, FlashcardOrigin]] = [
    (SectionName.KEY_CONCEPTS, FlashcardOrigin.KEY_CONCEPT),
    (SectionName.KEY_TAKEAWAYS, FlashcardOrigin.TAKEAWAY),
    (SectionName.KEY_RESULTS, FlashcardOrigin.FACT),
    (SectionName.METHODS, FlashcardOrigin.FACT),
    (SectionName.LIMITATIONS, FlashcardOrigin.FACT),
]

_POLISH_SYSTEM = (
    "You are writing the ANSWER side of study flashcards for a research paper, "
    "for a reader with no background in the field. Each input line is "
    "'<number> :: <answer draft>'. Rewrite ONLY the text after the '::' into a "
    "crisp, self-contained answer of one or two sentences in plain language. "
    "Output exactly one line per input line, in the same order, keeping the "
    "same leading number and the same ' :: ' separator. Do not add lines, do "
    "not drop lines, do not merge lines, do not number differently, and never "
    "state anything the draft does not support. Use no markdown other than "
    "**bold**: no headings, no bullets, no tables, no links, no code fences."
)


# --------------------------------------------------------------------------
# Small text helpers
# --------------------------------------------------------------------------
def _collapse(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def _strip_markers(text: str) -> str:
    """Remove inline citation brackets, single (``[3]``) or grouped (``[7, 24]``).

    Grouped markers matter as much as single ones: leaving ``[24, 15]`` in the
    text let the numeric-cloze extractor blank a reference-list index and ask
    the reader to recall it.
    """
    return (
        _collapse(_CITATION_GROUP_RE.sub("", text or ""))
        .replace(" .", ".")
        .replace(" ,", ",")
        .strip()
    )


def _blank_number(text: str) -> Optional[tuple[str, str]]:
    """Blank the first number in ``text`` that is actually worth recalling.

    Returns ``(blanked_text, value)``, or ``None`` when the sentence holds no
    number worth asking about — in which case the caller must fall back to a
    different card shape rather than ship a nonsense cloze.

    "Blank whichever number the regex hits first" was not good enough, and
    shipped decks are the evidence. Real cards asked readers to fill in a
    bibliography index ("...through factorization tricks [____] and conditional
    computation"), a figure ordinal ("Figure ____: The Transformer - model
    architecture") and the year buried inside a dataset name
    ("...development set, newstest____"). Not one of those is a fact about the
    paper; each one is a pointer or part of a name.

    Three rules reject them, and the first number surviving all three wins:

    - inside a bracket → an index into the reference list, not a quantity;
    - preceded by "Figure"/"Table"/"Section"/... → an ordinal label;
    - glued to the end of a word ("newstest2013", "WMT2014") → part of that
      token, and blanking it leaves a mutilated word.
    """
    if not text:
        return None
    bracket_spans = [m.span() for m in _BRACKET_SPAN_RE.finditer(text)]
    for match in _NUMBER_RE.finditer(text):
        value = match.group(0).strip()
        if not any(ch.isdigit() for ch in value):
            continue
        start, end = match.span()
        if any(bs <= start < be for bs, be in bracket_spans):
            continue
        if is_ordinal_number(text, start):
            continue
        prev = text[start - 1] if start > 0 else ""
        if prev.isalpha() or prev == "_":
            continue
        # Hyphen-joined names are one token too: "TG-2", "GPT-4", "COVID-19".
        # Blanking the tail mutilates a proper noun and asks the reader to
        # recall a model number rather than a finding. A real card shipped as
        # "The installation diagram of POLAR on TG-____ 1 is shown in ...".
        if prev in "-–" and start > 1 and text[start - 2].isalnum():
            continue
        blanked = (text[:start] + "____" + text[end:]).strip()
        return blanked, value
    return None


def _first_marker(text: str) -> Optional[int]:
    match = _MARKER_RE.search(text or "")
    return int(match.group(1)) if match else None


def _truncate(text: str, limit: int) -> str:
    """Cut at a sentence boundary where possible, a word boundary otherwise."""
    text = _collapse(text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for punct in (". ", "! ", "? "):
        idx = cut.rfind(punct)
        if idx > limit * 0.5:
            return cut[: idx + 1].strip()
    idx = cut.rfind(" ")
    if idx > 0:
        cut = cut[:idx]
    return cut.rstrip(",;:- ") + "…"


def normalize_front(front: str) -> str:
    """Normalized form used for de-duplication and for ``content_key``.

    Casefolded, markdown emphasis removed, punctuation-insensitive — so
    "**Attention**", "attention?" and "Attention" are one card, and a
    regeneration whose wording drifts only in punctuation still matches its
    existing row (and therefore keeps its schedule).
    """
    text = _collapse(front or "").replace("**", "").replace("*", "")
    text = re.sub(r"[^\w\s]", " ", text)
    return _WS_RE.sub(" ", text).strip().casefold()


def content_key_for(paper_id: str, origin: FlashcardOrigin, front: str) -> str:
    """``sha1(paper_id|origin|normalized front)[:16]`` — the upsert key.

    Not an id: it is the ``(paper_id, content_key)`` unique key that lets a
    regenerated deck recognise a card it has produced before and leave that
    card's SM-2 schedule alone.
    """
    raw = f"{paper_id}|{origin.value}|{normalize_front(front)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _blocks(body: str) -> list[str]:
    """Split a section body into candidate entries.

    Handles both shapes the report writes: blank-line separated paragraphs (what
    the study agent emits) and single-line markdown bullets (what a polished
    Key Takeaways list emits).
    """
    out: list[str] = []
    for para in re.split(r"\n\s*\n", body or ""):
        para = para.strip()
        if not para:
            continue
        lines = [ln for ln in para.split("\n") if ln.strip()]
        # A paragraph made of bullets is several entries, not one.
        if len(lines) > 1 and all(_BULLET_RE.match(ln) for ln in lines):
            out.extend(_collapse(_BULLET_RE.sub("", ln)) for ln in lines)
        else:
            out.append(_collapse(_BULLET_RE.sub("", para, count=1)))
    return [block for block in out if block]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]


def _is_usable(text: str) -> bool:
    """Reject draft scaffolding and figure/table captions.

    The caption check is the single choke point that keeps captions out of the
    deck no matter which extractor produced the card: every candidate passes
    through :meth:`_Candidate.is_useful`, which calls this on both sides. That
    matters because captions do not only arrive from the pool — they are baked
    into the *persisted* body of reports generated before the report-side
    filters landed, and those rows are not going to rewrite themselves. A card
    fronted "How does the paper characterize **Figure 5: Many of the
    attention**?" was shipping from exactly that path.
    """
    if not text or _BOILERPLATE_RE.match(text):
        return False
    if _URLISH_RE.search(text):
        return False
    return not is_caption_sentence(text)


def _usable_sentence(text: str) -> bool:
    """Gate for any sentence pulled from the whole-paper pool before it can
    become a card front or back.

    This is the direct fix for cards that used to front on reference-list
    junk (``**AI**``, ``**BS**``, ``**CVF**`` — acronyms mined out of a
    bibliography) and on quoted reference entries: :func:`is_boilerplate`
    catches furniture/citation-shaped text, :func:`prose_score` catches OCR
    noise and the low function-word density typical of a "Author, A. In
    Proceedings of ..." line.
    """
    text = (text or "").strip()
    if len(text) < 30 or not _is_usable(text):
        return False
    if is_boilerplate(text):
        return False
    return prose_score(text) >= _MIN_CHUNK_PROSE


# --------------------------------------------------------------------------
# Candidate cards
# --------------------------------------------------------------------------
class _Candidate:
    """A card before de-duplication, quota trimming and LLM polish."""

    __slots__ = (
        "front",
        "back",
        "hint",
        "origin",
        "section",
        "page",
        "chunk_id",
        "tags",
        "cloze_value",
        "stem",
    )

    def __init__(
        self,
        *,
        front: str,
        back: str,
        origin: FlashcardOrigin,
        section: Optional[SectionName] = None,
        hint: Optional[str] = None,
        page: Optional[int] = None,
        chunk_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        cloze_value: Optional[str] = None,
        stem: Optional[tuple[str, str, dict[str, str]]] = None,
    ) -> None:
        self.front = _truncate(front, _MAX_FRONT_CHARS)
        self.back = _truncate(back, _MAX_BACK_CHARS)
        self.hint = hint
        #: For a ``____`` front, the value the back must keep in order to
        #: answer it. Guards the LLM polish step — see :func:`_polish_backs`.
        self.cloze_value = cloze_value
        #: ``(family, seed, params)`` for a front built from a stem family, so
        #: :func:`_widen_new_card_phrasing` can re-render it from the wider
        #: pool. ``None`` for a front that is not template-driven (a bare
        #: glossary term), which is then left exactly as it is.
        self.stem = stem
        self.origin = origin
        self.section = section
        self.page = page
        self.chunk_id = chunk_id
        self.tags = tags or []

    def is_useful(self) -> bool:
        """Reject empty backs, stubs, and backs that merely restate the front.

        A card whose answer is its own question teaches nothing but still costs
        a review slot every day, so it is dropped rather than shown.
        """
        if not self.front or not self.back:
            return False
        if len(self.back) < _MIN_BACK_CHARS:
            return False
        front_norm = normalize_front(self.front)
        back_norm = normalize_front(self.back)
        if not front_norm or not back_norm:
            return False
        if back_norm == front_norm or back_norm in front_norm:
            return False
        return _is_usable(self.front) and _is_usable(self.back)


def _citation_index(section: Section) -> dict[int, Citation]:
    return {c.marker: c for c in (section.citations or []) if c.marker}


def _provenance(
    raw_text: str, citations: dict[int, Citation]
) -> tuple[Optional[int], Optional[str]]:
    """Resolve the entry's first ``[n]`` marker to ``(page, chunk_id)``."""
    marker = _first_marker(raw_text)
    if marker is None:
        return None, None
    citation = citations.get(marker)
    if citation is None:
        return None, None
    return citation.page, citation.chunk_id


def _hint_for(section: SectionName, page: Optional[int]) -> str:
    where = f"{section.value}"
    return f"See **{where}**" + (f", page {page}." if page else ".")


def _pool_hint(page: Optional[int]) -> str:
    """Hint for a card mined from the whole-paper pool (no report section)."""
    return f"From the paper, page {page}." if page else "From the paper."


#: A "term" that trails off into a bare caption label — "Scaled Dot-Product
#: Attention Multi-Head Attention Figure" — is two captions run together by the
#: summarizer's bold lead-in, not a concept. It escapes `is_caption_sentence`
#: because the label is at the *end* and carries no number.
_TRAILING_LABEL_RE = re.compile(
    r"\b(?:fig(?:ure)?|table|chart|plate|scheme|eq(?:uation)?|appendix)\s*$",
    re.IGNORECASE,
)


def _is_real_subject(term: str) -> bool:
    """True if ``term`` can front a card as the thing being asked about."""
    term = (term or "").strip()
    return bool(term) and not _TRAILING_LABEL_RE.search(term)


def _term_question(term: str) -> str:
    """A Key-Concepts-style question front, phrasing picked by :func:`_stable_choice`."""
    return _render_stem("term", term, {"term": term}, widened=False)


# --------------------------------------------------------------------------
# Per-origin extraction
# --------------------------------------------------------------------------
def _term_cards(
    section: Section, origin: FlashcardOrigin, *, as_question: bool
) -> list[_Candidate]:
    """Cards from a ``**Term** — explanation`` section (Glossary / Key Concepts).

    Glossary cards keep the bare term on the front (that is the recall the
    glossary exists to drill). Key Concepts cards are rephrased as a question,
    because "Self-attention" as a prompt invites a one-word answer while "What
    is self-attention, and why does the paper need it?" invites the explanation
    the section actually contains.
    """
    citations = _citation_index(section)
    out: list[_Candidate] = []
    for block in _blocks(section.body_markdown):
        match = _BOLD_PAIR_RE.match(block)
        if not match:
            continue
        term = _strip_markers(match.group("term"))
        definition = _strip_markers(match.group("body"))
        if not definition or not _is_real_subject(term):
            continue
        page, chunk_id = _provenance(block, citations)
        # A bare glossary front is the term itself, not a template, so it has
        # no stem to widen — and must not grow one, or its key would move.
        front, stem = (
            _stemmed("term", term, {"term": term})
            if as_question
            else (f"**{term}**", None)
        )
        out.append(
            _Candidate(
                front=front,
                back=definition,
                origin=origin,
                section=section.name,
                hint=_hint_for(section.name, page),
                page=page,
                chunk_id=chunk_id,
                tags=[origin.value],
                stem=stem,
            )
        )
    return out


def _takeaway_cards(section: Section) -> list[_Candidate]:
    """Cards from Key Takeaways bullets — each bullet is one claim to recall.

    The prompt is chosen so it cannot contain its own answer, which is the easy
    way to get this wrong (asking "what does the paper claim about *<the entire
    claim>*?" is not a flashcard). In descending order of quality:

    1. the claim names a **bold** subject → ask about that subject;
    2. otherwise it contains a number → blank the number out (the number is
       almost always the part worth remembering);
    3. otherwise → give the opening few words and ask for the rest.
    """
    citations = _citation_index(section)
    out: list[_Candidate] = []
    for block in _blocks(section.body_markdown):
        claim = _strip_markers(block)
        if len(claim) < _MIN_BACK_CHARS:
            continue
        page, chunk_id = _provenance(block, citations)
        front, stem = _recall_front(claim, label="takeaway")
        out.append(
            _Candidate(
                front=front,
                back=claim,
                origin=FlashcardOrigin.TAKEAWAY,
                section=section.name,
                hint=_hint_for(section.name, page),
                page=page,
                chunk_id=chunk_id,
                tags=["takeaway"],
                stem=stem,
            )
        )
    return out


#: A rendered front plus the ``(family, seed, params)`` needed to re-render it.
_Stemmed = tuple[str, Optional[tuple[str, str, dict[str, str]]]]


def _stemmed(family: str, seed: str, params: dict[str, str]) -> _Stemmed:
    """Render the legacy front and hand back what is needed to widen it later."""
    return _render_stem(family, seed, params, widened=False), (family, seed, params)


def _recall_front(claim: str, *, label: str) -> _Stemmed:
    """Turn one claim into a prompt that withholds the claim itself.

    Shared by the takeaway and abstract-fallback paths, because the failure mode
    is the same in both: the obvious template ("what does the paper say about
    <claim>?") quotes the answer straight back at the reader.
    """
    bold = re.search(r"\*\*([^*]{2,60})\*\*", claim)
    if bold:
        subject = bold.group(1).strip().rstrip(".:;,")
        # Only useful if the bold span is a subject, not the whole sentence.
        if subject and len(subject.split()) <= 6 and _is_real_subject(subject):
            return _stemmed("bold", subject, {"subject": subject})

    plain = claim.replace("**", "")
    blanked_pair = _blank_number(plain)
    if blanked_pair is not None:
        blanked, _value = blanked_pair
        return _stemmed("number", blanked, {"blanked": blanked})

    words = _collapse(plain).split()
    lead = " ".join(words[:5]).rstrip(".:;,")
    if lead and len(words) > 6:
        return _stemmed("lead", lead, {"label": label, "lead": lead})
    return "What is one of the paper's headline claims?", None


def _fact_cards(section: Section, *, limit: int) -> list[_Candidate]:
    """Cloze cards from the concrete numbers in Key Results / Methods.

    A sentence with a number in it is the cheapest reliable signal that a
    passage is about *this* paper — a score, a dataset size, a layer count, an
    ablation delta. The number is blanked out and the sentence becomes the
    prompt; the back restores it. No LLM is needed to know which token matters.
    """
    citations = _citation_index(section)
    out: list[_Candidate] = []
    seen: set[str] = set()
    for block in _blocks(section.body_markdown):
        # A paragraph usually carries its ``[n]`` marker once, at the end, so a
        # per-sentence lookup finds nothing for every sentence but the last.
        # Fall back to the paragraph's marker rather than dropping the page.
        block_page, block_chunk = _provenance(block, citations)
        for sentence in _sentences(block):
            if len(out) >= limit:
                return out
            clean = _strip_markers(sentence)
            if len(clean) < 45 or len(clean) > 300 or not _is_usable(clean):
                continue
            blanked_pair = _blank_number(clean)
            if blanked_pair is None:
                continue
            blanked, value = blanked_pair
            key = normalize_front(blanked)
            if key in seen:
                continue
            seen.add(key)
            page, chunk_id = _provenance(sentence, citations)
            if page is None:
                page, chunk_id = block_page, block_chunk
            front, stem = _stemmed("number", blanked, {"blanked": blanked})
            out.append(
                _Candidate(
                    front=front,
                    back=clean,
                    origin=FlashcardOrigin.FACT,
                    section=section.name,
                    hint=_hint_for(section.name, page),
                    page=page,
                    chunk_id=chunk_id,
                    tags=["fact", section.name.value.lower().replace(" ", "-")],
                    cloze_value=value,
                    stem=stem,
                )
            )
    return out


def _abstract_cards(title: str, abstract: str) -> list[_Candidate]:
    """Last-resort deck source: the paper's own abstract.

    Reached only when the report has no usable content at all (never generated,
    or generated before the model was available). A thin deck built from the
    abstract is still better than an empty screen and a mystery, and every card
    is honestly attributed to the Overview.
    """
    out: list[_Candidate] = []
    sentences = [s for s in _sentences(_collapse(abstract)) if len(s) >= 60]
    where = f"From the abstract of **{title}**." if title else "From the paper's abstract."
    for sentence in sentences[:10]:
        front, stem = _recall_front(sentence, label="statement")
        out.append(
            _Candidate(
                front=front,
                back=sentence,
                origin=FlashcardOrigin.FACT,
                section=SectionName.OVERVIEW,
                hint=where,
                tags=["abstract"],
                stem=stem,
            )
        )
    return out


# --------------------------------------------------------------------------
# Whole-paper pool — cards mined from raw chunks, not the report
# --------------------------------------------------------------------------
def _chunk_text(chunk: AnyChunk) -> str:
    return getattr(chunk, "text", "") or ""


def _chunk_page(chunk: AnyChunk) -> Optional[int]:
    return getattr(chunk, "page", None)


def _chunk_definition_cards(
    chunks: Sequence[AnyChunk], *, limit: int
) -> list[_Candidate]:
    """``Term is defined as ...`` / ``Term refers to ...`` style sentences.

    These are the pool's stand-in for Key Concepts: the report only ever sees
    the summarizer's rephrasing of a handful of terms, but the paper itself
    states many more definitions verbatim. Every sentence is gated by
    :func:`_usable_sentence` first, so a reference-list line never becomes a
    term.
    """
    out: list[_Candidate] = []
    seen_terms: set[str] = set()
    for chunk in chunks:
        if len(out) >= limit:
            break
        text = _chunk_text(chunk)
        if not text.strip() or is_boilerplate(text):
            continue
        for sentence in _sentences(text):
            if len(out) >= limit:
                break
            clean = _collapse(sentence)
            if not _usable_sentence(clean):
                continue
            match = None
            for pattern in _DEFINITION_PATTERNS:
                match = pattern.match(clean)
                if match:
                    break
            if not match:
                continue
            term = _collapse(match.group("term")).strip(" ,.;:-")
            term = re.sub(r"^(?:the|a|an)\s+", "", term, flags=re.IGNORECASE)
            body = _collapse(match.group("body")).strip()
            if not term or len(term.split()) > 8:
                continue
            if term.casefold() in _WEAK_DEFINITION_SUBJECTS:
                continue
            if len(body) < _MIN_BACK_CHARS:
                continue
            norm_term = term.casefold()
            if norm_term in seen_terms:
                continue
            seen_terms.add(norm_term)
            front, stem = _stemmed("term", term, {"term": term})
            out.append(
                _Candidate(
                    front=front,
                    back=body,
                    origin=FlashcardOrigin.KEY_CONCEPT,
                    section=None,
                    hint=_pool_hint(_chunk_page(chunk)),
                    page=_chunk_page(chunk),
                    chunk_id=chunk.id,
                    tags=["key_concept", "paper-pool"],
                    stem=stem,
                )
            )
    return out


def _chunk_fact_cards(chunks: Sequence[AnyChunk], *, limit: int) -> list[_Candidate]:
    """Numeric-cloze cards from sentences anywhere in the paper.

    Same idea as :func:`_fact_cards`, but swept across the whole document
    instead of just Key Results / Methods / Limitations — the report only
    keeps a handful of numbers, the paper has dozens.
    """
    out: list[_Candidate] = []
    seen: set[str] = set()
    for chunk in chunks:
        if len(out) >= limit:
            break
        text = _chunk_text(chunk)
        if not text.strip() or is_boilerplate(text):
            continue
        for sentence in _sentences(text):
            if len(out) >= limit:
                break
            # Strip citation brackets here too, not just collapse whitespace.
            # `_fact_cards` has always done this; skipping it on the pool path
            # is what let ``[24, 15]`` survive to become the blanked token.
            clean = _strip_markers(sentence)
            if len(clean) < 45 or len(clean) > 300:
                continue
            if not _usable_sentence(clean):
                continue
            blanked_pair = _blank_number(clean)
            if blanked_pair is None:
                continue
            blanked, value = blanked_pair
            key = normalize_front(blanked)
            if key in seen:
                continue
            seen.add(key)
            front, stem = _stemmed("number", blanked, {"blanked": blanked})
            out.append(
                _Candidate(
                    front=front,
                    back=clean,
                    origin=FlashcardOrigin.FACT,
                    section=None,
                    hint=_pool_hint(_chunk_page(chunk)),
                    page=_chunk_page(chunk),
                    chunk_id=chunk.id,
                    tags=["fact", "paper-pool"],
                    cloze_value=value,
                    stem=stem,
                )
            )
    return out


def _chunk_recall_cards(chunks: Sequence[AnyChunk], *, limit: int) -> list[_Candidate]:
    """General recall cards from salient sentences that are neither a
    definition nor a numeric claim — the pool's stand-in for Key Takeaways.

    Skips anything :func:`_chunk_fact_cards` or :func:`_chunk_definition_cards`
    would already have claimed (a number, or a definition pattern), so the
    three pool buckets do not fight over the same sentence.
    """
    out: list[_Candidate] = []
    seen: set[str] = set()
    for chunk in chunks:
        if len(out) >= limit:
            break
        text = _chunk_text(chunk)
        if not text.strip() or is_boilerplate(text):
            continue
        for sentence in _sentences(text):
            if len(out) >= limit:
                break
            clean = _strip_markers(sentence)
            if len(clean) < 60 or len(clean) > 280:
                continue
            if not _usable_sentence(clean):
                continue
            # Yield only to a cloze that `_chunk_fact_cards` will actually
            # build. Testing for *any* number instead meant a sentence whose
            # only digits were a citation index fell between the two buckets:
            # rejected here as "the fact path's job", then rejected there as an
            # ineligible blank target, so it produced no card at all.
            if _blank_number(clean) is not None:
                continue  # _chunk_fact_cards' job
            if any(pattern.match(clean) for pattern in _DEFINITION_PATTERNS):
                continue  # _chunk_definition_cards' job
            key = normalize_front(clean)
            if key in seen:
                continue
            seen.add(key)
            front, stem = _recall_front(clean, label="passage")
            out.append(
                _Candidate(
                    front=front,
                    back=clean,
                    origin=FlashcardOrigin.TAKEAWAY,
                    section=None,
                    hint=_pool_hint(_chunk_page(chunk)),
                    page=_chunk_page(chunk),
                    chunk_id=chunk.id,
                    tags=["takeaway", "paper-pool"],
                    stem=stem,
                )
            )
    return out


def _collect_pool(chunks: Sequence[AnyChunk], target: int) -> list[list[_Candidate]]:
    """One candidate bucket per pool-derived card type.

    Each bucket is capped generously (``target * 2``, bounded) rather than at
    exactly ``target`` — dedupe and interleaving both cost candidates, and the
    pool can be up to 400 chunks, so there is no shortage to protect against by
    capping tighter.
    """
    cap = max(target * 2, MIN_DECK_CARDS)
    return [
        _chunk_definition_cards(chunks, limit=cap),
        _chunk_recall_cards(chunks, limit=cap),
        _chunk_fact_cards(chunks, limit=cap),
    ]


def _widen_new_card_phrasing(
    bucket: list[_Candidate], paper_id: str, avoid_keys: frozenset[str]
) -> None:
    """Re-render fronts from the wider stem pool — but only for cards this deck
    has never produced. Mutates ``bucket`` in place.

    This gate is the whole design, and it exists because ``content_key`` is
    ``sha1(paper_id | origin | normalize_front(front))``: the card's *wording*
    is inside its identity. :func:`_stable_choice` picks a stem with
    ``hash % len(options)``, so merely appending a template changes the modulus
    and therefore re-keys **every card in every existing deck**. Each one would
    stop matching its stored row, be re-inserted as a "new" card, and leave its
    SM-2 schedule stranded on a near-duplicate the user still sees. Measured on
    this library that would have been ~130 cards across four decks.

    So: render the legacy front first (:func:`_stemmed`), and widen only when
    that front's key is *not* already in the deck. A card the deck knows keeps
    the exact wording it was stored under; a card the deck has never seen is
    free to use any stem, because it has no history to protect. Decks end up
    with mixed phrasing as they grow, which is the point.

    ``avoid_keys`` empty (a first generation) means every card is new, so a
    fresh deck draws on the full pool immediately.
    """
    if not bucket:
        return
    for candidate in bucket:
        if candidate.stem is None:
            continue
        if content_key_for(paper_id, candidate.origin, candidate.front) in avoid_keys:
            continue  # already in the deck under this exact wording — freeze it
        family, seed, params = candidate.stem
        candidate.front = _truncate(
            _render_stem(family, seed, params, widened=True), _MAX_FRONT_CHARS
        )


def _deprioritize(
    bucket: list[_Candidate], paper_id: str, avoid_keys: frozenset[str]
) -> list[_Candidate]:
    """Push candidates whose ``content_key`` is already in the deck to the
    back, order otherwise preserved.

    Deprioritized, never discarded: once unseen material runs out the seen
    cards are still there to fill the deck, so a regeneration can never come
    back empty just because everything looked familiar.
    """
    if not avoid_keys or not bucket:
        return bucket
    fresh: list[_Candidate] = []
    seen_before: list[_Candidate] = []
    for candidate in bucket:
        key = content_key_for(paper_id, candidate.origin, candidate.front)
        (seen_before if key in avoid_keys else fresh).append(candidate)
    return fresh + seen_before


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def _interleave(buckets: Sequence[list[_Candidate]], target: int) -> list[_Candidate]:
    """Round-robin across the origin buckets until ``target`` is reached.

    This is what enforces the mix. Taking the buckets in order would produce a
    deck of 20 glossary cards on any paper with a rich glossary — exactly the
    vocabulary list this feature is trying not to be.
    """
    picked: list[_Candidate] = []
    cursors = [0] * len(buckets)
    progressed = True
    while len(picked) < target and progressed:
        progressed = False
        for index, bucket in enumerate(buckets):
            if len(picked) >= target:
                break
            cursor = cursors[index]
            if cursor < len(bucket):
                picked.append(bucket[cursor])
                cursors[index] = cursor + 1
                progressed = True
    return picked


def _dedupe(candidates: Iterable[_Candidate]) -> list[_Candidate]:
    """Drop unusable cards and near-identical fronts, keeping first-seen order."""
    seen: set[str] = set()
    out: list[_Candidate] = []
    for candidate in candidates:
        if not candidate.is_useful():
            continue
        key = normalize_front(candidate.front)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _mentions_value(text: str, value: str) -> bool:
    """True if ``text`` still states ``value``, ignoring units and separators.

    Digits only, so the check survives the model rewording around the number
    ("41.8%" → "a BLEU score of 41.8") while still catching the case that
    matters: the number vanishing entirely. A spelled-out "six" does not match
    "6" and is treated as a loss — deliberately, because the safe direction
    here is keeping the deterministic draft, which always answers its own
    question.
    """
    wanted = re.sub(r"\D", "", value or "")
    if not wanted:
        return True
    # Normalise both sides the same way — compare number *tokens*, not a raw
    # substring. Stripping only separators from the text left "28.4" unfound
    # inside "score of 28.4" because the value had already lost its decimal
    # point; digit-crushing the whole text instead would let the "284" of one
    # number be satisfied by three unrelated digits sitting next to each other.
    return any(
        re.sub(r"\D", "", token) == wanted
        for token in re.findall(r"\d[\d.,]*", text or "")
    )


def _polish_backs(
    llm: Optional[LLMProvider], candidates: Sequence[_Candidate]
) -> None:
    """Rewrite the backs in place, or leave the drafts exactly as they are.

    The strict ``n :: text`` round-trip is the safety property: the deck is
    already correct before this runs, so anything the model returns that does
    not line up one-for-one with the request is discarded rather than
    reconciled. Never raises — ``complete_with_fallback`` swallows provider
    errors and returns the draft, and a mismatched parse is treated the same
    way.
    """
    if llm is None or not candidates:
        return
    draft = "\n".join(
        f"{index} :: {candidate.back}" for index, candidate in enumerate(candidates, 1)
    )
    try:
        polished = complete_with_fallback(
            llm,
            _POLISH_SYSTEM,
            draft,
            temperature=0.2,
            max_tokens=min(4000, 120 * len(candidates) + 200),
        )
    except Exception:  # noqa: BLE001 - defensive; the helper already swallows
        log.warning("flashcard polish call failed; keeping drafts", exc_info=True)
        return

    parsed: dict[int, str] = {}
    for line in (polished or "").splitlines():
        match = re.match(r"^\s*(\d{1,3})\s*::\s*(.+?)\s*$", line)
        if match:
            parsed[int(match.group(1))] = _strip_markers(match.group(2))

    expected = set(range(1, len(candidates) + 1))
    if set(parsed) != expected:
        log.warning(
            "flashcard polish did not round-trip; keeping deterministic backs",
            extra={"expected": len(candidates), "parsed": len(parsed)},
        )
        return

    for index, candidate in enumerate(candidates, 1):
        text = _truncate(parsed[index], _MAX_BACK_CHARS)
        # A polished back that collapsed into a stub or into the question is
        # worse than the draft it replaced.
        if len(text) < _MIN_BACK_CHARS:
            continue
        # A polished back that no longer contains the blanked value does not
        # answer its own question. This shipped: the front asked "What number
        # completes this claim? ... the previous hidden state ht−____" and the
        # model rewrote the back into a fluent paraphrase with no number in it
        # at all, leaving a card literally impossible to answer. Paraphrasing
        # is exactly what polish is for, so the fix is to keep the draft for
        # cloze cards specifically rather than to weaken the instruction.
        if candidate.cloze_value and not _mentions_value(text, candidate.cloze_value):
            log.debug(
                "polish dropped a cloze answer; keeping the draft back",
                extra={"value": candidate.cloze_value},
            )
            continue
        candidate.back = text


def build_deck(
    report: Optional[Report],
    *,
    paper_id: str,
    target_count: int = 20,
    llm: Optional[LLMProvider] = None,
    now: Optional[datetime] = None,
    paper_title: str = "",
    abstract: str = "",
    extra_chunks: Sequence[AnyChunk] = (),
    avoid_keys: frozenset[str] = frozenset(),
) -> list[Flashcard]:
    """Build a deck for ``paper_id`` from its persisted ``report`` plus, when
    given, a whole-paper ``extra_chunks`` pool (see
    :func:`deepvision.study.source_pool.pool_for_paper`).

    The report is mined first and stays preferred — it is the citation-bearing,
    quality-filtered part — but ``extra_chunks`` candidates are interleaved
    alongside it rather than appended after, so the deck is a genuine mix.
    ``avoid_keys`` (typically the content keys a persisted deck already has)
    deprioritizes repeats rather than dropping them, so a regeneration surfaces
    new material first without ever coming back with nothing.

    Never raises: any failure inside extraction or polish degrades to a smaller
    deck (in the limit, an empty list), and the caller decides what to do about
    that. The returned cards carry a default :class:`CardStrength`; the
    persistence layer keeps the existing state for any card whose
    ``content_key`` it already has.
    """
    from deepvision.study.card_queries import utcnow  # local: avoids a cycle

    moment = now or utcnow()
    target = max(MIN_DECK_CARDS, min(MAX_DECK_CARDS, int(target_count or 20)))

    report_buckets: list[list[_Candidate]] = []
    try:
        report_buckets = _collect(report, target)
    except Exception:  # noqa: BLE001 - extraction must never break generation
        log.warning(
            "flashcard extraction failed; falling back to the abstract",
            extra={"paper_id": paper_id},
            exc_info=True,
        )
        report_buckets = []

    pool_buckets: list[list[_Candidate]] = []
    if extra_chunks:
        try:
            pool_buckets = _collect_pool(extra_chunks, target)
        except Exception:  # noqa: BLE001 - the pool is a bonus, never a blocker
            log.warning(
                "flashcard pool extraction failed; using the report alone",
                extra={"paper_id": paper_id},
                exc_info=True,
            )
            pool_buckets = []

    buckets: list[list[_Candidate]] = report_buckets + pool_buckets
    if not any(buckets):
        buckets = [_abstract_cards(paper_title, abstract)]

    # Widen phrasing BEFORE deprioritizing, so `_deprioritize` sees the fronts
    # the cards will actually ship with. Runs unconditionally: with no
    # `avoid_keys` every card is new, which is exactly when the full stem pool
    # should be used.
    for bucket in buckets:
        _widen_new_card_phrasing(bucket, paper_id, avoid_keys)

    if avoid_keys:
        buckets = [_deprioritize(bucket, paper_id, avoid_keys) for bucket in buckets]

    candidates = _dedupe(_interleave(buckets, target))
    if not candidates:
        log.warning("no flashcards could be extracted", extra={"paper_id": paper_id})
        return []

    _polish_backs(llm, candidates)
    candidates = _dedupe(candidates)  # polish can collapse two backs into one

    cards: list[Flashcard] = []
    for candidate in candidates:
        cards.append(
            Flashcard(
                id=new_id("card"),
                paper_id=paper_id,
                front=candidate.front,
                back=candidate.back,
                hint=candidate.hint,
                origin=candidate.origin,
                content_key=content_key_for(paper_id, candidate.origin, candidate.front),
                source_section=candidate.section,
                source_page=candidate.page,
                chunk_id=candidate.chunk_id,
                tags=candidate.tags,
                starred=False,
                created_at=moment,
                updated_at=moment,
            )
        )
    return cards


def _collect(report: Optional[Report], target: int) -> list[list[_Candidate]]:
    """One candidate bucket per origin, in interleaving order."""
    if report is None:
        return []
    glossary: list[_Candidate] = []
    concepts: list[_Candidate] = []
    takeaways: list[_Candidate] = []
    facts: list[_Candidate] = []

    for name, origin in _SOURCES:
        section = report.section(name)
        if section is None or not (section.body_markdown or "").strip():
            continue
        # No entry in _SOURCES yields FlashcardOrigin.GLOSSARY any more — the
        # report's Glossary section is gone (see the module docstring) — so
        # this branch is unreachable from this loop today. It is kept working
        # rather than deleted because FlashcardOrigin.GLOSSARY itself is not
        # going away: cards persisted before this change still carry it on
        # the wire, and this is the one place that knows how to build a
        # glossary-style (bare term front) card, should that ever be wired
        # back in from elsewhere.
        if origin is FlashcardOrigin.GLOSSARY:
            glossary.extend(_term_cards(section, origin, as_question=False))
        elif origin is FlashcardOrigin.KEY_CONCEPT:
            concepts.extend(_term_cards(section, origin, as_question=True))
            # A Key Concepts section written as prose rather than term pairs
            # still has facts worth drilling.
            if not concepts:
                facts.extend(_fact_cards(section, limit=target))
        elif origin is FlashcardOrigin.TAKEAWAY:
            takeaways.extend(_takeaway_cards(section))
        else:
            facts.extend(_fact_cards(section, limit=target))

    return [glossary, concepts, takeaways, facts]


class FlashcardAgent(Agent[list[Flashcard]]):
    """Agent wrapper around :func:`build_deck`.

    Reads ``context.extra['report']`` (a :class:`~deepvision.models.Report`),
    ``context.extra['target_count']``, and optionally ``context.extra
    ['extra_chunks']`` (a whole-paper pool) and ``context.extra['avoid_keys']``
    (content keys already in the deck). Exists so deck generation sits in the
    agent layer with every other LLM-touching unit and shares the
    ``complete_with_fallback`` discipline; the callable form is what the deck
    generator actually uses.
    """

    def run(self, context: AgentContext) -> list[Flashcard]:
        report = context.extra.get("report")
        return build_deck(
            report if isinstance(report, Report) else None,
            paper_id=context.paper_id,
            target_count=int(context.extra.get("target_count") or 20),
            llm=self.llm,
            paper_title=str(context.extra.get("paper_title") or ""),
            abstract=str(context.extra.get("abstract") or ""),
            extra_chunks=context.extra.get("extra_chunks") or (),
            avoid_keys=context.extra.get("avoid_keys") or frozenset(),
        )
