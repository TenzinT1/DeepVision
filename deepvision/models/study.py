"""Study models — flashcards (session-based scheduling) and quizzes.

This module is the single source of truth for the **Study** layer: the app's
third top-level screen, sitting between Report and Chat.

**Why Study is top-level and not a report section.** Three reasons, all
structural:

1. Report sections are regenerated *wholesale* — ``report_generator`` overwrites
   ``ReportRow.sections`` — so anything living in a section is destroyed on the
   next generate. Spaced-repetition progress must survive that, so it lives in
   its own tables keyed by paper.
2. The whole point of spaced repetition is the **cross-paper due queue**: "what
   do I owe today", across the entire library. A per-report widget cannot answer
   that question.
3. Reports export to Markdown and PDF, where an interactive widget cannot
   render at all.

The Report page therefore gets small **launch cards** only — live counts
("Flashcards · 24 cards · 6 due") that deep-link into Study filtered to that
paper. Never the widget itself. Those counts come from
:class:`PaperStudyStats`.

Contents:

- :class:`Flashcard` + :class:`FlashcardReview` — session-based scheduling and
  its append-only audit log. There are no dates and no intervals: a card is
  placed within one sitting, and the only thing that survives the sitting is
  the log itself, from which :class:`CardStrength` is *derived* on read. The
  exact rules are specified in :func:`describe_scheduler_contract`;
  ``deepvision/study/session_scheduler.py`` implements exactly that.
- :class:`Quiz` / :class:`QuizQuestion` / :class:`QuizAttempt` — mixed-kind,
  difficulty-tiered questions with per-question explanations and citations, and
  server-side grading.

**Security-shaped contract:** :class:`QuizQuestion` holds the answer and the
explanation and is *never* serialized to the client before the user answers.
The wire shape sent to the browser is :class:`QuizQuestionPublic`, which has no
answer field to leak. See :class:`QuizPublic`.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from deepvision.models.report import SectionName

__all__ = [
    # flashcards / session scheduling
    "Rating",
    "LEGACY_RECALLED_RATINGS",
    "rating_recalled",
    "FlashcardOrigin",
    "SESSION_REQUEUE_GAPS",
    "CardStrength",
    "Flashcard",
    "FlashcardReview",
    "DueCard",
    "DeckProgress",
    "describe_scheduler_contract",
    # quizzes
    "QuestionKind",
    "Difficulty",
    "AttemptMode",
    "QuizCitation",
    "QuizOption",
    "QuizQuestion",
    "QuizQuestionPublic",
    "Quiz",
    "QuizPublic",
    "QuizSummary",
    "SubmittedAnswer",
    "GradedAnswer",
    "QuizAttempt",
    "QuizAttemptSummary",
    "SHORT_ANSWER_TOKEN_OVERLAP_THRESHOLD",
    # cross-paper study surface
    "StudyItemKind",
    "PaperStudyStats",
]


# ==========================================================================
# Flashcards — enums
# ==========================================================================
class Rating(str, Enum):
    """The three review buttons. Values are the wire contract — never rename.

    ============ =====================================================
    Rating       Meaning
    ============ =====================================================
    ``again``    Failed to recall. Comes back soon, and resets strength.
    ``almost``   Recalled with effort. Comes back later in the session.
    ``got_it``   Clean recall. Done for this session.
    ============ =====================================================

    Persisted rows may still carry the retired SM-2 values
    ``hard``/``good``/``easy``; read them through :func:`rating_recalled`,
    never by comparing against this enum's members.
    """

    AGAIN = "again"
    ALMOST = "almost"
    GOT_IT = "got_it"

    @property
    def is_recalled(self) -> bool:
        """True if the answer counts as recalled — everything but ``again``.

        The one place that distinction is defined; strength derivation reads
        it, and legacy rows go through :func:`rating_recalled` instead.
        """
        return self is not Rating.AGAIN


#: Ratings written by the retired SM-2 scheduler. Persisted rows still carry
#: them and :data:`flashcard_reviews` is append-only, so history must stay
#: readable: ``hard``/``good``/``easy`` all meant "recalled", exactly like
#: :attr:`Rating.ALMOST` and :attr:`Rating.GOT_IT` do now. ``again`` kept its
#: meaning, so it needs no entry here.
LEGACY_RECALLED_RATINGS: frozenset[str] = frozenset({"hard", "good", "easy"})


def rating_recalled(value: str) -> bool:
    """True if a persisted rating string counts as recall, old model or new."""
    return value in LEGACY_RECALLED_RATINGS or value in {
        Rating.ALMOST.value,
        Rating.GOT_IT.value,
    }


class FlashcardOrigin(str, Enum):
    """What the generator built a card from. Wire contract — never rename.

    Drives the little origin pill on the card back, and lets the generator
    guarantee a *mix* rather than 30 vocabulary cards:

    - ``glossary`` — a term/definition pair. Front = term, back = definition.
      **No report section produces this origin any more**: the report's
      ``Glossary`` section was merged into ``Key Concepts`` (it emitted
      near-identical entries), so new decks use ``key_concept`` instead. The
      value stays because it is a wire contract and persisted cards carry it.
    - ``key_concept`` — a ``Key Concepts`` entry, rephrased as a question. This
      is where vocabulary cards now come from.
    - ``takeaway`` — a ``Key Takeaways`` claim, turned into a cloze-ish recall
      prompt.
    - ``fact`` — a paper-specific fact (a number, dataset, baseline, ablation
      result) pulled from ``Key Results`` / ``Methods``. These are the cards that
      make the deck about *this paper* rather than about the field.
    - ``manual`` — created by the user, never overwritten by regeneration.
    """

    GLOSSARY = "glossary"
    KEY_CONCEPT = "key_concept"
    TAKEAWAY = "takeaway"
    FACT = "fact"
    MANUAL = "manual"


# ==========================================================================
# Session scheduling — THE tunable numbers
#
# There is exactly one knob pair in this scheduler and it lives here. Tune
# these two integers and nothing else: the review endpoint returns the gap to
# the client, so the frontend holds no copy of them to drift out of sync.
# ==========================================================================

#: How many *other* cards to place between a card and its return, per rating.
#: "Again" brings it back soon, "Almost" brings it back later, and ``got_it``
#: has no entry because it does not come back this session at all.
#:
#: These are starting guesses, not measured values. Real decks here run 15-54
#: cards, so 6 is roughly a third of the smallest deck and 12 is most of it.
#: If a card is rated with fewer than its gap of cards left in the session, it
#: goes to the END of the session rather than being dropped.
SESSION_REQUEUE_GAPS: dict[Rating, int] = {
    Rating.AGAIN: 6,
    Rating.ALMOST: 12,
}


def describe_scheduler_contract() -> str:
    """Return the session-scheduling contract as prose (the implementation spec).

    This exists so the rule set has exactly one home, the way the SM-2 contract
    it replaces did. ``deepvision/study/session_scheduler.py`` implements this
    verbatim and quotes it.
    """
    return _SCHEDULER_CONTRACT


_SCHEDULER_CONTRACT = """\
Session scheduling contract (deepvision) — apply exactly.

There are NO dates and NO intervals. A card is scheduled within one sitting,
and nothing about that sitting is persisted.

Ratings: again | almost | got_it

--------------------------------------------------------------------------
A. Placement inside the session
--------------------------------------------------------------------------
The server returns the gap; the client owns the queue and does the insert.

  again  -> the card returns after SESSION_REQUEUE_GAPS[again] other cards
  almost -> the card returns after SESSION_REQUEUE_GAPS[almost] other cards
  got_it -> the card does not return this session (gap is null)

If fewer cards remain than the gap, the card goes to the END of the session.
It is never dropped: a card you failed must be seen again before you stop.

--------------------------------------------------------------------------
B. What survives a session — nothing is stored
--------------------------------------------------------------------------
No table records scheduling state. Two quantities are DERIVED on read from
the append-only `flashcard_reviews` log, which already carries card_id,
rating and reviewed_at for every answer ever given:

  strength(card) = how many consecutive non-`again` ratings the card has,
                   counting back from its most recent review. A card never
                   reviewed, or whose latest rating was `again`, has 0.
                   Legacy `hard`/`good`/`easy` rows count as recalled, so
                   history written by the retired SM-2 scheduler still counts.

  due            = total cards - distinct cards reviewed during the LOCAL
                   calendar day. Distinct, so a card seen three times after
                   `again` still only counts once. Resets at local midnight.

`reviewed_at` is naive UTC (the whole server uses `utcnow()`), so the local
day must be converted to UTC bounds before comparing — a UTC day boundary
would roll the count over mid-afternoon for anyone west of Greenwich.

--------------------------------------------------------------------------
C. Queue order
--------------------------------------------------------------------------
Ascending strength, then least-recently-reviewed first, then a stable tie
break on card id. Cards you are worst at come first; cards you keep getting
right sink toward the back. This is the only thing that carries a rating's
meaning into a later session.

--------------------------------------------------------------------------
D. Logging
--------------------------------------------------------------------------
Every rating appends exactly one FlashcardReview row, never updates and never
deletes one. The SM-2 columns on `flashcards` (state, ease_factor,
interval_days, repetitions, lapses, learning_step, due_at, last_reviewed_at)
and the prior_*/post_*/was_lapse/due_at_after columns on `flashcard_reviews`
are NOT read and NOT meaningfully written after this change. They are left in
place only because SQLModel `create_all` cannot migrate a table; rows keep
whatever defaults the schema gives them.
"""


# ==========================================================================
# Flashcards — models
# ==========================================================================
class CardStrength(BaseModel):
    """What the review log says about one card, derived — never stored.

    Replaces the retired ``SrsState``. Nothing here is a column: every field is
    computed from :data:`flashcard_reviews` on read, which is what lets the
    scheduler carry a rating's meaning into a later session without persisting
    a single byte of schedule. See :func:`describe_scheduler_contract` §B.
    """

    model_config = ConfigDict(use_enum_values=False)

    strength: int = Field(
        default=0,
        ge=0,
        description="Consecutive non-'again' ratings, counting back from the newest.",
    )
    reviews: int = Field(default=0, ge=0, description="Lifetime reviews of this card.")
    last_reviewed_at: Optional[datetime] = Field(
        default=None, description="Newest review timestamp (naive UTC), or null."
    )
    seen_today: bool = Field(
        default=False,
        description="Reviewed at least once during the viewer's LOCAL day.",
    )


class Flashcard(BaseModel):
    """One card in a paper's deck: content + SM-2 scheduling state.

    Cards are keyed by ``paper_id`` (the deck *is* the paper), but the due queue
    that drives the Study screen is deliberately **cross-paper** — see
    :class:`DueCard`.

    Regeneration semantics: re-generating a deck must *upsert* by
    ``content_key`` and preserve the SRS state of cards that survive, so a user
    who has been reviewing for three weeks does not lose their schedule because
    the report was re-run. Cards with ``origin == 'manual'`` are never touched by
    regeneration.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(..., description="Card id ('card_...'; utils.ids.new_id('card')).")
    paper_id: str = Field(..., description="Owning paper — the deck this card belongs to.")

    front: str = Field(
        ...,
        description="Prompt side (plain text or **bold**/*italic* markdown).",
        examples=["What does 'emergent ability' mean in this paper?"],
    )
    back: str = Field(
        ..., description="Answer side. Keep to 1-3 sentences; long answers do not stick."
    )
    hint: Optional[str] = Field(
        default=None, description="Optional nudge revealed before the answer."
    )
    origin: FlashcardOrigin = Field(
        default=FlashcardOrigin.FACT,
        description="What the card was generated from (glossary/key_concept/takeaway/fact/manual).",
    )
    content_key: str = Field(
        default="",
        description=(
            "Stable dedupe key for upsert-on-regenerate: "
            "sha1(paper_id|origin|normalized front)[:16]. Two generations that "
            "produce the same question must map to the same row."
        ),
    )

    source_section: Optional[SectionName] = Field(
        default=None, description="Report section the card was derived from."
    )
    source_page: Optional[int] = Field(
        default=None, ge=1, description="1-based page in the PDF, when known."
    )
    chunk_id: Optional[str] = Field(
        default=None, description="Backing chunk id, for the citation popover."
    )
    tags: list[str] = Field(
        default_factory=list, description="Free-form tags, e.g. ['metrics', 'ablation']."
    )

    starred: bool = Field(
        default=False, description="User-flagged 'come back to this' marker."
    )
    strength: CardStrength = Field(
        default_factory=CardStrength,
        description="Derived from the review log; never persisted. See the contract.",
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class FlashcardReview(BaseModel):
    """One row of the **append-only** review log.

    Never updated, never deleted. Two reasons this is a table and not just a
    counter on the card: the schedule is otherwise unauditable (you cannot tell
    *why* a card is due in 33 days), and a future "Study stats" view — reviews
    per day, lapse-prone cards — needs the history, which cannot
    be reconstructed from current state.

    Every field pair ``prior_* / post_*`` brackets exactly one application of the
    SM-2 contract, so replaying the log reproduces the card's current state.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(..., description="Review id ('rev_...'; utils.ids.new_id('rev')).")
    card_id: str = Field(..., description="Card that was reviewed.")
    paper_id: str = Field(..., description="Denormalized for cheap per-paper stats.")
    rating: str = Field(
        ...,
        description=(
            "'again' | 'almost' | 'got_it'. A plain str, not the Rating enum: "
            "rows written by the retired SM-2 scheduler carry 'hard'/'good'/"
            "'easy', and this log is append-only so they must keep validating. "
            "Read them through rating_recalled()."
        ),
    )

    was_lapse: bool = Field(
        default=False,
        description="True iff this review took a 'review' card to 'again' (lapses += 1).",
    )
    elapsed_ms: Optional[int] = Field(
        default=None, ge=0, description="Client-measured think time; advisory only."
    )
    reviewed_at: datetime = Field(
        default_factory=datetime.utcnow, description="UTC time the rating was applied."
    )
    due_at_after: datetime = Field(
        ..., description="The due_at this review scheduled (post-update)."
    )


class DueCard(BaseModel):
    """A card in the **cross-paper** due queue, with just enough paper context.

    The Study screen's default view is "everything due today, across the whole
    library" — this is the reason Study is top-level at all. Each entry carries
    the paper title/label so the queue can show provenance without an N+1 fetch.
    """

    model_config = ConfigDict(use_enum_values=False)

    card: Flashcard = Field(..., description="The card to review.")
    paper_title: str = Field(default="", description="Denormalized PaperMeta.title.")
    paper_label: str = Field(
        default="", description="Denormalized PaperMeta.arxiv_label, e.g. 'arXiv:2411.10842'."
    )


class DeckProgress(BaseModel):
    """Progress counters for one deck (or, aggregated, for the whole library)."""

    total: int = Field(default=0, ge=0, description="Cards in the deck.")
    due_count: int = Field(
        default=0,
        ge=0,
        description="Still to do today: total minus distinct cards seen this local day.",
    )
    starred_count: int = Field(default=0, ge=0, description="Cards with starred == true.")
    reviews_today: int = Field(
        default=0, ge=0, description="FlashcardReview rows since local midnight UTC."
    )


# ==========================================================================
# Quizzes — enums
# ==========================================================================
class QuestionKind(str, Enum):
    """Question type. Wire contract — the UI renders a different input per kind.

    - ``multiple_choice`` — exactly 4 :class:`QuizOption` s, exactly one correct.
    - ``true_false`` — exactly 2 options, with the fixed ids ``"true"`` / ``"false"``
      and the labels ``"True"`` / ``"False"``. Generators must emit them in that
      order; the UI renders two buttons, not a radio list.
    - ``short_answer`` — free text, graded server-side by normalized comparison
      against ``accepted_answers`` (see :data:`SHORT_ANSWER_TOKEN_OVERLAP_THRESHOLD`).
    """

    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"


class Difficulty(str, Enum):
    """Bloom-ish difficulty tier. Wire contract — never rename.

    - ``recall`` — "what is X" / "which dataset did they use". Answerable from
      a single Key Concepts definition or a single number in Key Results.
    - ``understand`` — "why did the baseline fail", "what does this trend mean".
      Requires connecting two facts from the paper.
    - ``apply`` — "you have a 3B model and no labelled data — which of the
      paper's techniques applies, and why". Transfer to a new situation.

    A generated quiz must span all three (see :attr:`Quiz.difficulty_mix`); an
    all-``recall`` quiz is trivia, not study.
    """

    RECALL = "recall"
    UNDERSTAND = "understand"
    APPLY = "apply"


class AttemptMode(str, Enum):
    """How an attempt was started. Wire contract — never rename.

    - ``full`` — every question in the quiz.
    - ``retry_missed`` — the "retry only what I missed" mode: an attempt scoped
      to a **subset of question ids**, namely those answered incorrectly in
      ``QuizAttempt.retry_of_attempt_id``. Scored over that subset only, so its
      ``score`` is not comparable to a ``full`` attempt's — the UI must label it.
    """

    FULL = "full"
    RETRY_MISSED = "retry_missed"


#: Short-answer fallback grading: after normalization (casefold, strip, collapse
#: internal whitespace, drop leading/trailing punctuation), an answer is correct
#: if it equals any entry in ``accepted_answers`` OR if the share of the accepted
#: answer's content tokens (tokens of length >= 3, minus a small stopword set)
#: present in the submission is at least this. Deterministic and cheap: grading
#: is synchronous and must never call a model.
SHORT_ANSWER_TOKEN_OVERLAP_THRESHOLD: float = 0.6


# ==========================================================================
# Quizzes — models
# ==========================================================================
class QuizCitation(BaseModel):
    """Where a question's answer comes from — shown with the explanation.

    Every question carries one. A quiz question whose answer cannot be pointed
    at in the source is a hallucination risk, so the generator must drop any
    question it cannot cite rather than emit ``section: null``.
    """

    model_config = ConfigDict(use_enum_values=False)

    section: SectionName = Field(
        ..., description="Report section that supports the answer."
    )
    page: Optional[int] = Field(
        default=None, ge=1, description="1-based PDF page, when known."
    )
    chunk_id: Optional[str] = Field(
        default=None, description="Backing chunk id — lets the UI open the citation popover."
    )
    snippet: str = Field(
        default="", description="Short supporting quote from the source (<= 240 chars)."
    )


class QuizOption(BaseModel):
    """One selectable answer option.

    Deliberately has **no** ``is_correct`` field: this model is what goes over
    the wire before the user answers, and a correctness flag here would be
    readable in devtools. Correctness lives in
    :attr:`QuizQuestion.correct_option_id`, server-side only.
    """

    id: str = Field(
        ...,
        description="Stable option id within the question ('a'|'b'|'c'|'d', or 'true'|'false').",
        examples=["b"],
    )
    text: str = Field(..., description="Option label shown to the user.")


class QuizQuestion(BaseModel):
    """A question **with its answer** — the persisted, server-side shape.

    NEVER return this to the client before the user has answered. Serialize
    :class:`QuizQuestionPublic` instead; the answer and explanation only travel
    back inside a :class:`GradedAnswer` after grading.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(..., description="Question id ('qq_...'; utils.ids.new_id('qq')).")
    quiz_id: str = Field(..., description="Owning quiz.")
    paper_id: str = Field(..., description="Denormalized owning paper.")
    index: int = Field(..., ge=0, description="0-based position in the quiz.")

    prompt: str = Field(..., description="The question text.")
    kind: QuestionKind = Field(..., description="multiple_choice | true_false | short_answer.")
    difficulty: Difficulty = Field(..., description="recall | understand | apply.")

    options: list[QuizOption] = Field(
        default_factory=list,
        description=(
            "4 options for multiple_choice; the fixed True/False pair for "
            "true_false; empty for short_answer."
        ),
    )
    correct_option_id: Optional[str] = Field(
        default=None,
        description="SERVER-ONLY. The winning QuizOption.id (mcq/true_false).",
    )
    correct_answer_text: Optional[str] = Field(
        default=None, description="SERVER-ONLY. Canonical short_answer answer."
    )
    accepted_answers: list[str] = Field(
        default_factory=list,
        description=(
            "SERVER-ONLY. Alternative acceptable short answers (synonyms, "
            "abbreviations). Always includes correct_answer_text."
        ),
    )
    explanation: str = Field(
        default="",
        description=(
            "SERVER-ONLY until graded. Explains WHY the right answer is right "
            "and, where useful, why the tempting wrong option is wrong. This is "
            "the whole pedagogical payload — never ship an empty one."
        ),
    )
    citation: Optional[QuizCitation] = Field(
        default=None, description="Where in the paper the answer is grounded."
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QuizQuestionPublic(BaseModel):
    """The **hidden-until-answered** wire shape: a question minus its answer.

    This type structurally cannot leak the answer — ``correct_option_id``,
    ``correct_answer_text``, ``accepted_answers`` and ``explanation`` are absent
    from the schema, not merely nulled. That is the point: nulling a field still
    invites a future edit that populates it.

    The citation *is* included so the UI can offer "show me where this comes
    from" as a hint — but as a **pointer only**: section, page and ``chunk_id``,
    with ``snippet`` blanked. The snippet is a verbatim quote of the source the
    question was generated from, and for a ``short_answer`` question mined from a
    Key Concepts definition the accepted answer usually sits inside that quote word
    for word. "It points at a section, not at the answer" only holds once the
    quote is removed. The full snippet travels later, in
    :class:`GradedAnswer.citation`, where the answer is revealed anyway.
    ``routers/quiz.py::_pointer_citation`` is what enforces this.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str
    index: int = Field(..., ge=0)
    prompt: str
    kind: QuestionKind
    difficulty: Difficulty
    options: list[QuizOption] = Field(
        default_factory=list, description="Empty for short_answer."
    )
    citation: Optional[QuizCitation] = Field(default=None)


class Quiz(BaseModel):
    """A generated quiz for one paper — server-side shape, answers included."""

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(..., description="Quiz id ('quiz_...'; utils.ids.new_id('quiz')).")
    paper_id: str = Field(..., description="Owning paper.")
    title: str = Field(default="", description="e.g. 'Quiz · 10 questions'.")
    questions: list[QuizQuestion] = Field(default_factory=list)
    difficulty_mix: dict[str, int] = Field(
        default_factory=dict,
        description="Counts per Difficulty value, e.g. {'recall': 4, 'understand': 4, 'apply': 2}.",
    )
    kind_mix: dict[str, int] = Field(
        default_factory=dict,
        description="Counts per QuestionKind value, e.g. {'multiple_choice': 6, ...}.",
    )
    model_used: Optional[str] = Field(default=None)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class QuizPublic(BaseModel):
    """What ``GET /api/quiz/{quiz_id}`` returns — the quiz with answers stripped.

    ``question_ids`` on a ``retry_missed`` fetch is the subset being retried, in
    the original quiz order.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str
    paper_id: str
    paper_title: str = Field(default="", description="Denormalized, for the header.")
    title: str = Field(default="")
    questions: list[QuizQuestionPublic] = Field(default_factory=list)
    difficulty_mix: dict[str, int] = Field(default_factory=dict)
    kind_mix: dict[str, int] = Field(default_factory=dict)
    generated_at: datetime
    mode: AttemptMode = Field(
        default=AttemptMode.FULL,
        description="'retry_missed' when this is a scoped retry payload.",
    )
    retry_of_attempt_id: Optional[str] = Field(
        default=None, description="Set iff mode == 'retry_missed'."
    )


class QuizSummary(BaseModel):
    """One row in a paper's quiz list — no questions, so it is cheap."""

    model_config = ConfigDict(use_enum_values=False)

    id: str
    paper_id: str
    title: str = Field(default="")
    question_count: int = Field(default=0, ge=0)
    difficulty_mix: dict[str, int] = Field(default_factory=dict)
    generated_at: datetime
    attempt_count: int = Field(default=0, ge=0)
    best_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Best attempt score in [0,1], null if never attempted."
    )
    last_attempted_at: Optional[datetime] = Field(default=None)


class SubmittedAnswer(BaseModel):
    """One answer the client submits. Never carries a correctness claim.

    Exactly one of ``selected_option_id`` / ``answer_text`` is meaningful:
    the former for ``multiple_choice`` and ``true_false``, the latter for
    ``short_answer``. Both null = unanswered, which grades as incorrect (it is
    not an error — the user can submit a partially finished quiz).
    """

    question_id: str = Field(..., description="Question being answered.")
    selected_option_id: Optional[str] = Field(
        default=None, description="QuizOption.id, for multiple_choice / true_false."
    )
    answer_text: Optional[str] = Field(
        default=None, description="Free text, for short_answer."
    )
    elapsed_ms: Optional[int] = Field(
        default=None, ge=0, description="Client-measured think time; advisory only."
    )


class GradedAnswer(BaseModel):
    """One graded answer — the **only** shape that reveals the correct answer.

    Returned exclusively in a :class:`QuizAttempt` response, i.e. after the user
    has committed. This is where the explanation finally travels to the client.
    """

    model_config = ConfigDict(use_enum_values=False)

    question_id: str
    index: int = Field(..., ge=0, description="Position in the quiz, for ordering.")
    prompt: str = Field(default="", description="Echoed so the review screen needs no join.")
    kind: QuestionKind
    difficulty: Difficulty
    options: list[QuizOption] = Field(default_factory=list)

    selected_option_id: Optional[str] = Field(default=None)
    answer_text: Optional[str] = Field(default=None)
    is_correct: bool = Field(default=False, description="Server's verdict. Authoritative.")

    correct_option_id: Optional[str] = Field(
        default=None, description="Revealed now that the answer is committed."
    )
    correct_answer_text: Optional[str] = Field(default=None, description="Revealed now.")
    explanation: str = Field(
        default="", description="Why the right answer is right (and the wrong one wrong)."
    )
    citation: Optional[QuizCitation] = Field(default=None)


class QuizAttempt(BaseModel):
    """One graded run through a quiz (or a missed-questions subset of it).

    ``score`` is ``correct / total`` over ``question_ids`` — for a
    ``retry_missed`` attempt that denominator is the subset size, so scores from
    the two modes are not comparable and the UI must not chart them on one line.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(..., description="Attempt id ('qa_...'; utils.ids.new_id('qa')).")
    quiz_id: str
    paper_id: str
    mode: AttemptMode = Field(default=AttemptMode.FULL)
    retry_of_attempt_id: Optional[str] = Field(
        default=None, description="The attempt whose misses this one retries."
    )
    question_ids: list[str] = Field(
        default_factory=list, description="Questions in scope, in quiz order."
    )
    answers: list[GradedAnswer] = Field(default_factory=list)

    total: int = Field(default=0, ge=0, description="len(question_ids).")
    correct: int = Field(default=0, ge=0)
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="correct / total; 0 when total == 0.")

    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime = Field(default_factory=datetime.utcnow)

    missed_question_ids: list[str] = Field(
        default_factory=list,
        description="Questions answered incorrectly — the exact input to a 'retry missed' run.",
    )


class QuizAttemptSummary(BaseModel):
    """One row of attempt history — no per-answer detail, so the list is cheap."""

    model_config = ConfigDict(use_enum_values=False)

    id: str
    quiz_id: str
    paper_id: str
    mode: AttemptMode = Field(default=AttemptMode.FULL)
    total: int = Field(default=0, ge=0)
    correct: int = Field(default=0, ge=0)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    missed_count: int = Field(default=0, ge=0)
    started_at: datetime
    finished_at: datetime


# ==========================================================================
# Cross-paper study surface
# ==========================================================================
class StudyItemKind(str, Enum):
    """Which study tool a launch card / filter refers to. Wire contract."""

    FLASHCARDS = "flashcards"
    QUIZ = "quiz"


class PaperStudyStats(BaseModel):
    """Live per-paper study state — **exactly what the Report launch cards need**.

    The Report page renders two small cards from this and nothing else:

    - "Flashcards · {cards_total} cards"  → deep-links to
      ``/study?paper={paper_id}&tab=flashcards``
    - "Quiz · {quiz_count} quizzes · best {best_score}"     → deep-links to
      ``/study?paper={paper_id}&tab=quiz``

    Cards, never the widget: the widget lives on the Study screen. ``has_deck``
    / ``quiz_count == 0`` is the "Generate" state, not an error.
    """

    paper_id: str
    paper_title: str = Field(default="")
    paper_label: str = Field(default="", description="e.g. 'arXiv:2411.10842'.")

    has_deck: bool = Field(default=False, description="A deck has been generated.")
    cards_total: int = Field(default=0, ge=0)
    cards_starred: int = Field(default=0, ge=0)
    deck_generated_at: Optional[datetime] = Field(default=None)

    quiz_count: int = Field(default=0, ge=0)
    quiz_attempt_count: int = Field(default=0, ge=0)
    quiz_best_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Best score across this paper's FULL attempts; falls back to any "
            "attempt only when there has never been a full one. A "
            "'retry_missed' run is scored over a smaller, self-selected "
            "denominator, so counting it here would let '6/6 on the six I "
            "missed' outrank the 4/10 that produced it — and this number is "
            "what the Report launch card renders. Same rule as "
            "QuizSummary.best_score and QuizAttemptHistoryResponse.best_score."
        ),
    )
    quiz_last_attempted_at: Optional[datetime] = Field(default=None)
