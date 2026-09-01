"""Deterministic, offline quiz grading.

Everything in this module is a **pure function**: no database, no network, and —
emphatically — **no model call**. Grading has to be fast (it runs inside the
``POST /api/quiz/{quiz_id}/attempt`` request), reproducible (the same submission
must always score the same, or attempt history is meaningless), and available
when no LLM is reachable at all. An LLM grader would fail all three.

The contract implemented here is the one specified in
``deepvision/api/routers/quiz.py`` and ``models/study.py``:

- ``multiple_choice`` / ``true_false`` — exact option-id equality. The client
  never learns the correct id before it commits, so there is nothing to be
  clever about.
- ``short_answer`` — tolerant, but *explicitly* tolerant:

  1. Normalize both sides (NFKC, casefold, ASCII-fold the typographic quotes and
     dashes, collapse internal whitespace, drop leading/trailing punctuation).
  2. Correct if the normalized submission equals **any** entry in
     ``accepted_answers`` — compared punctuation-insensitively, so
     ``self-attention`` and ``self attention`` are the same answer.
  3. Otherwise correct if the share of an accepted answer's *content tokens*
     present in the submission is at least
     :data:`~deepvision.models.study.SHORT_ANSWER_TOKEN_OVERLAP_THRESHOLD`
     (0.6). Content tokens are tokens of length >= 3 that are not stopwords,
     plus every numeric token regardless of length (``8`` layers,
     ``2.5`` BLEU — numbers are the most informative token in a research
     paper and must not be filtered out by a length rule).

Where it is in doubt, it says **wrong** — and the caller always returns the
expected answer inside :class:`~deepvision.models.study.GradedAnswer` so the
user can self-assess a near miss rather than argue with a regex. That is the
deliberate bias: an over-generous grader silently teaches the wrong thing.

An unanswered question is graded incorrect and is **not** an error: submitting a
partially finished quiz is a supported flow.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from deepvision.models.study import (
    SHORT_ANSWER_TOKEN_OVERLAP_THRESHOLD,
    GradedAnswer,
    QuestionKind,
    QuizQuestion,
    SubmittedAnswer,
)

__all__ = [
    "STOPWORDS",
    "ShortAnswerVerdict",
    "normalize_text",
    "punctuation_insensitive",
    "content_tokens",
    "token_overlap",
    "is_non_answer",
    "grade_choice",
    "grade_short_answer",
    "expected_answer_label",
    "feedback_explanation",
    "grade_question",
    "grade_answers",
]


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
#: Leading/trailing runs of anything that is not a letter, digit or underscore.
_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")
#: Typographic characters a user (or a model) may paste, folded to ASCII so
#: "don’t" and "don't" — or "3–5" and "3-5" — compare equal.
_UNICODE_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    " ": " ", "…": "...",
}
#: Tokenizer: a number (with decimal/thousands separators and an optional
#: percent sign) or a word (which may contain internal hyphens/apostrophes).
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*%?|[a-z0-9]+(?:['\-][a-z0-9]+)*")
_NUMERIC_TOKEN_RE = re.compile(r"^\d")

#: Deliberately small. A large stopword list starts deleting the words that
#: carry the answer ("no", "not", "all" are stopwords in many lists and are
#: exactly what distinguishes a right answer from a wrong one), so this covers
#: only closed-class filler.
STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "that", "this", "these", "those", "with", "from",
        "into", "its", "it's", "are", "was", "were", "been", "being", "has",
        "have", "had", "which", "who", "whom", "whose", "what", "when", "where",
        "how", "why", "but", "also", "such", "than", "then", "there", "their",
        "they", "them", "our", "ours", "you", "your", "yours", "his", "her",
        "hers", "she", "him", "its'", "can", "could", "would", "should", "will",
        "shall", "may", "might", "must", "does", "did", "doing", "done", "each",
        "any", "some", "more", "most", "much", "many", "very", "about", "over",
        "under", "between", "because", "while", "during", "both", "either",
        "neither", "same", "other", "another", "used", "using", "use", "uses",
    }
)

#: Submissions that are explicit non-answers, in source form. Folded through
#: :func:`punctuation_insensitive` into :data:`_NON_ANSWERS` below.
_NON_ANSWER_SOURCES: tuple[str, ...] = (
    "",
    "-",
    "?",
    "??",
    "idk",
    "i don't know",
    "i do not know",
    "don't know",
    "dunno",
    "no idea",
    "not sure",
    "unsure",
    "no clue",
    "n/a",
    "na",
    "none",
    "nothing",
    "skip",
    "pass",
    "tbd",
    "todo",
)


def normalize_text(value: Optional[str]) -> str:
    """Casefold, ASCII-fold and whitespace-normalize ``value``.

    Also strips leading/trailing punctuation, so ``"Attention!"`` and
    ``"attention"`` are the same string. Internal punctuation is preserved here
    — :func:`punctuation_insensitive` is the looser key.
    """
    text = unicodedata.normalize("NFKC", value or "")
    for src, dst in _UNICODE_FOLD.items():
        text = text.replace(src, dst)
    text = _WS_RE.sub(" ", text).strip().casefold()
    return _EDGE_PUNCT_RE.sub("", text).strip()


def punctuation_insensitive(value: Optional[str]) -> str:
    """Normalize, then reduce every run of non-alphanumerics to one space.

    This is the key used for the "exact match" leg of short-answer grading, so
    ``self-attention``, ``self attention`` and ``Self-Attention.`` all collapse
    to ``self attention``. Decimal points inside numbers are the one casualty
    (``0.5`` becomes ``0 5``) — harmless, because both sides are folded the same
    way and the token path keeps the number intact.
    """
    normalized = normalize_text(value)
    return _WS_RE.sub(" ", re.sub(r"[^a-z0-9]+", " ", normalized)).strip()


#: The comparison keys for :func:`is_non_answer`. Built once, from the source
#: strings above, through the exact same fold a submission goes through.
_NON_ANSWERS: frozenset[str] = frozenset(
    punctuation_insensitive(text) for text in _NON_ANSWER_SOURCES
)


def _fold_token(token: str) -> str:
    """Fold a trivial English plural so ``layers`` matches ``layer``.

    Deliberately the *only* stemming performed: anything more aggressive
    (``-ing``/``-ed`` stripping, Porter) starts conflating genuinely different
    technical terms, and this grader errs toward saying "wrong".
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(value: Optional[str]) -> list[str]:
    """Return the meaning-carrying tokens of ``value``, in order.

    A token counts as content if it is numeric (any length — ``8`` layers is the
    whole answer) or is at least 3 characters long and not a
    :data:`STOPWORDS` entry.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(normalize_text(value)):
        if _NUMERIC_TOKEN_RE.match(raw):
            tokens.append(raw.rstrip("."))
            continue
        # Split a hyphenated/apostrophised compound into its parts so
        # "byte-pair encoding" and "byte pair encodings" agree. Both sides go
        # through this, so the comparison stays symmetric.
        for part in re.split(r"['\-]", raw):
            if len(part) < 3 or part in STOPWORDS:
                continue
            tokens.append(part)
    return tokens


def token_overlap(expected: Optional[str], submitted: Optional[str]) -> float:
    """Share of ``expected``'s content tokens that appear in ``submitted``.

    Returns ``0.0`` when ``expected`` has no content tokens at all (e.g. the
    accepted answer is the single stopword-ish word "all"): in that case only
    the exact-match leg may declare a submission correct, which is the
    conservative choice.
    """
    wanted = {_fold_token(tok) for tok in content_tokens(expected)}
    if not wanted:
        return 0.0
    given = {_fold_token(tok) for tok in content_tokens(submitted)}
    if not given:
        return 0.0
    hits = sum(1 for tok in wanted if tok in given)
    return hits / len(wanted)


def is_non_answer(submitted: Optional[str]) -> bool:
    """True for a blank submission or an explicit "I don't know".

    Without this, a single-content-token expected answer would be matched by any
    sentence that happens to contain that token — including "I have no idea what
    a transformer is", which must never score as correct.
    """
    return punctuation_insensitive(submitted) in _NON_ANSWERS


# --------------------------------------------------------------------------
# Graders
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ShortAnswerVerdict:
    """Why a short answer was (or was not) accepted — useful in tests and logs."""

    is_correct: bool
    matched_answer: Optional[str] = None
    overlap: float = 0.0
    exact: bool = False


def grade_choice(
    selected_option_id: Optional[str], correct_option_id: Optional[str]
) -> bool:
    """Exact option-id equality (case/whitespace tolerant).

    ``None`` on either side is incorrect: an unanswered question is wrong, and a
    question with no recorded answer key can never be satisfied.
    """
    if not selected_option_id or not correct_option_id:
        return False
    return selected_option_id.strip().casefold() == correct_option_id.strip().casefold()


def grade_short_answer(
    submitted: Optional[str],
    accepted_answers: Sequence[str],
    *,
    threshold: float = SHORT_ANSWER_TOKEN_OVERLAP_THRESHOLD,
) -> ShortAnswerVerdict:
    """Grade a free-text answer against the accepted set. Never raises.

    Two legs, in order: punctuation-insensitive exact match, then content-token
    overlap against each accepted answer (best match wins). An explicit
    non-answer short-circuits to incorrect before either leg runs.
    """
    if is_non_answer(submitted):
        return ShortAnswerVerdict(is_correct=False)

    submitted_key = punctuation_insensitive(submitted)
    if not submitted_key:
        return ShortAnswerVerdict(is_correct=False)

    candidates = [ans for ans in (accepted_answers or []) if (ans or "").strip()]
    for accepted in candidates:
        if punctuation_insensitive(accepted) == submitted_key:
            return ShortAnswerVerdict(
                is_correct=True, matched_answer=accepted, overlap=1.0, exact=True
            )

    best_overlap = 0.0
    best_answer: Optional[str] = None
    for accepted in candidates:
        overlap = token_overlap(accepted, submitted)
        if overlap > best_overlap:
            best_overlap = overlap
            best_answer = accepted

    if best_answer is not None and best_overlap >= threshold:
        return ShortAnswerVerdict(
            is_correct=True, matched_answer=best_answer, overlap=best_overlap
        )
    return ShortAnswerVerdict(
        is_correct=False, matched_answer=best_answer, overlap=best_overlap
    )


def expected_answer_label(question: QuizQuestion) -> str:
    """Human-readable "the right answer was …" string for ``question``.

    For a choice question that is the winning option's *text* (the id ``c`` is
    meaningless to a reader); for a short answer it is the canonical answer.
    """
    if question.kind is QuestionKind.SHORT_ANSWER:
        return (question.correct_answer_text or "").strip()
    for option in question.options:
        if grade_choice(option.id, question.correct_option_id):
            return (option.text or "").strip()
    return (question.correct_option_id or "").strip()


def feedback_explanation(question: QuizQuestion) -> str:
    """The explanation as shown after grading, with the answer spelled out.

    Short-answer grading is tolerant but imperfect, so the expected answer is
    appended whenever the explanation does not already contain it — the user can
    then self-assess a near miss instead of being told only "incorrect".
    """
    explanation = (question.explanation or "").strip()
    if question.kind is not QuestionKind.SHORT_ANSWER:
        return explanation
    expected = expected_answer_label(question)
    if not expected:
        return explanation
    if expected and punctuation_insensitive(expected) in punctuation_insensitive(
        explanation
    ):
        return explanation
    suffix = f"Expected answer: **{expected}**."
    return f"{explanation} {suffix}".strip() if explanation else suffix


def grade_question(
    question: QuizQuestion, submitted: Optional[SubmittedAnswer]
) -> GradedAnswer:
    """Grade one question and build its :class:`GradedAnswer`.

    This is the *post-commit* shape, so it is the one place where the answer key
    and the explanation are allowed to travel to the client. Pure: it reads only
    its two arguments.

    Any ``is_correct``-looking value on the submission is ignored — ``is_correct``
    below is always the server's own verdict.
    """
    selected_option_id = submitted.selected_option_id if submitted else None
    answer_text = submitted.answer_text if submitted else None

    if question.kind is QuestionKind.SHORT_ANSWER:
        verdict = grade_short_answer(answer_text, question.accepted_answers)
        is_correct = verdict.is_correct
        # A short answer never carries an option id; drop a stray one rather
        # than echoing a value the UI would try to highlight.
        selected_option_id = None
    else:
        is_correct = grade_choice(selected_option_id, question.correct_option_id)
        answer_text = None

    return GradedAnswer(
        question_id=question.id,
        index=question.index,
        prompt=question.prompt,
        kind=question.kind,
        difficulty=question.difficulty,
        options=list(question.options),
        selected_option_id=selected_option_id,
        answer_text=answer_text,
        is_correct=is_correct,
        correct_option_id=question.correct_option_id,
        correct_answer_text=question.correct_answer_text,
        explanation=feedback_explanation(question),
        citation=question.citation,
    )


def grade_answers(
    questions: Sequence[QuizQuestion], submitted: Iterable[SubmittedAnswer]
) -> list[GradedAnswer]:
    """Grade every question in ``questions`` against ``submitted``.

    ``questions`` defines the scope and the output order: a question with no
    matching submission is graded incorrect (unanswered), and a submission for a
    question outside the scope is ignored here — the caller rejects those with a
    400 before grading, so reaching this function they are already impossible.
    Duplicate submissions for one question resolve **last-wins**.
    """
    by_question: dict[str, SubmittedAnswer] = {}
    for answer in submitted or []:
        if answer and answer.question_id:
            by_question[answer.question_id] = answer
    return [grade_question(q, by_question.get(q.id)) for q in questions]
