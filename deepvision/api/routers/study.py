"""Study router — the cross-paper study surface.

Owned by: **STUDY-CORE builder**. Do not edit ``flashcards.py`` or ``quiz.py``,
and never touch ``routers/__init__.py`` — this module is already registered.

Routes (all mounted under ``/api/study``):

- ``GET /api/study/overview`` -> :class:`StudyOverviewResponse` — the header
  numbers for the Study screen: due today, due this week, library-wide deck
  counters, quiz counts, a 7-day review sparkline and the streak. Counts only,
  no card bodies; cheap enough to refetch on window focus.
- ``GET /api/study/due`` -> :class:`DueQueueResponse` — **the** cross-paper
  review queue, and the reason Study is a top-level nav item. Params:
  ``limit`` (default 50, max 200), ``paper_id`` (optional filter — this is what
  the Report launch card deep-links to), ``include_new`` (default true),
  ``shuffle`` (default false). Order: ``due_at ASC, id ASC`` (most overdue
  first). Shuffle applies **after** ORDER BY + LIMIT.
- ``GET /api/study/papers`` -> :class:`StudyPapersResponse` — per-paper stats for
  every paper with study material; one request instead of N.
- ``GET /api/study/papers/{paper_id}/stats`` -> :class:`PaperStudyStats` — the
  single-paper version, and exactly what the two Report launch cards render
  ("Flashcards - 24 cards - 6 due"). 404 if the paper does not exist; a paper
  with no deck and no quizzes returns zeros with ``has_deck: false``, which is
  the "Generate" state, not an error.

Implementation notes:

- Every count that scales with the library (the due queue, the library-wide
  totals) is computed in SQL with ``COUNT``/``SUM(CASE ...)``/``MIN``/``MAX``
  against the ``ix_flashcards_due_at`` / ``ix_flashcards_paper_due`` indexes —
  never a full-table read filtered in Python. The one exception is the 7-day
  review sparkline, which reads raw ``reviewed_at`` timestamps, but only for a
  7-day window, so it is bounded by review *volume in a week*, not by the size
  of the library.
- Everything here is read-only and synchronous. No model calls, ever.
- This router does not import ``routers/flashcards.py`` or ``routers/quiz.py``
  (they are sibling routers), but row -> Pydantic conversion is **not**
  re-implemented here: it comes from :mod:`deepvision.study.card_queries`,
  which is the shared, non-router home for it. An earlier revision duplicated
  that conversion and the copy quietly lost the original's defensive
  coercions — a single card whose persisted ``ease_factor`` sat below the 1.3
"""

from __future__ import annotations

import random
from collections import Counter
from datetime import datetime, time, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException
from sqlalchemy import case, func
from sqlmodel import select

from deepvision.api.schemas import (
    DueQueueResponse,
    PaperStudyStats,
    StudyOverviewResponse,
    StudyPapersResponse,
)
from deepvision.db import session_scope
from deepvision.db.schema import (
    FlashcardReviewRow,
    FlashcardRow,
    PaperRow,
    QuizAttemptRow,
    QuizRow,
)
from deepvision.models.study import (
    AttemptMode,
    DeckProgress,
    DueCard,
)
from deepvision.study.card_queries import (
    card_from_row,
    count_due,
    order_for_queue,
    strengths_for,
)
from deepvision.study.session_scheduler import local_day_bounds_utc
from deepvision.utils import get_logger

__all__ = ["router"]

router = APIRouter(prefix="/study", tags=["study"])

log = get_logger(__name__)


#: Distinct-day lookback for the streak calculation is unbounded by design (a
#: DISTINCT query over an indexed timestamp column costs one row per *day the
#: app was used*, not per review — so there is no library-growth reason to cap
#: it the way the due-queue read is capped).


# --------------------------------------------------------------------------
# Row -> Pydantic conversion
#
# Deliberately a thin alias, not a second implementation. ``card_from_row``
# re-applies the ease floor and coerces junk enum/None columns on the way out,
# so a card written by an older build cannot 500 the cross-paper due queue —
# which, being library-wide, would take the whole Study screen down over one
# bad row.
# --------------------------------------------------------------------------
_flashcard_from_row = card_from_row


# --------------------------------------------------------------------------
# Aggregate helpers (all pure SQL COUNT/SUM/MIN/MAX — no full-row reads)
# --------------------------------------------------------------------------
def _deck_progress_totals(session, now: datetime) -> DeckProgress:
    """Library-wide :class:`DeckProgress` — every flashcard, across every paper."""
    total = int(
        session.exec(select(func.count(FlashcardRow.id))).one() or 0
    )
    starred = int(
        session.exec(
            select(func.count(FlashcardRow.id)).where(FlashcardRow.starred.is_(True))
        ).one()
        or 0
    )
    day_start, day_end = local_day_bounds_utc(now)
    # DISTINCT: a card rated `again` and seen three more times in the same
    # sitting is one card studied, so it counts down the day's workload once.
    seen_today = int(
        session.exec(
            select(func.count(func.distinct(FlashcardReviewRow.card_id)))
            .where(FlashcardReviewRow.reviewed_at >= day_start)
            .where(FlashcardReviewRow.reviewed_at < day_end)
        ).one()
        or 0
    )
    reviews_today = int(
        session.exec(
            select(func.count(FlashcardReviewRow.id))
            .where(FlashcardReviewRow.reviewed_at >= day_start)
            .where(FlashcardReviewRow.reviewed_at < day_end)
        ).one()
        or 0
    )
    return DeckProgress(
        total=total,
        due_count=max(0, total - seen_today),
        starred_count=starred,
        reviews_today=reviews_today,
    )


def _reviews_last_7_days(session, now: datetime) -> list[int]:
    """Review counts per day, oldest -> newest, length 7 (today inclusive)."""
    today = now.date()
    days = [today - timedelta(days=i) for i in range(6, -1, -1)]
    window_start = datetime.combine(days[0], time.min)
    timestamps = session.exec(
        select(FlashcardReviewRow.reviewed_at).where(
            FlashcardReviewRow.reviewed_at >= window_start
        )
    ).all()
    counts = Counter(ts.date() for ts in timestamps)
    return [counts.get(d, 0) for d in days]


def _streak_days(session) -> int:
    """Consecutive days (ending today) with >= 1 review, across every paper."""
    day_strings = session.exec(
        select(func.date(FlashcardReviewRow.reviewed_at)).distinct()
    ).all()
    review_days: set = set()
    for value in day_strings:
        if not value:
            continue
        try:
            review_days.add(datetime.strptime(value, "%Y-%m-%d").date())
        except (TypeError, ValueError):
            continue

    streak = 0
    day = datetime.utcnow().date()
    while day in review_days:
        streak += 1
        day -= timedelta(days=1)
    return streak


def _build_paper_stats(
    session,
    paper_id: str,
    now: datetime,
    *,
    paper_row: Optional[PaperRow] = None,
) -> PaperStudyStats:
    """Launch-card stats for one paper. Never raises for a paper with no study material."""
    if paper_row is None:
        paper_row = session.get(PaperRow, paper_id)
    paper_title = paper_row.title if paper_row else ""
    paper_label = paper_row.arxiv_label if paper_row else ""

    deck_row = session.exec(
        select(
            func.count(FlashcardRow.id),
            func.sum(case((FlashcardRow.starred.is_(True), 1), else_=0)),
            # No dedicated "deck generated at" timestamp is persisted anywhere
            # (there is no decks table — only individual card rows), and
            # updated_at is unusable as a proxy because a *review* also bumps
            # it. The newest card's created_at is the best available signal:
            # regenerating a deck upserts existing cards (created_at
            # preserved) and typically introduces at least one new one.
            func.max(FlashcardRow.created_at),
        ).where(FlashcardRow.paper_id == paper_id)
    ).first()
    cards_total, cards_starred, deck_generated_at = deck_row or (0, 0, None)
    cards_total = int(cards_total or 0)

    quiz_count = session.exec(
        select(func.count(QuizRow.id)).where(QuizRow.paper_id == paper_id)
    ).one()

    # ``best`` counts FULL attempts only, and falls back to any attempt only
    # when the paper has never had a full one.
    #
    # A ``retry_missed`` run is scored over a smaller, self-selected denominator
    # — the questions you already got wrong — so a plain MAX() over every
    # attempt lets "6/6 on the six I missed" (1.0) outrank the real "4/10"
    # (0.4) that produced it. This value is what the Report page's Quiz launch
    # card renders as "best 100%", and ``routers/quiz.py`` already refuses to
    # mix the two modes for exactly that reason; taking the plain MAX here made
    # the launch card the one surface in the app that disagreed with the
    # attempt history sitting one click away.
    attempt_row = session.exec(
        select(
            func.count(QuizAttemptRow.id),
            func.max(
                case(
                    (QuizAttemptRow.mode == AttemptMode.FULL.value, QuizAttemptRow.score),
                    else_=None,
                )
            ),
            func.max(QuizAttemptRow.score),
            func.max(QuizAttemptRow.finished_at),
        ).where(QuizAttemptRow.paper_id == paper_id)
    ).first()
    quiz_attempt_count, best_full, best_any, quiz_last_attempted_at = attempt_row or (
        0,
        None,
        None,
        None,
    )
    quiz_best_score = best_full if best_full is not None else best_any

    return PaperStudyStats(
        paper_id=paper_id,
        paper_title=paper_title,
        paper_label=paper_label,
        has_deck=cards_total > 0,
        cards_total=cards_total,
        cards_starred=int(cards_starred or 0),
        deck_generated_at=deck_generated_at,
        quiz_count=int(quiz_count or 0),
        quiz_attempt_count=int(quiz_attempt_count or 0),
        quiz_best_score=quiz_best_score,
        quiz_last_attempted_at=quiz_last_attempted_at,
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@router.get("/overview", response_model=StudyOverviewResponse)
def get_study_overview() -> StudyOverviewResponse:
    """Cross-paper study counters for the Study screen header."""
    now = datetime.utcnow()
    with session_scope() as session:
        totals = _deck_progress_totals(session, now)

        papers_with_decks = session.exec(
            select(func.count(func.distinct(FlashcardRow.paper_id)))
        ).one()
        papers_with_quizzes = session.exec(
            select(func.count(func.distinct(QuizRow.paper_id)))
        ).one()
        quiz_count = session.exec(select(func.count(QuizRow.id))).one()
        quiz_attempt_count = session.exec(select(func.count(QuizAttemptRow.id))).one()
        # Quizzes never attempted. A backlog, not a daily figure: a quiz is a
        # one-off assessment, so unlike `due` it does not reset at midnight.
        # DISTINCT because a quiz taken three times is still one quiz done.
        attempted_quizzes = session.exec(
            select(func.count(func.distinct(QuizAttemptRow.quiz_id)))
        ).one()
        quizzes_due = max(0, int(quiz_count or 0) - int(attempted_quizzes or 0))

        reviews_last_7_days = _reviews_last_7_days(session, now)
        streak_days = _streak_days(session)

    return StudyOverviewResponse(
        due=totals.due_count,
        quizzes_due=quizzes_due,
        totals=totals,
        papers_with_decks=int(papers_with_decks or 0),
        papers_with_quizzes=int(papers_with_quizzes or 0),
        quiz_count=int(quiz_count or 0),
        quiz_attempt_count=int(quiz_attempt_count or 0),
        reviews_last_7_days=reviews_last_7_days,
        streak_days=streak_days,
    )


@router.get("/due", response_model=DueQueueResponse)
def get_due_queue(
    limit: int = 50,
    paper_id: Optional[str] = None,
    include_new: bool = True,
    shuffle: bool = False,
) -> DueQueueResponse:
    """The cross-paper due queue, most overdue first.

    ``due_at <= now`` is filtered in SQL against the composite/plain due-date
    indexes; the ``ORDER BY due_at ASC, id ASC`` + ``LIMIT`` also run in SQL, so
    only the page actually returned is ever materialized into Python. Shuffle
    (when requested) is applied to that already-limited page only.
    """
    limit = max(1, min(limit, 200))
    now = datetime.utcnow()

    with session_scope() as session:
        base = select(FlashcardRow)
        if paper_id is not None:
            base = base.where(FlashcardRow.paper_id == paper_id)

        all_rows = list(session.exec(base).all())
        # The same definition the header tile uses: cards left to do TODAY, not
        # the deck size. These two were briefly different numbers both labelled
        # "due" — the field kept its SM-2 meaning ("due_at <= now") after the
        # filter it named was deleted.
        total_due = count_due(session, paper_id=paper_id, now=now)

        # Queue order is derived from the review log, so it cannot be an SQL
        # ORDER BY. Sort the scope, then take the page.
        strengths = strengths_for(session, [r.id for r in all_rows], now=now)
        rows = order_for_queue(all_rows, strengths)[:limit]

        if shuffle and rows:
            random.shuffle(rows)

        paper_ids = {r.paper_id for r in rows}
        papers: dict[str, PaperRow] = {}
        if paper_ids:
            paper_rows = session.exec(
                select(PaperRow).where(PaperRow.id.in_(paper_ids))
            ).all()
            papers = {p.id: p for p in paper_rows}

        cards: list[DueCard] = []
        for row in rows:
            paper = papers.get(row.paper_id)
            cards.append(
                DueCard(
                    card=_flashcard_from_row(row, strengths.get(row.id)),
                    paper_title=paper.title if paper else "",
                    paper_label=paper.arxiv_label if paper else "",
                )
            )

    return DueQueueResponse(
        cards=cards,
        count=len(cards),
        total_due=int(total_due or 0),
        paper_id=paper_id,
        shuffled=shuffle,
    )


@router.get("/papers", response_model=StudyPapersResponse)
def list_study_papers() -> StudyPapersResponse:
    """Per-paper study stats for every paper with a deck or a quiz."""
    now = datetime.utcnow()
    with session_scope() as session:
        deck_paper_ids = set(session.exec(select(FlashcardRow.paper_id).distinct()).all())
        quiz_paper_ids = set(session.exec(select(QuizRow.paper_id).distinct()).all())
        paper_ids = deck_paper_ids | quiz_paper_ids
        if not paper_ids:
            return StudyPapersResponse(papers=[], count=0)

        paper_rows = {
            p.id: p
            for p in session.exec(select(PaperRow).where(PaperRow.id.in_(paper_ids))).all()
        }
        stats = [
            _build_paper_stats(session, pid, now, paper_row=paper_rows.get(pid))
            for pid in paper_ids
        ]

    stats.sort(key=lambda s: (s.paper_title or s.paper_id).lower())
    return StudyPapersResponse(papers=stats, count=len(stats))


@router.get("/papers/{paper_id}/stats", response_model=PaperStudyStats)
def get_paper_study_stats(paper_id: str) -> PaperStudyStats:
    """Launch-card stats for one paper (what the Report page renders).

    404 only if the *paper* does not exist. A paper with no deck and no quiz
    yet returns a valid zero-state payload (``has_deck: false``) — that is the
    "Generate" state for the launch cards, not an error.
    """
    now = datetime.utcnow()
    with session_scope() as session:
        paper_row = session.get(PaperRow, paper_id)
        if paper_row is None:
            raise HTTPException(status_code=404, detail=f"paper not found: {paper_id!r}")
        return _build_paper_stats(session, paper_id, now, paper_row=paper_row)
