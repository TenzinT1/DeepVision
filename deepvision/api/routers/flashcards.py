"""Flashcards router — deck generation and SM-2 review.

Owned by: **FLASHCARDS/SRS builder**. Does not touch ``study.py`` or ``quiz.py``,
and never ``routers/__init__.py`` — this module is already registered there.

Routes (all mounted under ``/api/flashcards``):

- ``POST /api/flashcards/generate`` → :class:`FlashcardGenerateResponse`
  (**202 queued** / 200 cached). Generation runs an LLM pass, so it is
  job-based: an :class:`~deepvision.models.IngestJob` is minted, persisted
  before the response returns, and run on a daemon thread by
  ``api/study_job_runner.py``, so the frontend's existing ``pollJob`` drives the
  UI unchanged. The job also builds a retriever (``_build_retriever``, wrapping
  ``report.agent_bridge.build_retriever``) so the deck is sourced from a
  whole-paper pool, not just the report; a retriever that fails to build
  degrades to report-only generation rather than failing the job.
- ``GET /api/flashcards/deck/{paper_id}`` → :class:`FlashcardDeckResponse`.
- ``POST /api/flashcards/{card_id}/review`` → :class:`FlashcardReviewResponse`.
  **Synchronous, and never calls a model** — it is arithmetic
  (:mod:`deepvision.study.session_scheduler`) plus one append.
- ``POST /api/flashcards/{card_id}/star`` → :class:`FlashcardStarResponse`.
  Sets (does not toggle) ``starred``, so a retry cannot flip it back.
- ``POST /api/flashcards/deck/{paper_id}/reset`` → :class:`FlashcardResetResponse`.

Three invariants worth stating where they are easy to break:

**The review path writes two rows, and one of them is append-only.** The
``flashcards`` row is updated in place; a ``flashcard_reviews`` row bracketing
the transition (``prior_*`` → ``post_*``) is *inserted*. That log is never
updated and never deleted — replaying it reproduces the card's current state,
which is the only way to answer "why is this due in 33 days".

**Reviews are serialized per card.** Two ratings for the same card arriving
together (a double-tap, a retried request, two tabs) would otherwise both read
the same prior state and both write a review row, double-advancing the schedule
from a single answer. :func:`_card_lock` makes the read-modify-write one
critical section.

**Due is decided in SQL.** Every due count and every due listing filters
``due_at <= now`` against the ``due_at`` indexes (see
:mod:`deepvision.study.card_queries`). Nothing here reads a deck into Python to
filter it.
"""

from __future__ import annotations

import random
import threading
from datetime import datetime
from typing import NamedTuple, Optional

from fastapi import APIRouter, HTTPException, Response

from deepvision.api.deps import get_settings
from deepvision.api.schemas import (
    FlashcardDeckResponse,
    FlashcardGenerateRequest,
    FlashcardGenerateResponse,
    FlashcardResetResponse,
    FlashcardReviewRequest,
    FlashcardReviewResponse,
    FlashcardStarRequest,
    FlashcardStarResponse,
)
from deepvision.api.study_job_runner import ProgressFn, launch_study_job
from deepvision.db import session_scope
from deepvision.db.schema import FlashcardReviewRow, FlashcardRow, PaperRow
from deepvision.study import card_queries as q
from deepvision.study.deck_generator import card_count, generate_deck, has_report
from deepvision.study.session_scheduler import requeue_gap
from deepvision.utils import get_logger
from deepvision.utils.ids import new_id

__all__ = ["router"]

router = APIRouter(prefix="/flashcards", tags=["flashcards"])

log = get_logger(__name__)

#: Job kind, used as half of the study job runner's in-flight reservation key.
_JOB_KIND = "flashcards"

#: Per-card review locks. Reviews are rare and in-process, so a plain dict of
#: locks (guarded by ``_LOCKS_GUARD`` while it is being read or extended) is
#: both sufficient and obviously correct. It is bounded by the number of cards
#: reviewed since boot, which is exactly the working set.
_CARD_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _card_lock(card_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _CARD_LOCKS.get(card_id)
        if lock is None:
            lock = threading.Lock()
            _CARD_LOCKS[card_id] = lock
        return lock


class _Paper(NamedTuple):
    """The three paper fields these responses need, detached from the session.

    Returning the ORM row itself would be a lazy-load trap: the session closes
    before the response is built.
    """

    id: str
    arxiv_id: str
    title: str


def _build_retriever(settings):
    """A retriever for the whole-paper source pool, or ``None`` on any failure.

    Deck generation must never error because the pool couldn't be built —
    ``generate_deck``/``build_deck`` both already degrade a ``None`` retriever
    to today's report-only behaviour, so a broken embedding provider here
    costs variety, not the job.
    """
    try:
        from deepvision.report.agent_bridge import build_retriever

        return build_retriever(settings)
    except Exception:  # noqa: BLE001 - the pool is a bonus, never a blocker
        log.warning(
            "could not build a retriever for the flashcard source pool; "
            "falling back to report-only generation",
            exc_info=True,
        )
        return None


def _require_paper(paper_id: str) -> _Paper:
    with session_scope() as session:
        row = session.get(PaperRow, paper_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"paper not found: {paper_id}")
        return _Paper(id=row.id, arxiv_id=row.arxiv_id or "", title=row.title or "")


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


# --------------------------------------------------------------------------
# Generate
# --------------------------------------------------------------------------
@router.post("/generate", response_model=FlashcardGenerateResponse, status_code=202)
def generate_flashcards(
    request: FlashcardGenerateRequest, response: Response
) -> FlashcardGenerateResponse:
    """Queue deck generation (202), or return the existing deck (200).

    ``force`` regenerates: cards are upserted on ``(paper_id, content_key)``, so
    a card that survives the regeneration keeps its schedule and a ``manual``
    card is not touched at all.
    """
    paper = _require_paper(request.paper_id)

    if not request.force and card_count(request.paper_id) > 0:
        now = q.utcnow()
        with session_scope() as session:
            rows = q.load_deck(session, request.paper_id, now=now)
            cards = [q.card_from_row(row) for row in rows]
            progress = q.deck_progress(session, request.paper_id, now=now)
        response.status_code = 200
        return FlashcardGenerateResponse(
            paper_id=request.paper_id,
            cached=True,
            job=None,
            deck=cards,
            progress=progress,
        )

    if not has_report(request.paper_id):
        # A precondition the user can fix in one click, not a server fault: the
        # deck is built from the report's sections.
        raise HTTPException(
            status_code=409,
            detail=(
                f"paper {request.paper_id!r} has no report yet — open the Report "
                "tab to generate one first; flashcards are built from it"
            ),
        )

    settings = request.settings_override or get_settings()
    target = int(request.count)

    def work(progress: ProgressFn) -> str:
        result = generate_deck(
            request.paper_id,
            settings,
            target_count=target,
            on_progress=progress,
            retriever=_build_retriever(settings),
        )
        return f"Deck ready: {result.summary}"

    try:
        job = launch_study_job(
            _JOB_KIND,
            request.paper_id,
            work,
            arxiv_id=paper.arxiv_id or "",
            start_message="Building flashcards from the report",
            done_message="Deck ready",
        ).job
    except Exception as exc:  # pragma: no cover - defensive
        log.error(
            "failed to queue flashcard generation",
            extra={"paper_id": request.paper_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=500, detail=f"failed to queue flashcard generation: {exc}"
        ) from exc

    response.status_code = 202
    return FlashcardGenerateResponse(
        paper_id=request.paper_id, cached=False, job=job, deck=None, progress=None
    )


# --------------------------------------------------------------------------
# Deck
# --------------------------------------------------------------------------
@router.get("/deck/{paper_id}", response_model=FlashcardDeckResponse)
def get_deck(
    paper_id: str,
    state: Optional[str] = None,
    starred: Optional[bool] = None,
    due_only: bool = False,
    shuffle: bool = False,
) -> FlashcardDeckResponse:
    """Return one paper's deck with its progress counters.

    Cards come back most-overdue-first. ``shuffle`` shuffles the *returned*
    list, after the filter and the ordering — never the query, which would
    break the "oldest debt first" contract the ordering exists to provide.
    ``progress`` always describes the whole deck, not the filtered slice, so the
    header counters do not jump when the user toggles a filter.
    """
    paper = _require_paper(paper_id)
    now = q.utcnow()

    with session_scope() as session:
        rows = q.load_deck(
            session,
            paper_id,
            starred=starred,
            due_only=due_only,
            now=now,
        )
        cards = [q.card_from_row(row) for row in rows]
        progress = q.deck_progress(session, paper_id, now=now)
        generated_at = q.deck_generated_at(session, paper_id)

    if shuffle:
        random.shuffle(cards)

    return FlashcardDeckResponse(
        paper_id=paper_id,
        paper_title=paper.title or "",
        cards=cards,
        progress=progress,
        generated_at=_iso(generated_at),
        shuffled=shuffle,
    )


# --------------------------------------------------------------------------
# Review — the SM-2 write path
# --------------------------------------------------------------------------
@router.post("/{card_id}/review", response_model=FlashcardReviewResponse)
def review_card(card_id: str, request: FlashcardReviewRequest) -> FlashcardReviewResponse:
    """Apply one SM-2 rating. Synchronous; never calls a model.

    Serialized per card so two ratings that arrive together cannot both read the
    same prior state and double-advance the schedule from one answer.
    """
    with _card_lock(card_id):
        # Stamped *inside* the lock: taken before it, two queued ratings could be
        # written with timestamps in the opposite order to the order they were
        # applied, and the append-only log would no longer replay to the card's
        # actual state.
        now = q.utcnow()
        with session_scope() as session:
            row = session.get(FlashcardRow, card_id)
            if row is None:
                raise HTTPException(
                    status_code=404, detail=f"flashcard not found: {card_id}"
                )

            review_id = new_id("rev")
            session.add(
                FlashcardReviewRow(
                    id=review_id,
                    card_id=row.id,
                    paper_id=row.paper_id,
                    rating=request.rating.value,
                    elapsed_ms=request.elapsed_ms,
                    reviewed_at=now,
                    # prior_*/post_*/was_lapse/due_at_after are left at their
                    # schema defaults: they described SM-2 transitions, and
                    # there are none any more. The columns survive only
                    # because create_all cannot drop them.
                )
            )
            session.flush()

            paper_id = row.paper_id

    # A second short session so the counters are read *after* the commit above
    # — otherwise reviews_today would be one behind its own review.
    with session_scope() as session:
        progress = q.deck_progress(session, paper_id, now=now)
        remaining = q.count_due(session, paper_id=paper_id, now=now)
        seen_today = q.seen_today_count(session, paper_id=paper_id, now=now)
        # Derived AFTER the commit above, so it counts the review just written.
        strength = (
            q.strengths_for(session, [card_id], now=now).get(card_id)
            or CardStrength()
        ).strength

    return FlashcardReviewResponse(
        card_id=card_id,
        rating=request.rating.value,
        review_id=review_id,
        requeue_after=requeue_gap(request.rating),
        strength=strength,
        seen_today=seen_today,
        due=remaining,
        progress=progress,
    )


# --------------------------------------------------------------------------
# Star
# --------------------------------------------------------------------------
@router.post("/{card_id}/star", response_model=FlashcardStarResponse)
def star_card(card_id: str, request: FlashcardStarRequest) -> FlashcardStarResponse:
    """Set the star flag on a card (idempotent — the client sends the value)."""
    now = q.utcnow()
    with session_scope() as session:
        row = session.get(FlashcardRow, card_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"flashcard not found: {card_id}")
        row.starred = bool(request.starred)
        row.updated_at = now
        session.add(row)
        session.flush()
        paper_id = row.paper_id
        starred = row.starred

    with session_scope() as session:
        starred_count = q.deck_progress(session, paper_id, now=now).starred_count

    return FlashcardStarResponse(
        card_id=card_id, starred=starred, starred_count=starred_count
    )


# --------------------------------------------------------------------------
# Reset
# --------------------------------------------------------------------------
@router.post("/deck/{paper_id}/reset", response_model=FlashcardResetResponse)
def reset_deck(paper_id: str) -> FlashcardResetResponse:
    """Return every card in the deck to 'new'; keep text and review history.

    The ``flashcard_reviews`` log is deliberately left intact. It is append-only
    history, and deleting it would make the reset itself unauditable — the one
    thing a "start over" button must not do is erase the evidence that it
    happened.
    """
    _require_paper(paper_id)
    now = q.utcnow()

    with session_scope() as session:
        rows = q.load_deck(session, paper_id, now=now)
        for row in rows:
            q.reset_row(row, now=now)
            session.add(row)
        session.flush()
        reset_count = len(rows)

    with session_scope() as session:
        progress = q.deck_progress(session, paper_id, now=now)

    log.info(
        "flashcard deck reset", extra={"paper_id": paper_id, "cards": reset_count}
    )
    return FlashcardResetResponse(
        paper_id=paper_id, reset_count=reset_count, progress=progress
    )
