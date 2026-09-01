"""Flashcard persistence helpers — row↔model conversion and the indexed SQL.

Everything the deck view, the review endpoint and the cross-paper due queue need
to touch the ``flashcards`` / ``flashcard_reviews`` tables, in one place so the
three routers cannot each grow their own subtly different due predicate.

**The one rule.** "Due" is ``due_at <= now``, and that comparison happens **in
SQL**, against ``ix_flashcards_due_at`` (library-wide) or
``ix_flashcards_paper_due`` (one paper). Never load a deck into Python and
filter it there: the library grows without bound, and a Study screen that has to
read every card the user has ever generated in order to show six is the single
most likely way this feature ends up feeling broken.

Ordering is always ``due_at ASC, id ASC`` — oldest debt first, with the id as a
tiebreaker so a page boundary can never drop or duplicate a card the way an
unstable sort would.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional, Sequence

from sqlalchemy import case
from sqlmodel import Session, col, func, select

from deepvision.db.schema import FlashcardReviewRow, FlashcardRow, PaperRow
from deepvision.models.report import SectionName
from deepvision.models.study import (
    CardStrength,
    DeckProgress,
    DueCard,
    Flashcard,
    FlashcardOrigin,
)
from deepvision.study.session_scheduler import (
    local_day_bounds_utc,
    queue_sort_key,
    strength_from_ratings,
)
from deepvision.utils import get_logger

__all__ = [
    "card_from_row",
    "count_due",
    "deck_generated_at",
    "deck_progress",
    "due_cards",
    "due_rows",
    "load_deck",
    "reviews_between",
    "seen_today_count",
    "strengths_for",
    "utcnow",
]

log = get_logger(__name__)

def utcnow() -> datetime:
    """Naive UTC now — matching how every ``datetime`` column here is stored.

    SQLite has no timezone type and the schema's defaults are
    ``datetime.utcnow``, so mixing in an aware ``datetime.now(timezone.utc)``
    would raise on the very first comparison. One helper, one convention.
    """
    return datetime.utcnow()


# --------------------------------------------------------------------------
# Row ↔ model
# --------------------------------------------------------------------------
def _enum_or(value: Optional[str], enum_cls, default):
    """Coerce a persisted string into ``enum_cls``, falling back on junk.

    Enum values are the wire contract, but a row written by an older build (or
    hand-edited) must not take down a review session — a card with an
    unrecognised origin is still a perfectly good card.
    """
    if not value:
        return default
    try:
        return enum_cls(value)
    except ValueError:
        log.warning(
            "unrecognised persisted enum value; using default",
            extra={"value": value, "enum": enum_cls.__name__},
        )
        return default


def strengths_for(
    session: Session,
    card_ids: Sequence[str],
    *,
    now: Optional[datetime] = None,
) -> dict[str, CardStrength]:
    """Derive every card's :class:`CardStrength` from the review log.

    One query for all of them, folded in Python — not one query per card. The
    log is append-only and small (it holds one row per answer ever given), and
    a per-card query would be N+1 against the busiest table in the app.

    Cards with no reviews are absent from the result; callers should treat a
    miss as a default :class:`CardStrength` (strength 0, never reviewed), which
    is exactly what a brand-new card is.
    """
    ids = list(dict.fromkeys(card_ids))
    if not ids:
        return {}
    day_start, day_end = local_day_bounds_utc(now)
    rows = session.exec(
        select(
            FlashcardReviewRow.card_id,
            FlashcardReviewRow.rating,
            FlashcardReviewRow.reviewed_at,
        )
        .where(col(FlashcardReviewRow.card_id).in_(ids))
        .order_by(col(FlashcardReviewRow.reviewed_at).desc())
    ).all()

    grouped: dict[str, list[tuple[str, datetime]]] = {}
    for card_id, rating, reviewed_at in rows:
        grouped.setdefault(card_id, []).append((rating or "", reviewed_at))

    out: dict[str, CardStrength] = {}
    for card_id, entries in grouped.items():
        # `entries` is newest-first because the query ordered it that way; the
        # strength fold depends on that and would silently invert otherwise.
        newest = entries[0][1]
        out[card_id] = CardStrength(
            strength=strength_from_ratings([r for r, _ in entries]),
            reviews=len(entries),
            last_reviewed_at=newest,
            seen_today=any(day_start <= when < day_end for _, when in entries),
        )
    return out


def seen_today_count(
    session: Session,
    *,
    paper_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> int:
    """Distinct cards reviewed during the viewer's LOCAL day.

    ``DISTINCT`` is the contract: a card rated ``again`` and then seen twice
    more in the same sitting is one card studied, not three, so the "due"
    figure it feeds counts down by one.
    """
    day_start, day_end = local_day_bounds_utc(now)
    statement = (
        select(func.count(func.distinct(FlashcardReviewRow.card_id)))
        .where(FlashcardReviewRow.reviewed_at >= day_start)
        .where(FlashcardReviewRow.reviewed_at < day_end)
    )
    if paper_id:
        statement = statement.where(FlashcardReviewRow.paper_id == paper_id)
    return int(session.exec(statement).one() or 0)


def card_from_row(
    row: FlashcardRow, strength: Optional[CardStrength] = None
) -> Flashcard:
    """Re-validate a persisted row into the API's :class:`Flashcard` shape.

    ``strength`` is derived by :func:`strengths_for`, not stored on the row.
    Omitting it yields the never-reviewed default, which is correct for a card
    with no history.
    """
    section: Optional[SectionName] = None
    if row.source_section:
        try:
            section = SectionName(row.source_section)
        except ValueError:
            section = None
    return Flashcard(
        id=row.id,
        paper_id=row.paper_id,
        front=row.front or "",
        back=row.back or "",
        hint=row.hint,
        origin=_enum_or(row.origin, FlashcardOrigin, FlashcardOrigin.FACT),
        content_key=row.content_key or "",
        source_section=section,
        source_page=row.source_page,
        chunk_id=row.chunk_id,
        tags=list(row.tags or []),
        starred=bool(row.starred),
        strength=strength or CardStrength(),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


# --------------------------------------------------------------------------
# Deck reads
# --------------------------------------------------------------------------
def load_deck(
    session: Session,
    paper_id: str,
    *,
    starred: Optional[bool] = None,
) -> list[FlashcardRow]:
    """One paper's cards, ordered by id.

    Deliberately NOT in queue order: queue order depends on derived strength,
    which no column holds, so it is applied by :func:`order_for_queue` once the
    log has been read. This is the raw deck, for the deck browser.
    """
    statement = select(FlashcardRow).where(FlashcardRow.paper_id == paper_id)
    if starred is not None:
        statement = statement.where(FlashcardRow.starred == starred)
    return list(session.exec(statement).order_by(col(FlashcardRow.id)).all())


def order_for_queue(
    rows: Sequence[FlashcardRow], strengths: dict[str, CardStrength]
) -> list[FlashcardRow]:
    """Sort rows into session order: weakest first, longest-unseen next.

    See :func:`~deepvision.study.session_scheduler.queue_sort_key`.
    """
    def key(row: FlashcardRow):
        st = strengths.get(row.id) or CardStrength()
        return queue_sort_key(st.strength, st.last_reviewed_at, row.id)

    return sorted(rows, key=key)


def due_rows(
    session: Session,
    *,
    paper_id: Optional[str] = None,
    limit: int = 50,
    now: Optional[datetime] = None,
) -> list[FlashcardRow]:
    """The session queue — **cross-paper** unless ``paper_id`` narrows it.

    Every card is eligible: with no dates there is no "not yet due". What the
    ordering does instead is put the cards you are worst at first, so a short
    sitting still spends its time where it is most useful.

    Unlike the SM-2 version this cannot push ORDER BY + LIMIT into SQLite,
    because strength is derived from the review log rather than stored. The
    rows are therefore loaded and sorted in Python. That is fine at this
    scale (hundreds of cards, one row per answer ever given) and would need
    revisiting only if a library reached tens of thousands.
    """
    statement = select(FlashcardRow)
    if paper_id:
        statement = statement.where(FlashcardRow.paper_id == paper_id)
    rows = list(session.exec(statement).all())
    ordered = order_for_queue(rows, strengths_for(session, [r.id for r in rows], now=now))
    return ordered[:limit] if limit and limit > 0 else ordered


def count_due(
    session: Session,
    *,
    paper_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> int:
    """Cards still to do today: total in scope minus distinct cards seen today.

    Floored at zero — re-reviewing a card you already did today must not push
    the figure negative.
    """
    statement = select(func.count()).select_from(FlashcardRow)
    if paper_id:
        statement = statement.where(FlashcardRow.paper_id == paper_id)
    total = int(session.exec(statement).one() or 0)
    return max(0, total - seen_today_count(session, paper_id=paper_id, now=now))


def due_cards(
    session: Session,
    rows: Sequence[FlashcardRow],
    *,
    now: Optional[datetime] = None,
) -> list[DueCard]:
    """Wrap rows as :class:`DueCard` s, resolving paper titles in ONE query.

    The queue is cross-paper, so the naive version is an N+1 of one ``papers``
    lookup per card. One ``IN`` query instead.
    """
    paper_ids = {row.paper_id for row in rows}
    titles: dict[str, tuple[str, str]] = {}
    if paper_ids:
        papers = session.exec(
            select(PaperRow).where(col(PaperRow.id).in_(paper_ids))
        ).all()
        titles = {p.id: (p.title or "", p.arxiv_label or "") for p in papers}
    strengths = strengths_for(session, [row.id for row in rows], now=now)
    out: list[DueCard] = []
    for row in rows:
        title, label = titles.get(row.paper_id, ("", ""))
        out.append(
            DueCard(
                card=card_from_row(row, strengths.get(row.id)),
                paper_title=title,
                paper_label=label,
            )
        )
    return out


# --------------------------------------------------------------------------
# Progress counters
# --------------------------------------------------------------------------
def _tally(condition) -> Any:
    """``SUM(CASE WHEN <condition> THEN 1 ELSE 0 END)`` — one counter, one column."""
    return func.coalesce(func.sum(case((condition, 1), else_=0)), 0)


def deck_progress(
    session: Session,
    paper_id: Optional[str] = None,
    *,
    now: Optional[datetime] = None,
) -> DeckProgress:
    """Counters for one deck, or — with ``paper_id=None`` — the whole library.

    The old state tallies (new / learning / review / mature) are gone with
    SM-2: those were positions in a day-based lifecycle that no longer exists.
    What is left is what the session model can honestly report — how big the
    deck is, how much of it is still to do today, and how much has been
    starred.
    """
    moment = now or utcnow()

    statement = select(
        func.count(),
        _tally(FlashcardRow.starred.is_(True)),
    ).select_from(FlashcardRow)
    if paper_id:
        statement = statement.where(FlashcardRow.paper_id == paper_id)
    row = session.exec(statement).one()
    total, starred_c = (int(value or 0) for value in row)

    seen = seen_today_count(session, paper_id=paper_id, now=moment)
    day_start, _day_end = local_day_bounds_utc(moment)
    return DeckProgress(
        total=total,
        due_count=max(0, total - seen),
        starred_count=starred_c,
        reviews_today=_reviews_since(session, paper_id, day_start),
    )


def _reviews_since(
    session: Session, paper_id: Optional[str], since: datetime
) -> int:
    statement = select(func.count()).select_from(FlashcardReviewRow).where(
        FlashcardReviewRow.reviewed_at >= since
    )
    if paper_id:
        statement = statement.where(FlashcardReviewRow.paper_id == paper_id)
    return int(session.exec(statement).one() or 0)


def reviews_between(
    session: Session,
    start: datetime,
    end: datetime,
    *,
    paper_id: Optional[str] = None,
) -> int:
    """Reviews in ``[start, end)`` — the sparkline's per-day bucket count."""
    statement = select(func.count()).select_from(FlashcardReviewRow).where(
        FlashcardReviewRow.reviewed_at >= start,
        FlashcardReviewRow.reviewed_at < end,
    )
    if paper_id:
        statement = statement.where(FlashcardReviewRow.paper_id == paper_id)
    return int(session.exec(statement).one() or 0)


def deck_generated_at(session: Session, paper_id: str) -> Optional[datetime]:
    """When this deck was last (re)generated — the newest card's birth date."""
    statement = select(func.max(FlashcardRow.created_at)).where(
        FlashcardRow.paper_id == paper_id
    )
    return session.exec(statement).one_or_none() or None


