"""Session scheduler — the implementation of ``describe_scheduler_contract()``.

Replaces the retired SM-2 engine (``study/srs.py``). The contract in
:func:`deepvision.models.study.describe_scheduler_contract` is the spec; this
module implements it verbatim and nothing else.

**Nothing here persists scheduling state.** That is the whole design. A card's
placement inside a sitting is the client's business (an array), and everything
that has to outlive the sitting is *derived* from the append-only
``flashcard_reviews`` log on read. No new table, no new column, no dates.

Every function is pure and takes ``now`` explicitly where it needs one, so the
local-day boundary and the strength fold are both directly testable without a
database.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence

from deepvision.models.study import (
    SESSION_REQUEUE_GAPS,
    Rating,
    rating_recalled,
)

__all__ = [
    "local_day_bounds_utc",
    "requeue_gap",
    "insertion_index",
    "strength_from_ratings",
    "queue_sort_key",
]


def requeue_gap(rating: Rating) -> Optional[int]:
    """How many other cards to put between this card and its return.

    ``None`` means "does not come back this session" — that is ``got_it``, and
    it is the *absence* of an entry in :data:`SESSION_REQUEUE_GAPS` rather than
    a sentinel, so adding a rating without deciding its gap cannot silently
    inherit one.
    """
    return SESSION_REQUEUE_GAPS.get(rating)


def insertion_index(current_index: int, remaining: int, gap: Optional[int]) -> Optional[int]:
    """Where to reinsert a just-rated card, as an index into the live queue.

    ``current_index`` is the position the card was answered at, ``remaining``
    the number of cards still queued *after* it. Returns ``None`` when the card
    retires from the session.

    The clamp is the contract's one edge case: with fewer cards left than the
    gap, the card goes to the very end rather than being dropped. Dropping it
    would mean a card you just failed is never seen again this sitting, which
    is precisely backwards.
    """
    if gap is None:
        return None
    return current_index + min(gap, remaining) + 1


def strength_from_ratings(ratings_newest_first: Sequence[str]) -> int:
    """Consecutive non-``again`` ratings, counting back from the newest.

    ``ratings_newest_first`` must be ordered newest → oldest. A card never
    reviewed (empty) or whose latest answer was ``again`` has strength 0.

    Legacy ``hard``/``good``/``easy`` rows count as recalled — see
    :func:`~deepvision.models.study.rating_recalled`. The review log is
    append-only and predates this scheduler, so the twelve rows already in it
    still mean what they meant; throwing that away would have made every card
    look brand new on the day this shipped.
    """
    streak = 0
    for value in ratings_newest_first:
        if not rating_recalled(value):
            break
        streak += 1
    return streak


def queue_sort_key(
    strength: int, last_reviewed_at: Optional[datetime], card_id: str
) -> tuple[int, float, str]:
    """Order the session queue: weakest first, then longest-unseen, then stable.

    A never-reviewed card sorts as "longest unseen" (``-inf``) within its
    strength band, so new material leads rather than trailing behind cards the
    reader has already failed once.

    The ``card_id`` tail is what makes the order deterministic — without it two
    cards with identical strength and timestamp could swap places between
    requests and the queue would look random.
    """
    stamp = float("-inf") if last_reviewed_at is None else last_reviewed_at.timestamp()
    return (strength, stamp, card_id)


def local_day_bounds_utc(now: Optional[datetime] = None) -> tuple[datetime, datetime]:
    """``[start, end)`` of the machine's LOCAL day, expressed as naive UTC.

    Both halves matter:

    *Local*, because this is a local-first app whose server runs on the
    reader's own machine, and "cards I did today" has to mean their today. A
    UTC boundary rolls over at 4-5pm on the US west coast, which would reset
    the day's count in the middle of an afternoon session.

    *Naive UTC*, because ``reviewed_at`` is written with ``datetime.utcnow()``
    and SQLite has no timezone type — the whole server agrees on one
    convention (see ``parseServerDate`` in ``frontend/src/api/types.ts``). The
    comparison therefore has to happen in that convention, not in local time.
    """
    local_now = (now or datetime.now()).astimezone()
    start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)

    def _to_naive_utc(value: datetime) -> datetime:
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    return _to_naive_utc(start_local), _to_naive_utc(end_local)


def distinct_seen(card_ids: Iterable[str]) -> int:
    """Count distinct cards. A card re-seen after ``again`` counts once."""
    return len(set(card_ids))
