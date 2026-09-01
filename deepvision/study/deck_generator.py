"""Deck generation — build a paper's flashcards and persist them idempotently.

Three responsibilities, all about *not losing the user's work*:

1. **Source the deck from the persisted report, plus a whole-paper pool.** The
   report has already been through the agent ensemble, so its Key Concepts
   (which now carries the vocabulary the old Glossary section held), Key
   Takeaways, Key Results and Methods are cleaned, cited and page-anchored —
   everything a card needs for provenance is already there, and it costs one
   small LLM polish pass instead of a second full retrieval run. But the
   report is a *summary*: measured on this project's own library it is
   30-37% of a paper's chars, and it is a fixed artifact, so it alone cannot
   supply the variety a re-generated deck needs. ``generate_deck`` also sweeps
   the paper with :func:`deepvision.study.source_pool.pool_for_paper` (needs a
   ``retriever``; ``None`` degrades to report-only, same as before this
   change) and hands that pool to :func:`~deepvision.agents.flashcard_agent.
   build_deck` as ``extra_chunks``.

2. **Upsert on ``(paper_id, content_key)``.** Regeneration must not reset
   anybody's schedule. A card whose normalized front hashes to a key already in
   the deck has its *text* refreshed and its SM-2 state left strictly alone; a
   card the user wrote themselves (``origin == 'manual'``) is never touched at
   all. The persisted deck's own content keys are read back and passed to
   ``build_deck`` as ``avoid_keys`` so a regeneration deprioritizes cards it
   has already produced and surfaces new ones — the target count caps how many
   *new* cards one generation writes, not the deck's total size, so
   regeneration **grows** the deck instead of reshuffling the same one.

3. Cards that a generation did **not** produce are deliberately kept rather
   than deleted. Deleting them would orphan their rows in the append-only
   review log and — much worse — would silently destroy the schedule of a
   card the user has been reviewing for three weeks, just because the report
   got re-run and the summariser phrased a definition differently. Deck reset
   and paper deletion are the intentional removal paths; regeneration is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from sqlmodel import func, select

from deepvision.agents.flashcard_agent import build_deck
from deepvision.db import session_scope
from deepvision.db.schema import FlashcardRow, PaperRow, ReportRow
from deepvision.models.report import Report
from deepvision.models.settings import AppSettings
from deepvision.models.study import Flashcard, FlashcardOrigin
from deepvision.study.card_queries import utcnow
from deepvision.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from deepvision.models.chunks import AnyChunk
    from deepvision.rag.retrieval import Retriever

__all__ = [
    "DeckResult",
    "NoReportError",
    "card_count",
    "generate_deck",
    "has_report",
    "persist_deck",
]

log = get_logger(__name__)

ProgressFn = Callable[[int, str], None]


class NoReportError(LookupError):
    """Raised when a deck is requested for a paper that has no report yet.

    Not an internal failure: the deck is *made of* the report, so this is a
    precondition the user can fix in one click (open the Report tab). The route
    turns it into a 409 with that message rather than a 500.
    """


@dataclass
class DeckResult:
    """What one generation run did."""

    paper_id: str
    created: int = 0
    updated: int = 0
    total: int = 0
    generated: int = 0

    @property
    def summary(self) -> str:
        return (
            f"{self.created} new card(s), {self.updated} refreshed, "
            f"{self.total} in the deck"
        )


def card_count(paper_id: str) -> int:
    """How many cards this paper's deck currently holds."""
    with session_scope() as session:
        statement = (
            select(func.count())
            .select_from(FlashcardRow)
            .where(FlashcardRow.paper_id == paper_id)
        )
        return int(session.exec(statement).one() or 0)


def has_report(paper_id: str) -> bool:
    """Whether a persisted report exists to build a deck from."""
    with session_scope() as session:
        row = session.exec(
            select(ReportRow).where(ReportRow.paper_id == paper_id)
        ).first()
        return row is not None


def _load_report(paper_id: str) -> Report:
    from deepvision.report.report_generator import DefaultReportGenerator

    try:
        return DefaultReportGenerator().load(paper_id)
    except LookupError as exc:
        raise NoReportError(
            f"no report for paper {paper_id!r} — open the Report tab to generate "
            "one first; flashcards are built from it"
        ) from exc


def _paper_context(paper_id: str) -> tuple[str, str]:
    with session_scope() as session:
        row = session.get(PaperRow, paper_id)
        if row is None:
            return "", ""
        return row.title or "", row.abstract or ""


def generate_deck(
    paper_id: str,
    settings: AppSettings,
    *,
    target_count: int = 20,
    on_progress: Optional[ProgressFn] = None,
    retriever: Optional["Retriever"] = None,
) -> DeckResult:
    """Build and persist ``paper_id``'s deck. The job worker's whole body.

    ``retriever`` (typically ``report.agent_bridge.build_retriever(settings)``,
    built by the caller so a construction failure degrades outside this
    function) is used to sweep the whole paper for material beyond the
    report; ``None`` degrades to the old report-only behaviour, never an
    error. ``target_count`` caps how many *new* cards this run writes, not the
    deck's total size — cards from a previous generation are read back and
    passed to :func:`~deepvision.agents.flashcard_agent.build_deck` as
    ``avoid_keys``, so a regeneration grows the deck instead of reshuffling it.

    Raises :class:`NoReportError` only — every other failure (a dead model, an
    unparseable section, a failed pool sweep) degrades inside
    :func:`~deepvision.agents.flashcard_agent.build_deck` to a smaller
    deterministic deck.
    """

    def progress(percent: int, message: str) -> None:
        if on_progress is not None:
            on_progress(percent, message)

    progress(10, "Reading the paper's report")
    report = _load_report(paper_id)
    title, abstract = _paper_context(paper_id)

    progress(20, "Sweeping the paper for material beyond the report")
    extra_chunks = _pool_chunks(paper_id, retriever)
    avoid_keys = _existing_content_keys(paper_id)

    progress(25, "Selecting terms, concepts and paper-specific facts")
    llm = _build_llm(settings)

    progress(45, f"Writing up to {target_count} cards")
    cards = build_deck(
        report,
        paper_id=paper_id,
        target_count=target_count,
        llm=llm,
        paper_title=title,
        abstract=abstract,
        extra_chunks=extra_chunks,
        avoid_keys=avoid_keys,
    )

    progress(85, "Saving the deck (existing cards keep their schedule)")
    result = persist_deck(paper_id, cards)
    # NOTE: the structured-context keys must not collide with reserved
    # ``logging.LogRecord`` attributes — a bare ``created=`` here raises
    # "Attempt to overwrite 'created' in LogRecord", which inside the job worker
    # means a deck that generated perfectly still reports as a failed job.
    log.info(
        "flashcard deck generated",
        extra={
            "paper_id": paper_id,
            "cards_created": result.created,
            "cards_updated": result.updated,
            "cards_total": result.total,
        },
    )
    return result


def _pool_chunks(paper_id: str, retriever: Optional["Retriever"]) -> list["AnyChunk"]:
    """The whole-paper pool, or ``[]`` on any failure or a missing retriever.

    ``pool_for_paper`` already never raises and already degrades a ``None``
    retriever to ``[]``; this wrapper exists so a defect introduced there
    later still cannot take deck generation down with it — the contract this
    whole module lives by.
    """
    try:
        from deepvision.study.source_pool import pool_for_paper

        return pool_for_paper(paper_id, retriever)
    except Exception:  # noqa: BLE001 - the pool is a bonus, never a blocker
        log.warning(
            "could not build a source pool for flashcards; using the report alone",
            extra={"paper_id": paper_id},
            exc_info=True,
        )
        return []


def _existing_content_keys(paper_id: str) -> frozenset[str]:
    """Content keys already in this paper's persisted deck.

    Passed to :func:`~deepvision.agents.flashcard_agent.build_deck` as
    ``avoid_keys`` so a regeneration deprioritizes cards it has already
    produced and surfaces new material instead of reshuffling the same deck.
    """
    with session_scope() as session:
        keys = session.exec(
            select(FlashcardRow.content_key).where(FlashcardRow.paper_id == paper_id)
        ).all()
    return frozenset(key for key in keys if key)


def _build_llm(settings: AppSettings):
    """Resolve the LLM through the factory, degrading to no polish on failure.

    A provider that cannot even be *constructed* (missing binary, bad config)
    must not fail deck generation — ``build_deck(llm=None)`` simply ships the
    deterministic draft.
    """
    try:
        from deepvision.config import get_config
        from deepvision.providers.factory import build_llm

        return build_llm(settings, strict=get_config().strict_providers)
    except Exception:  # noqa: BLE001 - provider setup must never break generation
        log.warning(
            "could not build an LLM for flashcard polish; using drafts",
            exc_info=True,
        )
        return None


def persist_deck(paper_id: str, cards: Sequence[Flashcard]) -> DeckResult:
    """Upsert ``cards`` into the ``flashcards`` table on ``(paper_id, content_key)``.

    The whole point is what this function does **not** write: for a card that
    already exists, only the content columns move. ``state``, ``ease_factor``,
    ``interval_days``, ``repetitions``, ``lapses``, ``learning_step``,
    ``due_at``, ``last_reviewed_at`` and ``starred`` are left exactly as the
    scheduler left them.
    """
    now = utcnow()
    result = DeckResult(paper_id=paper_id, generated=len(cards))
    if not cards:
        result.total = card_count(paper_id)
        return result

    with session_scope() as session:
        existing = {
            row.content_key: row
            for row in session.exec(
                select(FlashcardRow).where(FlashcardRow.paper_id == paper_id)
            ).all()
            if row.content_key
        }
        seen: set[str] = set()
        for card in cards:
            key = card.content_key
            if not key or key in seen:
                continue
            seen.add(key)
            row = existing.get(key)
            if row is None:
                session.add(_row_from_card(card, now))
                result.created += 1
                continue
            if row.origin == FlashcardOrigin.MANUAL.value:
                # The user wrote this one. Regeneration has no business
                # rewriting it, even if the hash happens to collide.
                continue
            _refresh_content(row, card, now)
            result.updated += 1
        session.flush()

    result.total = card_count(paper_id)
    return result


def _row_from_card(card: Flashcard, now: datetime) -> FlashcardRow:
    return FlashcardRow(
        id=card.id,
        paper_id=card.paper_id,
        front=card.front,
        back=card.back,
        hint=card.hint,
        origin=card.origin.value,
        content_key=card.content_key,
        source_section=card.source_section.value if card.source_section else None,
        source_page=card.source_page,
        chunk_id=card.chunk_id,
        tags=list(card.tags),
        starred=card.starred,
        # The SM-2 columns (state/ease_factor/interval_days/repetitions/lapses/
        # learning_step/due_at/last_reviewed_at) are deliberately NOT written.
        # They are vestigial after the move to session scheduling and only
        # still exist because SQLModel create_all cannot drop a column; the
        # schema defaults fill them and nothing reads them.
        created_at=now,
        updated_at=now,
    )


def _refresh_content(row: FlashcardRow, card: Flashcard, now: datetime) -> None:
    """Content columns only — the SM-2 columns are untouched on purpose."""
    row.front = card.front
    row.back = card.back
    row.hint = card.hint
    row.source_section = card.source_section.value if card.source_section else None
    row.source_page = card.source_page
    row.chunk_id = card.chunk_id
    row.tags = list(card.tags)
    row.updated_at = now
