"""SQLModel table definitions — the persistence schema.

SQLite (via SQLModel) stores papers, ingest jobs, reports, chat messages, and the
single settings row. Vectors live in ChromaDB (not here); binary assets (PDFs,
crops, page images) live under ``data/`` (not here). Rich nested structures
(author lists, section trees, log entries, citations) are stored as JSON columns
and validated back into the Pydantic models in :mod:`deepvision.models` at the
service boundary.

Relationships:
    PaperRow 1─* JobRow             (a paper can be (re-)ingested multiple times)
    PaperRow 1─1 ReportRow          (latest report for a paper)
    PaperRow 1─* ChapterReportRow   (one report per (paper, chapter) pair)
    PaperRow 1─* ChatRow            (chat history grounded in a paper)
    SettingsRow                      (singleton, id == 1)

    -- Study layer --
    PaperRow 1─* FlashcardRow       (a paper's deck)
    FlashcardRow 1─* FlashcardReviewRow   (append-only SM-2 audit log)
    PaperRow 1─* QuizRow            (a paper can have several quizzes)
    QuizRow  1─* QuizQuestionRow
    QuizRow  1─* QuizAttemptRow
    QuizAttemptRow 1─* QuizAnswerRow

**ADD-ONLY RULE.** The database is SQLite created by SQLModel ``create_all``,
which creates *new* tables on an existing database but never migrates or
re-shapes an existing one. Adding a table Just Works on next boot; altering an
existing table silently does nothing on every machine that has already run the
app. So: new features get **new tables**, never new columns on an old one.
"""

# NOTE (integration fix): ``from __future__ import annotations`` MUST NOT be used
# in this module. It stringifies the ``Relationship`` annotations
# (``list["JobRow"]`` etc.), and SQLModel then hands the raw string
# ``list['JobRow']`` to SQLAlchemy's ``relationship()``, which raises
# ``InvalidRequestError: ... seems to be using a generic class`` on the first
# ORM query — on every SQLModel/SQLAlchemy version. Every annotation below is
# runtime-valid without future-annotations (Python 3.9+ ``list[...]`` /
# ``dict[...]`` / ``Optional[...]`` / string forward-refs), so removing that
# import is the minimal, behavior-preserving fix. See the integration report.

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column, Index, UniqueConstraint
from sqlalchemy import JSON as SA_JSON
from sqlmodel import Field, Relationship, SQLModel

__all__ = [
    "PaperRow",
    "JobRow",
    "ReportRow",
    "ChapterReportRow",
    "ChatRow",
    "SettingsRow",
    # Study layer
    "FlashcardRow",
    "FlashcardReviewRow",
    "QuizRow",
    "QuizQuestionRow",
    "QuizAttemptRow",
    "QuizAnswerRow",
]


class PaperRow(SQLModel, table=True):
    """One paper in the library. Mirrors :class:`~deepvision.models.PaperMeta`."""

    __tablename__ = "papers"

    id: str = Field(primary_key=True, description="Internal paper id.")
    arxiv_id: str = Field(index=True, description="Bare arXiv id.")
    arxiv_label: str = Field(default="")
    version: Optional[str] = Field(default=None)
    title: str = Field(default="", index=True)
    authors: list[str] = Field(default_factory=list, sa_column=Column(SA_JSON))
    abstract: str = Field(default="")
    categories: list[str] = Field(default_factory=list, sa_column=Column(SA_JSON))
    published: Optional[str] = Field(default=None, description="ISO date string.")
    updated: Optional[str] = Field(default=None, description="ISO date string.")
    pdf_url: Optional[str] = Field(default=None)
    abs_url: Optional[str] = Field(default=None)

    status: str = Field(default="queued", index=True, description="PaperStatus value.")
    ingested: bool = Field(default=False)
    thumbnail_path: Optional[str] = Field(default=None)
    page_count: Optional[int] = Field(default=None)
    figure_count: Optional[int] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    jobs: list["JobRow"] = Relationship(back_populates="paper")
    report: Optional["ReportRow"] = Relationship(back_populates="paper")
    chats: list["ChatRow"] = Relationship(back_populates="paper")


class JobRow(SQLModel, table=True):
    """One ingestion job. Mirrors :class:`~deepvision.models.IngestJob`.

    ``stages`` and ``log`` are stored as JSON arrays of the corresponding model
    dumps.
    """

    __tablename__ = "jobs"

    id: str = Field(primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)
    arxiv_id: str = Field(default="")
    state: str = Field(default="queued", index=True, description="JobState value.")
    current_stage: Optional[str] = Field(default=None, description="JobStage value.")
    percent: int = Field(default=0)
    stages: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(SA_JSON)
    )
    log: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(SA_JSON))
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    paper: Optional[PaperRow] = Relationship(back_populates="jobs")


class ReportRow(SQLModel, table=True):
    """The latest report for a paper. Mirrors :class:`~deepvision.models.Report`.

    ``stats`` and ``sections`` (the full section tree incl. citations and media)
    are stored as JSON.
    """

    __tablename__ = "reports"

    id: str = Field(primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", index=True, unique=True)
    stats: dict[str, Any] = Field(default_factory=dict, sa_column=Column(SA_JSON))
    sections: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(SA_JSON)
    )
    model_used: Optional[str] = Field(default=None)
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    paper: Optional[PaperRow] = Relationship(back_populates="report")


class ChapterReportRow(SQLModel, table=True):
    """One **per-chapter** report. Mirrors a ``Report`` with ``scope='chapter'``.

    **Why a new table instead of extending ``reports``.** ``ReportRow.paper_id``
    is ``unique=True`` — the schema deliberately allows exactly one whole-paper
    report per paper — so chapter reports cannot live there without dropping
    that constraint. The DB is SQLite created by SQLModel ``create_all``, which
    creates *new* tables automatically on an existing database but never
    migrates or re-shapes an existing one: altering ``reports`` would silently
    do nothing on every machine that has already run the app, while adding this
    table Just Works on next boot. Hence: new table, keyed by
    ``(paper_id, chapter_id)``.

    ``stats`` and ``sections`` (the full section tree incl. citations and media)
    are stored as JSON, exactly like :class:`ReportRow`. The ``chapter_*``
    columns denormalize :class:`~deepvision.models.report.Chapter` so a stored
    chapter report can be rendered (title in the header, page range in the
    stats) without re-opening the PDF.

    Uniqueness is on ``(paper_id, chapter_id)``: re-generating a chapter report
    overwrites the existing row rather than appending a second one. ``id`` is a
    normal ``rpt_...`` report id and is what ends up in ``Report.id``.

    No ORM ``Relationship`` is declared back to :class:`PaperRow` on purpose —
    the foreign key gives referential clarity without touching ``PaperRow``'s
    existing mapper configuration. Deleting a paper must delete these rows
    explicitly (see ``ingestion/repo.delete_paper_rows``).
    """

    __tablename__ = "chapter_reports"
    __table_args__ = (
        UniqueConstraint("paper_id", "chapter_id", name="uq_chapter_report_paper_chapter"),
    )

    id: str = Field(primary_key=True, description="Report id (utils.ids.report_id).")
    paper_id: str = Field(foreign_key="papers.id", index=True)
    chapter_id: str = Field(index=True, description="Chapter.id (Chapter.make_id).")

    chapter_title: str = Field(default="", description="Denormalized Chapter.title.")
    chapter_level: int = Field(default=1, description="Denormalized Chapter.level.")
    chapter_page_start: int = Field(default=1, description="Denormalized Chapter.page_start.")
    chapter_page_end: int = Field(default=1, description="Denormalized Chapter.page_end.")
    chapter_source: str = Field(
        default="headings", description="ChapterSource value: 'outline' | 'headings'."
    )

    stats: dict[str, Any] = Field(default_factory=dict, sa_column=Column(SA_JSON))
    sections: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(SA_JSON)
    )
    model_used: Optional[str] = Field(default=None)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ChatRow(SQLModel, table=True):
    """One chat message. Mirrors :class:`~deepvision.models.ChatMessage`.

    ``citations`` and ``figures`` are stored as JSON. ``session_id`` groups a
    conversation thread.
    """

    __tablename__ = "chat_messages"

    id: str = Field(primary_key=True)
    session_id: str = Field(index=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)
    role: str = Field(default="user", description="ChatRole value.")
    text: str = Field(default="")
    citations: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(SA_JSON)
    )
    figures: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(SA_JSON)
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    paper: Optional[PaperRow] = Relationship(back_populates="chats")


# ==========================================================================
# Study layer — flashcards (SM-2) and quizzes
#
# All six tables are NEW (see the ADD-ONLY RULE in the module docstring); no
# existing table is touched. Everything is keyed by ``paper_id`` so deleting a
# paper can clean up with six deletes, but the *reads* that matter are
# cross-paper — hence the composite indexes below.
# ==========================================================================
class FlashcardRow(SQLModel, table=True):
    """One flashcard + its SM-2 scheduling state.

    Mirrors :class:`~deepvision.models.study.Flashcard` with
    The retired SM-2 state flattened into columns (the
    scheduler updates individual fields and the due query filters on
    ``due_at``; a JSON blob would make both awkward).

    **Indexes are load-bearing.** The Study screen's landing query is
    "every card in the library with ``due_at <= now``, soonest first" — across
    *all* papers. ``ix_flashcards_due_at`` makes that an index range scan
    instead of a full table scan of every card the user has ever generated.
    ``ix_flashcards_paper_due`` serves the same query filtered to one paper
    (the deep-link from a Report launch card) and the per-paper due counts.

    ``content_key`` is the upsert key for regeneration: re-generating a deck
    must preserve the SRS state of cards that survive, so the generator matches
    on ``(paper_id, content_key)`` rather than deleting and re-inserting.
    """

    __tablename__ = "flashcards"
    __table_args__ = (
        UniqueConstraint("paper_id", "content_key", name="uq_flashcard_paper_content"),
        Index("ix_flashcards_due_at", "due_at"),
        Index("ix_flashcards_paper_due", "paper_id", "due_at"),
        Index("ix_flashcards_paper_state", "paper_id", "state"),
    )

    id: str = Field(primary_key=True, description="Card id ('card_...').")
    paper_id: str = Field(foreign_key="papers.id", index=True)

    front: str = Field(default="")
    back: str = Field(default="")
    hint: Optional[str] = Field(default=None)
    origin: str = Field(default="fact", description="FlashcardOrigin value.")
    content_key: str = Field(
        default="", description="sha1(paper_id|origin|normalized front)[:16] — upsert key."
    )

    source_section: Optional[str] = Field(default=None, description="SectionName value.")
    source_page: Optional[int] = Field(default=None)
    chunk_id: Optional[str] = Field(default=None)
    tags: list[str] = Field(default_factory=list, sa_column=Column(SA_JSON))

    starred: bool = Field(default=False)

    # --- Vestigial SM-2 columns — NOT read, NOT written ---
    # The scheduler is session-based now (models/study.py::
    # describe_scheduler_contract): no dates, no intervals, and nothing about
    # scheduling is persisted. These eight columns survive only because
    # SQLModel create_all cannot drop a column. New rows take the defaults
    # below and nothing ever reads them. Do not revive them.
    state: str = Field(default="new", description="Vestigial. Not read.")
    ease_factor: float = Field(default=2.5, description="SM-2 EF; floor 1.3.")
    interval_days: int = Field(default=0)
    repetitions: int = Field(default=0)
    lapses: int = Field(default=0)
    learning_step: int = Field(default=0)
    due_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC. Due iff <= now. Indexed — the cross-paper queue reads this.",
    )
    last_reviewed_at: Optional[datetime] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class FlashcardReviewRow(SQLModel, table=True):
    """**Append-only** SM-2 audit log. One row per rating, ever.

    Never UPDATE and never DELETE a row here (except when the owning paper is
    deleted). Two reasons this exists rather than only keeping current state on
    :class:`FlashcardRow`: the schedule is otherwise unauditable — you cannot
    tell *why* a card is due in 33 days — and a future stats view (reviews/day,
    lapse-prone cards) needs history that current state cannot
    reconstruct.

    ``ix_flashcard_reviews_reviewed_at`` serves "reviews today" and the activity
    heatmap without scanning; ``ix_flashcard_reviews_card`` serves a single
    card's history in the card detail drawer.
    """

    __tablename__ = "flashcard_reviews"
    __table_args__ = (
        Index("ix_flashcard_reviews_reviewed_at", "reviewed_at"),
        Index("ix_flashcard_reviews_card", "card_id", "reviewed_at"),
        Index("ix_flashcard_reviews_paper_reviewed", "paper_id", "reviewed_at"),
    )

    id: str = Field(primary_key=True, description="Review id ('rev_...').")
    card_id: str = Field(foreign_key="flashcards.id", index=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)
    rating: str = Field(description="Rating value: again | hard | good | easy.")

    prior_state: str = Field(default="new")
    prior_ease_factor: float = Field(default=2.5)
    prior_interval_days: int = Field(default=0)
    prior_repetitions: int = Field(default=0)

    post_state: str = Field(default="learning")
    post_ease_factor: float = Field(default=2.5)
    post_interval_days: int = Field(default=0)
    post_repetitions: int = Field(default=0)

    was_lapse: bool = Field(default=False)
    elapsed_ms: Optional[int] = Field(default=None)
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)
    due_at_after: datetime = Field(default_factory=datetime.utcnow)


class QuizRow(SQLModel, table=True):
    """One generated quiz for a paper. Questions live in :class:`QuizQuestionRow`.

    A paper may have several quizzes (regenerating produces a new one rather
    than overwriting, so past attempt history stays meaningful) — hence
    ``paper_id`` is indexed but **not** unique.
    """

    __tablename__ = "quizzes"
    __table_args__ = (Index("ix_quizzes_paper_generated", "paper_id", "generated_at"),)

    id: str = Field(primary_key=True, description="Quiz id ('quiz_...').")
    paper_id: str = Field(foreign_key="papers.id", index=True)
    title: str = Field(default="")
    question_count: int = Field(default=0)
    difficulty_mix: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(SA_JSON), description="{Difficulty value: count}."
    )
    kind_mix: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(SA_JSON), description="{QuestionKind value: count}."
    )
    model_used: Optional[str] = Field(default=None)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QuizQuestionRow(SQLModel, table=True):
    """One question — **including the answer and the explanation**.

    This row is the reason grading is server-side. ``correct_option_id``,
    ``correct_answer_text``, ``accepted_answers`` and ``explanation`` must never
    reach the client before the user commits an answer: the pre-answer wire
    shape is ``QuizQuestionPublic``, which has no field to put them in. Build
    that shape explicitly — never ``QuizQuestion.model_dump()`` with an exclude
    list, which silently starts leaking the day someone adds a field.

    ``options`` is a JSON array of ``{"id": ..., "text": ...}`` in display
    order; option order is fixed at generation time (so a reload does not
    reshuffle the answer out from under the user).
    """

    __tablename__ = "quiz_questions"
    __table_args__ = (Index("ix_quiz_questions_quiz_index", "quiz_id", "index"),)

    id: str = Field(primary_key=True, description="Question id ('qq_...').")
    quiz_id: str = Field(foreign_key="quizzes.id", index=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)
    index: int = Field(default=0, description="0-based position in the quiz.")

    prompt: str = Field(default="")
    kind: str = Field(default="multiple_choice", description="QuestionKind value.")
    difficulty: str = Field(default="recall", description="Difficulty value.")
    options: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(SA_JSON), description="[{id, text}, ...]."
    )

    correct_option_id: Optional[str] = Field(default=None, description="SERVER-ONLY.")
    correct_answer_text: Optional[str] = Field(default=None, description="SERVER-ONLY.")
    accepted_answers: list[str] = Field(
        default_factory=list, sa_column=Column(SA_JSON), description="SERVER-ONLY."
    )
    explanation: str = Field(default="", description="SERVER-ONLY until graded.")

    citation_section: Optional[str] = Field(default=None, description="SectionName value.")
    citation_page: Optional[int] = Field(default=None)
    citation_chunk_id: Optional[str] = Field(default=None)
    citation_snippet: str = Field(default="")

    created_at: datetime = Field(default_factory=datetime.utcnow)


class QuizAttemptRow(SQLModel, table=True):
    """One graded run through a quiz, or through a subset of it.

    ``question_ids`` is the JSON list of questions actually in scope, in quiz
    order. For ``mode == 'retry_missed'`` that is the misses of
    ``retry_of_attempt_id`` — this is how "retry only what I missed" is modelled:
    an ordinary attempt whose scope is a question-id subset, not a second quiz.
    ``score`` is therefore relative to ``total`` (= ``len(question_ids)``) and is
    not comparable across modes.
    """

    __tablename__ = "quiz_attempts"
    __table_args__ = (
        Index("ix_quiz_attempts_quiz_finished", "quiz_id", "finished_at"),
        Index("ix_quiz_attempts_paper_finished", "paper_id", "finished_at"),
    )

    id: str = Field(primary_key=True, description="Attempt id ('qa_...').")
    quiz_id: str = Field(foreign_key="quizzes.id", index=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)

    mode: str = Field(default="full", description="AttemptMode: full | retry_missed.")
    retry_of_attempt_id: Optional[str] = Field(default=None)
    question_ids: list[str] = Field(default_factory=list, sa_column=Column(SA_JSON))
    missed_question_ids: list[str] = Field(default_factory=list, sa_column=Column(SA_JSON))

    total: int = Field(default=0)
    correct: int = Field(default=0)
    score: float = Field(default=0.0, description="correct / total, in [0, 1].")

    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime = Field(default_factory=datetime.utcnow)


class QuizAnswerRow(SQLModel, table=True):
    """One answer inside an attempt, with the server's verdict.

    ``is_correct`` is written by the grader, never accepted from the client.
    ``correct_option_id`` is snapshotted here so the attempt review screen still
    renders correctly if the quiz is later regenerated.
    """

    __tablename__ = "quiz_answers"
    __table_args__ = (
        UniqueConstraint("attempt_id", "question_id", name="uq_quiz_answer_attempt_question"),
        Index("ix_quiz_answers_attempt", "attempt_id"),
    )

    id: str = Field(primary_key=True, description="Answer id ('qans_...').")
    attempt_id: str = Field(foreign_key="quiz_attempts.id", index=True)
    question_id: str = Field(foreign_key="quiz_questions.id", index=True)
    quiz_id: str = Field(index=True)

    selected_option_id: Optional[str] = Field(default=None)
    answer_text: Optional[str] = Field(default=None)
    is_correct: bool = Field(default=False, description="Server verdict. Authoritative.")
    correct_option_id: Optional[str] = Field(
        default=None, description="Snapshot of the right answer at grading time."
    )
    elapsed_ms: Optional[int] = Field(default=None)
    answered_at: datetime = Field(default_factory=datetime.utcnow)


class SettingsRow(SQLModel, table=True):
    """Singleton settings row (id == 1). Mirrors :class:`~deepvision.models.AppSettings`.

    The whole settings object is stored as one JSON blob so the shape can evolve
    with the model without migrations.
    """

    __tablename__ = "settings"

    id: int = Field(default=1, primary_key=True)
    data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(SA_JSON))
    updated_at: datetime = Field(default_factory=datetime.utcnow)
