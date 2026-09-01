"""HTTP API request/response schemas — the wire contract.

Every endpoint's request and response shape is defined here in Pydantic v2 so the
frontend team and the six builder domains share exactly one definition. Domain
models from :mod:`deepvision.models` are reused directly in responses where the
wire shape equals the domain shape (Report, IngestJob, CompareResult, etc.);
thin request/list wrappers are defined here.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from deepvision.models.chat import ChatRequest, ChatResponse
from deepvision.models.job import IngestJob
from deepvision.models.paper import PaperMeta
from deepvision.models.report import Chapter, ChapterSource, Report, ReportScope
from deepvision.models.settings import (
    DETAIL_LEVEL_INFO,
    AppSettings,
    ProviderMode,
    ReportDetail,
)
from deepvision.models.study import (
    AttemptMode,
    CardStrength,
    DeckProgress,
    Difficulty,
    DueCard,
    Flashcard,
    GradedAnswer,
    FlashcardOrigin,
    PaperStudyStats,
    QuestionKind,
    QuizAttempt,
    QuizAttemptSummary,
    QuizPublic,
    QuizSummary,
    Rating,
    StudyItemKind,
    SubmittedAnswer,
)

__all__ = [
    # health
    "HealthResponse",
    # search
    "SearchSort",
    "SearchDateRange",
    "SearchResultItem",
    "SearchResponse",
    # ingest
    "IngestRequest",
    "IngestResponse",
    # jobs
    "JobStatusResponse",
    # papers
    "PapersListResponse",
    "DeletePaperResponse",
    "RerunResponse",
    # report
    # (Report reused directly)
    # chapters / per-chapter reports
    "ChaptersResponse",
    "ChapterReportRequest",
    "ChapterReportResponse",
    # chat / compare
    # media
    "MediaMetadata",
    # settings
    "ModelChoice",
    "ModelCatalog",
    "MODEL_CATALOG",
    "DetailLevelInfo",
    "DETAIL_LEVELS",
    "SettingsResponse",
    "UpdateSettingsRequest",
    # study — cross-paper
    "StudyOverviewResponse",
    "DueQueueResponse",
    "StudyPapersResponse",
    # study — flashcards
    "FlashcardGenerateRequest",
    "FlashcardGenerateResponse",
    "FlashcardDeckResponse",
    "FlashcardReviewRequest",
    "FlashcardReviewResponse",
    "FlashcardStarRequest",
    "FlashcardStarResponse",
    "FlashcardResetResponse",
    # study — quiz
    "QuizGenerateRequest",
    "QuizGenerateResponse",
    "QuizListResponse",
    "QuizAttemptRequest",
    "QuizCheckRequest",
    "QuizCheckResponse",
    "QuizRetryRequest",
    "QuizAttemptHistoryResponse",
    # errors
    "ErrorResponse",
    # re-exports for convenience
    "ChatRequest",
    "ChatResponse",
    "Report",
    "ReportScope",
    "Chapter",
    "ChapterSource",
    "IngestJob",
    "AppSettings",
    "ProviderMode",
    "ReportDetail",
    "PaperMeta",
    # study model re-exports (import from here OR from deepvision.models.study)
    "Rating",
    "CardStrength",
    "FlashcardOrigin",
    "Flashcard",
    "DueCard",
    "DeckProgress",
    "QuestionKind",
    "Difficulty",
    "AttemptMode",
    "QuizPublic",
    "QuizSummary",
    "QuizAttempt",
    "QuizAttemptSummary",
    "SubmittedAnswer",
    "GradedAnswer",
    "StudyItemKind",
    "PaperStudyStats",
]


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """``GET /health`` — liveness and backend status."""

    status: str = Field(default="ok", examples=["ok"])
    version: str = Field(..., description="Package version.")
    llm_online: bool = Field(default=False, description="LLM backend reachable.")
    vision_available: bool = Field(
        default=False, description="Local/API vision model available."
    )
    db_ok: bool = Field(default=True, description="Database reachable.")


# --------------------------------------------------------------------------
# Search  (GET /search)
# --------------------------------------------------------------------------
class SearchSort(str, Enum):
    """Sort options for arXiv search (matches the UI dropdown)."""

    RELEVANCE = "relevance"
    NEWEST = "newest"
    MOST_CITED = "most_cited"


class SearchDateRange(str, Enum):
    """Date-range filter options (matches the UI dropdown)."""

    ANY = "any"
    PAST_MONTH = "past_month"
    PAST_6_MONTHS = "past_6_months"
    PAST_YEAR = "past_year"
    RANGE_2024_2025 = "2024_2025"


class SearchResultItem(BaseModel):
    """One arXiv search hit, shaped for the search-result card.

    Reuses :class:`PaperMeta` for bibliographic fields and adds the
    ``ingested`` / ``in_library`` flags the card needs to decide between the
    'Ingest' and 'Open report' buttons.
    """

    paper: PaperMeta = Field(..., description="Bibliographic metadata.")
    ingested: bool = Field(
        default=False, description="True if already ingested (report available)."
    )
    in_library: bool = Field(
        default=False, description="True if present in the local library at all."
    )


class SearchResponse(BaseModel):
    """``GET /search`` response."""

    query: str = Field(..., description="Echoed query string.")
    results: list[SearchResultItem] = Field(default_factory=list)
    count: int = Field(default=0, ge=0, description="Number of results returned.")


# --------------------------------------------------------------------------
# Ingest  (POST /ingest)
# --------------------------------------------------------------------------
class IngestRequest(BaseModel):
    """``POST /ingest`` request.

    ``settings_override`` optionally overrides the persisted :class:`AppSettings`
    for this single ingestion run (e.g. a different provider or chunk size).
    """

    arxiv_id: str = Field(..., description="arXiv id to ingest.", examples=["2411.10842"])
    settings_override: Optional[AppSettings] = Field(
        default=None, description="Per-run settings override (optional)."
    )


class IngestResponse(BaseModel):
    """``POST /ingest`` response — the created job to poll.

    Also the response of ``POST /ingest/upload`` (the multipart PDF-upload
    route), deliberately: the frontend then reuses its existing ``pollJob`` /
    ``IngestionModal`` flow unchanged for an uploaded PDF. There is no
    upload-specific *request* schema — the route takes ``multipart/form-data``
    (``file``: the PDF; optional ``title``), not a JSON body.
    """

    job: IngestJob = Field(..., description="The queued ingestion job.")
    paper_id: str = Field(..., description="Internal paper id being ingested.")


# --------------------------------------------------------------------------
# Jobs  (GET /jobs/{id})
# --------------------------------------------------------------------------
class JobStatusResponse(BaseModel):
    """``GET /jobs/{id}`` response — full job state for the ingestion modal."""

    job: IngestJob = Field(..., description="Current job snapshot.")


# --------------------------------------------------------------------------
# Papers  (GET /papers, DELETE /papers/{id}, POST /papers/{id}/rerun)
# --------------------------------------------------------------------------
class PapersListResponse(BaseModel):
    """``GET /papers`` response — the library grid."""

    papers: list[PaperMeta] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)


class DeletePaperResponse(BaseModel):
    """``DELETE /papers/{id}`` response."""

    paper_id: str
    deleted: bool = Field(default=True)


class RerunResponse(BaseModel):
    """``POST /papers/{id}/rerun`` response — a new ingestion job."""

    job: IngestJob = Field(..., description="The re-run ingestion job.")
    paper_id: str


# --------------------------------------------------------------------------
# Chapters & per-chapter reports
#   GET  /papers/{paper_id}/chapters
#   POST /report/{paper_id}/chapter
#   GET  /report/{paper_id}/chapter/{chapter_id}
# --------------------------------------------------------------------------
class ChaptersResponse(BaseModel):
    """``GET /papers/{paper_id}/chapters`` — what the chapter picker renders.

    **Cheap by contract**: this endpoint opens the PDF and reads its outline (or
    re-runs the ingestion heading heuristic). It makes **no model calls** and
    must stay fast enough to serve a dropdown on page load.

    The user picks from this list — they never type or search a chapter name —
    so the list must be complete, ordered, and self-describing: render
    ``title`` indented by ``level``, with ``pages {page_start}–{page_end}`` as
    the secondary line.

    ``detected`` is ``false`` (and ``chapters`` empty) when the PDF has neither
    an outline nor recognisable headings, or when fewer than two chapters were
    found — a one-chapter "list" is just the whole paper, which the plain
    report already covers. The UI then disables the picker rather than showing
    a single useless option.
    """

    paper_id: str = Field(..., description="Paper the chapters belong to.")
    chapters: list[Chapter] = Field(
        default_factory=list, description="Chapters in page order (index ascending)."
    )
    source: Optional[ChapterSource] = Field(
        default=None,
        description="How the list was derived; null when nothing was detected.",
    )
    page_count: int = Field(
        default=0, ge=0, description="Total pages in the PDF (last chapter's page_end)."
    )
    detected: bool = Field(
        default=False, description="False when no usable chapter structure was found."
    )


class ChapterReportRequest(BaseModel):
    """``POST /report/{paper_id}/chapter`` request.

    ``chapter_id`` is a :attr:`Chapter.id` from ``GET /papers/{id}/chapters`` —
    the frontend always sends back an id it was given, never a free-text title.
    Unknown ids are ``404`` (the PDF may have been re-ingested and re-derived).

    ``force`` re-generates even when a chapter report is already persisted;
    without it, an existing report is returned immediately (``cached: true``)
    and no LLM work is done.
    """

    chapter_id: str = Field(
        ..., description="Chapter.id chosen in the picker.", examples=["ch-3f1a9c2b7d04"]
    )
    force: bool = Field(
        default=False, description="Re-generate even if a chapter report already exists."
    )
    settings_override: Optional[AppSettings] = Field(
        default=None, description="Per-run settings override (optional), as for /ingest."
    )


class ChapterReportResponse(BaseModel):
    """``POST /report/{paper_id}/chapter`` response — cached report, or a job.

    Chapter-report generation reuses the report agent ensemble (Research →
    Summarizer → Media → Study → Synthesis), i.e. real LLM calls: minutes on a
    local model. It therefore **never blocks the HTTP request**. Instead it
    reuses the existing async machinery unchanged:
    :class:`~deepvision.models.job.IngestJob` as the progress envelope,
    persisted to the ``jobs`` table and polled through the existing
    ``GET /jobs/{job_id}`` + frontend ``pollJob``.

    Exactly one of ``report`` / ``job`` is non-null:

    - **200, ``cached: true``** — a report for this chapter is already
      persisted (and ``force`` was not set): ``report`` is it, ``job`` is null.
      Render it immediately.
    - **202, ``cached: false``** — generation was queued: ``job`` is the fresh
      job (already persisted, so ``GET /jobs/{id}`` resolves at once) and
      ``report`` is null. Poll the job, then ``GET
      /report/{paper_id}/chapter/{chapter_id}``.

    A chapter job carries ``stages: []`` and ``current_stage: null`` — the seven
    ``JobStage`` values describe PDF ingestion and mean nothing here (an empty
    list is also what keeps a stage added to ``STAGE_ORDER`` from rendering as a
    phantom row that never completes). Progress
    is conveyed by ``percent`` plus ``log`` lines, which is all the Report
    page's inline progress row needs (it does not use the IngestionModal).
    """

    paper_id: str = Field(..., description="Paper the chapter belongs to.")
    chapter_id: str = Field(..., description="Chapter being reported on.")
    cached: bool = Field(
        default=False, description="True when an existing report is returned as-is."
    )
    job: Optional[IngestJob] = Field(
        default=None, description="Queued generation job to poll; null when cached."
    )
    report: Optional[Report] = Field(
        default=None, description="The persisted chapter report; null when queued."
    )


# --------------------------------------------------------------------------
# Compare  (POST /compare)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Media  (GET /media/{chunk_id})
# --------------------------------------------------------------------------
class MediaMetadata(BaseModel):
    """Metadata sidecar for a media chunk.

    ``GET /media/{chunk_id}`` returns raw image bytes by default; passing
    ``?meta=1`` returns this JSON instead (used by the frontend to render
    caption/provenance without downloading the image).
    """

    chunk_id: str
    paper_id: str
    kind: str = Field(default="figure", description="'figure' | 'table' | 'page'.")
    label: Optional[str] = Field(default=None, description="e.g. 'Fig 2'.")
    caption: Optional[str] = Field(default=None)
    provenance: Optional[str] = Field(
        default=None, description="Provenance value: 'vision' | 'ocr' | 'text'."
    )
    page: Optional[int] = Field(default=None, ge=1)
    content_type: str = Field(default="image/png")
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    url: str = Field(..., description="Byte-serving URL for this media chunk.")


# --------------------------------------------------------------------------
# Settings  (GET /settings, PUT /settings)
# --------------------------------------------------------------------------
class ModelChoice(BaseModel):
    """One selectable model in the Settings screen's model dropdown.

    The catalog is **published by the backend** rather than hard-coded in the
    UI, so adding or repricing a model is a one-line change here. Picking a
    choice writes ``id`` into ``AppSettings.<stage>_model`` *and* ``mode`` into
    ``AppSettings.<stage>_mode`` — the two always move together, which is what
    stops "API" from silently meaning one hard-wired vendor model.
    """

    id: str = Field(
        ...,
        description="Value written to AppSettings.<stage>_model.",
        examples=["claude-sonnet-5"],
    )
    label: str = Field(
        ..., description="Human label for the dropdown.", examples=["Claude Sonnet 5"]
    )
    mode: ProviderMode = Field(
        ...,
        description="Selecting this choice also sets AppSettings.<stage>_mode to this.",
    )
    vendor: str = Field(
        ...,
        description="'local' | 'anthropic' | 'openai' — which key/backend it needs.",
        examples=["anthropic"],
    )
    note: str = Field(
        default="",
        description="Short cost/speed hint shown under the option.",
        examples=["$3 / $15 per Mtok. Best balance - recommended."],
    )
    needs_key: bool = Field(
        ..., description="True for api entries (an API key must be set for this vendor)."
    )


class ModelCatalog(BaseModel):
    """The full set of selectable models, grouped by provider stage.

    Served on :attr:`SettingsResponse.model_catalog`; the constant instance is
    :data:`MODEL_CATALOG`.
    """

    llm: list[ModelChoice] = Field(default_factory=list, description="LLM stage choices.")
    vision: list[ModelChoice] = Field(
        default_factory=list, description="Vision stage choices."
    )
    embedding: list[ModelChoice] = Field(
        default_factory=list, description="Embedding stage choices."
    )


#: The canonical, backend-published model catalog. The settings router returns
#: this verbatim on ``GET /settings`` (and on ``PUT``); nothing else constructs
#: a catalog. Keep ``id`` values exactly as the underlying provider expects them
#: — they are passed straight through to the adapter.
#:
#: NOTE: Anthropic has **no embeddings API**, so the embedding stage offers only
#: the local model and OpenAI. Do not add an anthropic embedding entry.
MODEL_CATALOG = ModelCatalog(
    llm=[
        ModelChoice(
            id="llama3.1:8b",
            label="Local (free) - Llama-3.1-8B",
            mode=ProviderMode.LOCAL,
            vendor="local",
            note="Free, runs on your machine. Slow (minutes per report).",
            needs_key=False,
        ),
        ModelChoice(
            id="claude-sonnet-5",
            label="Claude Sonnet 5",
            mode=ProviderMode.API,
            vendor="anthropic",
            note="$3 / $15 per Mtok. Best balance - recommended.",
            needs_key=True,
        ),
        ModelChoice(
            id="claude-opus-5",
            label="Claude Opus 5",
            mode=ProviderMode.API,
            vendor="anthropic",
            note="$5 / $25 per Mtok. Highest quality.",
            needs_key=True,
        ),
        ModelChoice(
            id="claude-haiku-4-5",
            label="Claude Haiku 4.5",
            mode=ProviderMode.API,
            vendor="anthropic",
            note="$1 / $5 per Mtok. Fastest and cheapest.",
            needs_key=True,
        ),
        ModelChoice(
            id="gpt-4o-mini",
            label="GPT-4o mini",
            mode=ProviderMode.API,
            vendor="openai",
            note="OpenAI. Requires an OpenAI key.",
            needs_key=True,
        ),
    ],
    vision=[
        ModelChoice(
            id="Qwen/Qwen2-VL-7B-Instruct",
            label="Local (free) - Qwen2-VL-7B",
            mode=ProviderMode.LOCAL,
            vendor="local",
            note="Free, runs on your machine.",
            needs_key=False,
        ),
        ModelChoice(
            id="claude-sonnet-5",
            label="Claude Sonnet 5 (vision)",
            mode=ProviderMode.API,
            vendor="anthropic",
            note="$3 / $15 per Mtok. Recommended for figures.",
            needs_key=True,
        ),
        ModelChoice(
            id="claude-opus-5",
            label="Claude Opus 5 (vision)",
            mode=ProviderMode.API,
            vendor="anthropic",
            note="$5 / $25 per Mtok. Highest quality.",
            needs_key=True,
        ),
        ModelChoice(
            id="gpt-4o",
            label="GPT-4o (vision)",
            mode=ProviderMode.API,
            vendor="openai",
            note="OpenAI. Requires an OpenAI key.",
            needs_key=True,
        ),
    ],
    embedding=[
        ModelChoice(
            id="BAAI/bge-large-en-v1.5",
            label="Local (free) - bge-large",
            mode=ProviderMode.LOCAL,
            vendor="local",
            note="Free, runs on your machine.",
            needs_key=False,
        ),
        ModelChoice(
            id="text-embedding-3-large",
            label="OpenAI text-embedding-3-large",
            mode=ProviderMode.API,
            vendor="openai",
            note="OpenAI. Requires an OpenAI key.",
            needs_key=True,
        ),
    ],
)


class DetailLevelInfo(BaseModel):
    """One report-detail option, with the length contract it promises.

    Lets the Settings screen render the promise ("~2 paragraphs per section")
    next to the dropdown without duplicating it in TS — the source of truth is
    :data:`deepvision.models.settings.DETAIL_LEVEL_INFO`, which the summarizer
    reads too, so the UI's promise and the generator's target cannot drift.
    """

    value: ReportDetail = Field(
        ..., description="The wire value stored in AppSettings.report_detail."
    )
    label: str = Field(..., description="Human label.", examples=["Standard"])
    blurb: str = Field(
        ...,
        description="What this level actually produces.",
        examples=["~2 paragraphs per section (6-9 sentences)"],
    )


#: The detail levels in display order, derived from the single source of truth
#: in ``deepvision.models.settings``. Returned on ``SettingsResponse``.
DETAIL_LEVELS: list[DetailLevelInfo] = [
    DetailLevelInfo(value=level, label=info["label"], blurb=info["blurb"])
    for level, info in DETAIL_LEVEL_INFO.items()
]


class SettingsResponse(BaseModel):
    """``GET /settings`` response.

    ``needs_keys`` and ``vision_available`` are derived, so the frontend does not
    recompute them. API key *values* are never returned (only whether each is
    set) — see ``keys_present``.

    ``model_catalog`` and ``detail_levels`` are static reference data the UI
    renders its dropdowns from; the router should return :data:`MODEL_CATALOG`
    and :data:`DETAIL_LEVELS` unchanged.
    """

    settings: AppSettings = Field(..., description="Current settings (keys redacted).")
    needs_keys: bool = Field(default=False, description="Any stage set to 'api'.")
    keys_present: dict[str, bool] = Field(
        default_factory=dict,
        description="Which api keys are set: {'llm': bool, 'vision': bool, 'embedding': bool}.",
    )
    vision_available: bool = Field(
        default=False, description="Local/API vision model availability status."
    )
    vision_status_text: str = Field(
        default="", description="Human status line for the vision status pill."
    )
    model_catalog: ModelCatalog = Field(
        # Deep copy so the shared module-level constant can never be mutated
        # through a response instance.
        default_factory=lambda: MODEL_CATALOG.model_copy(deep=True),
        description="Selectable models per stage (return MODEL_CATALOG).",
    )
    detail_levels: list[DetailLevelInfo] = Field(
        default_factory=lambda: list(DETAIL_LEVELS),
        description="Report-detail options with their length blurbs (return DETAIL_LEVELS).",
    )


class UpdateSettingsRequest(BaseModel):
    """``PUT /settings`` request — the full settings object to persist."""

    settings: AppSettings = Field(..., description="New settings to save.")


# ==========================================================================
# Study — the interactive study layer (its own top-level screen)
#
# Three routers, one per builder, no shared files:
#   study.py       (prefix /study)      cross-paper overview, due queue, stats
#   flashcards.py  (prefix /flashcards) deck generation, listing, review, star
#   quiz.py        (prefix /quiz)       quiz generation, fetch, grade, history
#
# Cost split: GENERATION runs the LLM ensemble and is therefore job-based
# (IngestJob + GET /jobs/{id} + frontend pollJob, exactly like chapter reports).
# REVIEW and GRADING are pure arithmetic / string comparison and are
# synchronous — they must never call a model.
# ==========================================================================


# --------------------------------------------------------------------------
# Study — cross-paper (GET /study/overview, /study/due, /study/papers)
# --------------------------------------------------------------------------
class StudyOverviewResponse(BaseModel):
    """``GET /api/study/overview`` — the Study screen's header numbers.

    Cross-paper by design: "what do I owe today, across everything I've read"
    is the question a per-report widget cannot answer, and the reason Study is
    a top-level nav item rather than a report section.

    Cheap by contract — counts only, no card bodies. Safe to poll on focus.
    """

    due: int = Field(
        default=0,
        ge=0,
        description=(
            "Cards still to do today across every paper: total minus the distinct "
            "cards already reviewed during the viewer's LOCAL day. There are no "
            "due dates in the session model, so this is a daily workload figure, "
            "not a schedule."
        ),
    )
    totals: DeckProgress = Field(
        default_factory=DeckProgress,
        description="Library-wide deck counters (all papers aggregated).",
    )
    quizzes_due: int = Field(
        default=0,
        ge=0,
        description=(
            "Quizzes never attempted. Deliberately NOT a daily figure like "
            "`due`: a quiz is a one-off assessment, not spaced repetition, so "
            "this is a backlog that falls as you take quizzes and rises as you "
            "generate them — it does not reset at midnight."
        ),
    )
    papers_with_decks: int = Field(default=0, ge=0)
    papers_with_quizzes: int = Field(default=0, ge=0)
    quiz_count: int = Field(default=0, ge=0, description="Quizzes across the library.")
    quiz_attempt_count: int = Field(default=0, ge=0)
    reviews_last_7_days: list[int] = Field(
        default_factory=list,
        description="Review counts per day, oldest→newest, length 7 (sparkline).",
    )
    streak_days: int = Field(
        default=0, ge=0, description="Consecutive days (ending today) with >= 1 review."
    )


class DueQueueResponse(BaseModel):
    """``GET /api/study/due`` — the cross-paper review queue.

    Query params: ``limit`` (default 50, max 200), ``paper_id`` (optional, the
    Report launch-card deep link), ``include_new`` (default true),
    ``shuffle`` (default false — the deck-level shuffle toggle).

    Ordering (stable, and the reason the ``(due_at)`` index exists): most
    overdue first — ``ORDER BY due_at ASC, id ASC`` — so the queue drains
    oldest-debt-first. With ``shuffle=true`` the *selected* window is shuffled
    after the ORDER BY + LIMIT, never before: shuffling the whole table would
    hand back cards that are not the most overdue.
    """

    cards: list[DueCard] = Field(
        default_factory=list, description="Due cards, most overdue first."
    )
    count: int = Field(default=0, ge=0, description="len(cards) — this page.")
    total_due: int = Field(
        default=0, ge=0, description="Total due cards matching the filter (may exceed count)."
    )
    paper_id: Optional[str] = Field(
        default=None, description="Echoed filter; null when the queue is library-wide."
    )
    shuffled: bool = Field(default=False, description="Echoes the shuffle param.")


class StudyPapersResponse(BaseModel):
    """``GET /api/study/papers`` — per-paper study state for the Study screen's
    paper switcher, and the source of the Report page's launch-card numbers in
    bulk (one request instead of N).
    """

    papers: list[PaperStudyStats] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)


# --------------------------------------------------------------------------
# Flashcards
# --------------------------------------------------------------------------
class FlashcardGenerateRequest(BaseModel):
    """``POST /api/flashcards/generate`` request.

    Generation reads the paper's **Key Concepts** section (which carries both
    the term/definition pairs and the concept explanations, having absorbed the
    former Glossary), its **Key Takeaways** recap, and paper-specific facts from
    **Key Results** / **Methods** — then writes cards with a mix of
    :class:`~deepvision.models.study.FlashcardOrigin` values. A deck of nothing
    but vocabulary terms is a word list, not a study deck.
    """

    paper_id: str = Field(..., description="Paper to build a deck for.")
    count: int = Field(
        default=20, ge=5, le=60, description="Target number of cards to generate."
    )
    force: bool = Field(
        default=False,
        description=(
            "Regenerate even if a deck exists. Regeneration UPSERTS on "
            "(paper_id, content_key) and preserves the SRS state of surviving "
            "cards; 'manual' cards are never touched."
        ),
    )
    settings_override: Optional[AppSettings] = Field(
        default=None, description="Per-run settings override (optional), as for /ingest."
    )


class FlashcardGenerateResponse(BaseModel):
    """``POST /api/flashcards/generate`` — cached deck, or a queued job.

    Same two-shape contract as :class:`ChapterReportResponse`, on purpose — the
    frontend reuses ``pollJob`` unchanged:

    - **200, ``cached: true``** — a deck already exists and ``force`` was not
      set. ``deck``/``progress`` are populated, ``job`` is null. No LLM work.
    - **202, ``cached: false``** — generation was queued. ``job`` is the fresh,
      already-persisted :class:`IngestJob` (so ``GET /jobs/{id}`` resolves
      immediately) and ``deck`` is null. Poll it, then
      ``GET /flashcards/deck/{paper_id}``.

    Like a chapter job, a flashcard job carries ``stages: []`` and
    ``current_stage: null`` — the seven ingestion ``JobStage`` values describe
    PDF processing and would be lies here (and an empty list is what stops a
    stage added to ``STAGE_ORDER`` from rendering as a phantom row that never
    completes). Progress is ``percent`` + ``log``.
    """

    paper_id: str
    cached: bool = Field(default=False)
    job: Optional[IngestJob] = Field(
        default=None, description="Queued generation job to poll; null when cached."
    )
    deck: Optional[list[Flashcard]] = Field(
        default=None, description="The existing deck; null when queued."
    )
    progress: Optional[DeckProgress] = Field(
        default=None, description="Deck counters; null when queued."
    )


class FlashcardDeckResponse(BaseModel):
    """``GET /api/flashcards/deck/{paper_id}`` — one paper's full deck.

    Query params: ``starred``
    (bool), ``due_only`` (bool), ``shuffle`` (bool).
    """

    paper_id: str
    paper_title: str = Field(default="")
    cards: list[Flashcard] = Field(default_factory=list)
    progress: DeckProgress = Field(default_factory=DeckProgress)
    generated_at: Optional[str] = Field(
        default=None, description="ISO datetime the deck was last generated."
    )
    shuffled: bool = Field(default=False)


class FlashcardReviewRequest(BaseModel):
    """``POST /api/flashcards/{card_id}/review`` — submit one rating.

    Synchronous and cheap: applying SM-2 is arithmetic plus two writes (update
    the card, append the review row). Never call a model here.

    The client sends only the rating; **the server computes the schedule**. The
    frontend may preview the next interval locally from the constants in
    ``models/study.py``, but it never sends one.
    """

    rating: Rating = Field(..., description="'again' | 'hard' | 'good' | 'easy'.")
    elapsed_ms: Optional[int] = Field(
        default=None, ge=0, description="Client-measured think time; stored, advisory."
    )


class FlashcardReviewResponse(BaseModel):
    """``POST /api/flashcards/{card_id}/review`` response.

    Tells the client where to put the card back and nothing more: the session
    queue lives in the browser, and the server owns only the log it appended
    and the two figures derived from it.

    ``requeue_after`` is the single reason the 6/12 gaps have no TypeScript
    copy — the numbers live in ``SESSION_REQUEUE_GAPS`` and travel over the
    wire, so there is nothing to drift.
    """

    card_id: str = Field(..., description="The card that was rated.")
    rating: str = Field(..., description="again | almost | got_it.")
    review_id: str = Field(..., description="Id of the appended FlashcardReview row.")
    requeue_after: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "How many OTHER cards to place before this one returns. Null means "
            "the card retires from this session (got_it). If fewer cards remain "
            "than this, the client puts it at the END — never drops it."
        ),
    )
    strength: int = Field(
        default=0,
        ge=0,
        description="Consecutive non-'again' ratings after this review, derived from the log.",
    )
    seen_today: int = Field(
        default=0, ge=0, description="Distinct cards reviewed during the LOCAL day."
    )
    due: int = Field(
        default=0, ge=0, description="Cards still to do today in the current scope."
    )
    progress: DeckProgress = Field(
        default_factory=DeckProgress, description="Deck counters after the review."
    )


class FlashcardStarRequest(BaseModel):
    """``POST /api/flashcards/{card_id}/star`` — set (not toggle) the star.

    Explicitly idempotent: the client sends the desired value, so a double-tap
    or a retried request cannot flip it back.
    """

    starred: bool = Field(..., description="Desired value.")


class FlashcardStarResponse(BaseModel):
    """``POST /api/flashcards/{card_id}/star`` response."""

    card_id: str
    starred: bool
    starred_count: int = Field(default=0, ge=0, description="Starred cards in this deck now.")


class FlashcardResetResponse(BaseModel):
    """``POST /api/flashcards/deck/{paper_id}/reset`` — restart the schedule.

    Resets every card in the deck to ``state: 'new'`` / ``ease 2.5`` /
    ``interval 0`` / ``repetitions 0`` / ``due now``. Card **text is kept** and
    the ``flashcard_reviews`` log is **not** deleted (it is append-only history;
    erasing it would make the reset itself unauditable).
    """

    paper_id: str
    reset_count: int = Field(default=0, ge=0, description="Cards returned to 'new'.")
    progress: DeckProgress = Field(default_factory=DeckProgress)


# --------------------------------------------------------------------------
# Quiz
# --------------------------------------------------------------------------
class QuizGenerateRequest(BaseModel):
    """``POST /api/quiz/generate`` request.

    The generator must produce a **mix**: all three
    :class:`~deepvision.models.study.QuestionKind` s and all three
    :class:`~deepvision.models.study.Difficulty` tiers, each question carrying a
    non-empty ``explanation`` and a :class:`~deepvision.models.study.QuizCitation`
    back to the section and page it came from. Questions it cannot cite must be
    dropped, not emitted with a null citation.
    """

    paper_id: str = Field(..., description="Paper to quiz on.")
    question_count: int = Field(
        default=10, ge=3, le=30, description="Target number of questions."
    )
    difficulties: list[Difficulty] = Field(
        default_factory=lambda: [Difficulty.RECALL, Difficulty.UNDERSTAND, Difficulty.APPLY],
        description="Tiers to include. Default: all three.",
    )
    kinds: list[QuestionKind] = Field(
        default_factory=lambda: [
            QuestionKind.MULTIPLE_CHOICE,
            QuestionKind.TRUE_FALSE,
            QuestionKind.SHORT_ANSWER,
        ],
        description="Question types to include. Default: all three.",
    )
    chapter_id: Optional[str] = Field(
        default=None,
        description="Optional Chapter.id — scope the quiz to one chapter's page range.",
    )
    settings_override: Optional[AppSettings] = Field(default=None)


class QuizGenerateResponse(BaseModel):
    """``POST /api/quiz/generate`` — always a queued job (202).

    Unlike flashcards there is no cache short-circuit: generating a quiz
    deliberately creates a *new* quiz rather than replacing the old one, so past
    attempt history stays meaningful. Poll ``job``, then
    ``GET /quiz/paper/{paper_id}`` to pick up the newest quiz (or use
    ``quiz_id``, which is minted up-front and is valid the moment the job
    reaches ``done``).
    """

    paper_id: str
    quiz_id: str = Field(
        ..., description="Id the quiz WILL have — resolvable once the job is done."
    )
    job: IngestJob = Field(..., description="Queued generation job to poll.")


class QuizListResponse(BaseModel):
    """``GET /api/quiz/paper/{paper_id}`` — this paper's quizzes, newest first."""

    paper_id: str
    quizzes: list[QuizSummary] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)


class QuizAttemptRequest(BaseModel):
    """``POST /api/quiz/{quiz_id}/attempt`` — submit answers for grading.

    **Grading is server-side, always.** The client never sends (and cannot know)
    a correctness verdict: the answer keys live only in ``quiz_questions`` and
    are never serialized into the pre-answer payload
    (:class:`~deepvision.models.study.QuizQuestionPublic` has no field for them).
    Any ``is_correct``-looking value in a request body is ignored.

    Grading is deterministic and synchronous — option-id equality for
    ``multiple_choice``/``true_false``, normalized string / token-overlap
    comparison for ``short_answer``. It must never call a model.

    ``mode``/``retry_of_attempt_id`` mark a "retry only what I missed" run; the
    server validates that every submitted ``question_id`` really was missed in
    ``retry_of_attempt_id`` (400 otherwise) so the mode cannot be used to fake a
    high score on a cherry-picked subset.
    """

    answers: list[SubmittedAnswer] = Field(
        default_factory=list,
        description="One per answered question. Omitted questions grade as incorrect.",
    )
    mode: AttemptMode = Field(
        default=AttemptMode.FULL, description="'full' | 'retry_missed'."
    )
    retry_of_attempt_id: Optional[str] = Field(
        default=None, description="Required iff mode == 'retry_missed'."
    )
    started_at: Optional[str] = Field(
        default=None, description="ISO datetime the user began; server clamps to now if absent."
    )


class QuizCheckRequest(BaseModel):
    """``POST /api/quiz/{quiz_id}/check`` — grade ONE answer, persist NOTHING.

    This exists so the runner can give **immediate per-question feedback**
    without inventing an attempt. ``QuizQuestionPublic`` deliberately has no
    answer field, so the only way to know whether an answer is right is a server
    round trip — and doing that round trip against
    ``POST /{quiz_id}/attempt`` would write one ``quiz_attempts`` row *per
    question*, turning a single ten-question run into ten rows of history with
    nine junk scores (and a wrong ``attempt_count`` on every launch card).

    So: ``/check`` grades and returns, ``/attempt`` grades and records. The
    client calls ``/check`` once per question as the user answers, then
    ``/attempt`` exactly once at the end with the full answer list.

    Same grader either way (:func:`~deepvision.study.grading.grade_question`),
    so the live feedback can never disagree with the final scored attempt.
    """

    answer: SubmittedAnswer = Field(
        ..., description="The single answer to grade. Must belong to this quiz."
    )


class QuizCheckResponse(BaseModel):
    """``POST /api/quiz/{quiz_id}/check`` response — one graded answer.

    Nothing was written to the database. The returned
    :class:`~deepvision.models.study.GradedAnswer` carries the verdict, the
    correct answer and the explanation, exactly as it would inside a persisted
    attempt.
    """

    quiz_id: str
    graded: GradedAnswer = Field(..., description="Server verdict + explanation + citation.")


class QuizRetryRequest(BaseModel):
    """``POST /api/quiz/{quiz_id}/retry`` — build the "only what I missed" run.

    Returns a :class:`~deepvision.models.study.QuizPublic` containing **only**
    the questions missed in ``source_attempt_id``, in the original quiz order,
    with ``mode: 'retry_missed'`` and ``retry_of_attempt_id`` set. Answers stay
    hidden, exactly as on a first pass. 400 if the source attempt has no misses.
    """

    source_attempt_id: str = Field(
        ..., description="Attempt whose missed_question_ids define the scope."
    )


class QuizAttemptHistoryResponse(BaseModel):
    """``GET /api/quiz/{quiz_id}/attempts`` — attempt history, newest first.

    Summaries only (no per-answer detail): the history list is a table of scores
    and dates. Fetch one attempt's full :class:`~deepvision.models.study.QuizAttempt`
    (with explanations) from ``GET /api/quiz/attempts/{attempt_id}``.
    """

    quiz_id: str
    paper_id: str
    attempts: list[QuizAttemptSummary] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)
    best_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    """Uniform error envelope for non-2xx responses."""

    detail: str = Field(..., description="Human-readable error message.")
    code: Optional[str] = Field(default=None, description="Stable machine error code.")
