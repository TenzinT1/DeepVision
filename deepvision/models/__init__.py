"""DeepVision shared data models (Pydantic v2).

This package is the single source of truth for every domain shape that crosses a
module boundary. All six builder domains and the frontend import from here."""

from deepvision.models.chat import (
    AnswerKind,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
)
from deepvision.models.chunks import (
    AnyChunk,
    BBox,
    Chunk,
    ImageChunk,
    Modality,
    OCRChunk,
    Provenance,
    SourceRef,
    TextChunk,
    VisionInsightChunk,
)
from deepvision.models.job import (
    STAGE_ORDER,
    IngestJob,
    JobStage,
    JobState,
    LogEntry,
    StageProgress,
    StageStatus,
)
from deepvision.models.paper import PaperMeta, PaperStatus
from deepvision.models.report import (
    SECTION_ORDER,
    Chapter,
    ChapterSource,
    Citation,
    MediaRef,
    Report,
    ReportScope,
    ReportStats,
    Section,
    SectionName,
)
from deepvision.models.settings import (
    AppSettings,
    OCRLanguage,
    ProviderKeys,
    ProviderMode,
)
from deepvision.models.study import (
    AttemptMode,
    CardStrength,
    DeckProgress,
    Difficulty,
    DueCard,
    Flashcard,
    FlashcardOrigin,
    FlashcardReview,
    GradedAnswer,
    PaperStudyStats,
    QuestionKind,
    Quiz,
    QuizAttempt,
    QuizAttemptSummary,
    QuizCitation,
    QuizOption,
    QuizPublic,
    QuizQuestion,
    QuizQuestionPublic,
    QuizSummary,
    Rating,
    StudyItemKind,
    SubmittedAnswer,
    describe_scheduler_contract,
)

__all__ = [
    # chunks
    "Modality",
    "Provenance",
    "BBox",
    "SourceRef",
    "Chunk",
    "TextChunk",
    "OCRChunk",
    "VisionInsightChunk",
    "ImageChunk",
    "AnyChunk",
    # paper
    "PaperMeta",
    "PaperStatus",
    # job
    "IngestJob",
    "JobStage",
    "JobState",
    "StageStatus",
    "StageProgress",
    "LogEntry",
    "STAGE_ORDER",
    # report
    "Report",
    "ReportScope",
    "Section",
    "SectionName",
    "SECTION_ORDER",
    "Citation",
    "MediaRef",
    "ReportStats",
    "Chapter",
    "ChapterSource",
    # chat
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "AnswerKind",
    "ChatRole",
    # compare
    # settings
    "AppSettings",
    "ProviderMode",
    "ProviderKeys",
    "OCRLanguage",
    # study — flashcards / SRS
    "Rating",
    "CardStrength",
    "FlashcardOrigin",
    "Flashcard",
    "FlashcardReview",
    "DueCard",
    "DeckProgress",
    "describe_scheduler_contract",
    # study — quiz
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
    # study — cross-paper surface
    "StudyItemKind",
    "PaperStudyStats",
]
