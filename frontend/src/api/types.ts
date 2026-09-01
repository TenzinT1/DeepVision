/* ==========================================================================
   DeepVision API types — mirrors deepvision/api/schemas.py and
   deepvision/models/*.py field-for-field. Enum string VALUES are load-bearing
   (the backend and this UI both switch on them) — never rename them.
   Source of truth:/§4 and deepvision/models/*.py.
   ========================================================================== */

// --------------------------------------------------------------------------
// Enums (string literal unions — wire values match Pydantic str Enum values)
// --------------------------------------------------------------------------

export type PaperStatus = "queued" | "processing" | "ready" | "error";

/**
 * Fixed pipeline stage order — STAGE_ORDER in deepvision/models/job.py.
 *
 * "Generate report" is the last stage on purpose: report generation is part of
 * the ingest job, so `state: "done"` means the report actually exists. When
 * only that stage fails the job still finishes `done` (chunks/vectors/images
 * are persisted, chat + compare work) with the stage left in `error` — so the
 * "Open report →" affordance is gated on the stage, not on `state`.
 */
export type JobStage =
  | "Download"
  | "Extract text"
  | "Extract images"
  | "OCR"
  | "Vision analysis"
  | "Embedding"
  | "Generate report";

export const STAGE_ORDER: JobStage[] = [
  "Download",
  "Extract text",
  "Extract images",
  "OCR",
  "Vision analysis",
  "Embedding",
  "Generate report",
];

/** The report-generation stage, referenced by the ingestion modal's gating. */
export const REPORT_STAGE: JobStage = "Generate report";

export type JobState = "queued" | "running" | "done" | "error";

export type StageStatus = "pending" | "running" | "done" | "error";

export type Modality = "text" | "ocr" | "vision" | "image";

/**
 * Parse a timestamp coming off this API.
 *
 * Every `datetime` the backend persists and serialises is **naive UTC**
 * (`datetime.utcnow()` — SQLite has no timezone type, so the whole server
 * agrees on one convention and never attaches an offset). The resulting ISO
 * string therefore looks like `2026-08-05T15:38:12.069550` with no trailing
 * `Z`, and ECMAScript specifies that a date-time form with no offset is parsed
 * as **local time**. Handing one straight to `new Date()` silently shifts it by
 * the viewer's UTC offset — up to 14 hours, i.e. the wrong calendar day for
 * anyone not on UTC, which is how "generated Aug 5" renders as "Aug 4".
 *
 * So: if the string carries no offset and no `Z`, say `Z`. Values that already
 * carry one (anything the client itself produced with `toISOString()`) are left
 * alone.
 */
export function parseServerDate(iso: string): Date {
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(iso);
  return new Date(hasZone ? iso : `${iso}Z`);
}

/** Drives UI badge colour: text = plain, ocr = amber, vision = teal. */
export type Provenance = "text" | "ocr" | "vision";

/**
 * Fixed section order — SECTION_ORDER in deepvision/models/report.py.
 *
 * Eleven sections, ordered as a lesson for a reader with NO background in the
 * topic, under one rule: every section answers exactly one question no other
 * section answers (that question ships on `Section.question` and is rendered
 * under the heading). At a Glance places the paper in five scannable lines;
 * Overview orients in prose; Background gives the before-picture; Key Concepts
 * gives the vocabulary; Methods and Key Results cover what was done and found;
 * Figures shows the evidence; Limitations & Open Questions covers what the work
 * does NOT establish; Why It Matters covers consequences; Key Takeaways recaps;
 * Study Questions lets the reader self-check.
 *
 * Changed from the previous ten: `Glossary` merged into `Key Concepts`,
 * `Conclusions` became `Why It Matters`, `At a Glance` and
 * `Limitations & Open Questions` are new, and `Key Takeaways` moved from second
 * position to a recap slot after the evidence. Reports persisted under the old
 * set are backfilled with empty placeholders for the new names by
 * normalize_sections() on load, and their removed sections are dropped — a
 * re-run is needed for real content.
 */
export type SectionName =
  | "At a Glance"
  | "Overview"
  | "Background"
  | "Key Concepts"
  | "Methods"
  | "Key Results"
  | "Figures"
  | "Limitations & Open Questions"
  | "Why It Matters"
  | "Key Takeaways"
  | "Study Questions";

export const SECTION_ORDER: SectionName[] = [
  "At a Glance",
  "Overview",
  "Background",
  "Key Concepts",
  "Methods",
  "Key Results",
  "Figures",
  "Limitations & Open Questions",
  "Why It Matters",
  "Key Takeaways",
  "Study Questions",
];

/**
 * What a Report covers — ReportScope in deepvision/models/report.py.
 * A chapter report has the SAME eleven sections as a paper report; render it with
 * the same SectionCard components and only swap the header line.
 */
export type ReportScope = "paper" | "chapter";

/**
 * How a chapter list was detected — ChapterSource in deepvision/models/report.py.
 * `outline` = the PDF's embedded outline/bookmarks (books, theses: authoritative,
 * may be nested). `headings` = any non-outline derivation: the ingestion heading
 * heuristic in page order (the arXiv case: no outline, but detectable headings),
 * or the last-resort "Pages 1-10" buckets; always `level: 1`.
 */
export type ChapterSource = "outline" | "headings";


export type ChatRole = "user" | "assistant" | "system";

/**
 * How an assistant answer was produced — AnswerKind in deepvision/models/chat.py.
 * `content` is the only kind grounded in retrieved passages and the only one
 * with `citations[]` / `figures[]`; the other three are answered
 * deterministically from the database (no retrieval, so no citations/figures):
 * `citation` = a formatted bibliographic citation, `metadata` = a fact about
 * the paper (authors, year, id), `structure` = a fact about the document
 * itself (pages, figures, chapters).
 */
export type AnswerKind = "content" | "citation" | "metadata" | "structure";

export type ProviderMode = "local" | "api";

export type OCRLanguage = "eng" | "eng+fra" | "chi_sim" | "jpn" | "auto";

export type SearchSort = "relevance" | "newest" | "most_cited";

export type SearchDateRange =
  | "any"
  | "past_month"
  | "past_6_months"
  | "past_year"
  | "2024_2025";

// --------------------------------------------------------------------------
// Shared value types
// --------------------------------------------------------------------------

export interface BBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

// --------------------------------------------------------------------------
// Paper
// --------------------------------------------------------------------------

export interface PaperMeta {
  id: string;
  arxiv_id: string;
  arxiv_label: string;
  version?: string | null;
  title: string;
  authors: string[];
  abstract: string;
  categories: string[];
  published?: string | null; // ISO date 'YYYY-MM-DD'
  updated?: string | null; // ISO date 'YYYY-MM-DD'
  pdf_url?: string | null;
  abs_url?: string | null;
  status: PaperStatus;
  /**
   * Ingestion finished and the paper's content is usable (chunks, vectors and
   * images are persisted). Together with `status === 'ready'` this is the
   * CHAT / COMPARE gate. It says nothing about whether a report exists.
   */
  ingested: boolean;
  /**
   * A whole-paper report row actually exists. This — not `status` — is the
   * "Open report" gate.
   *
   * The two are deliberately separate: report generation is the last ingestion
   * stage and takes ~20 minutes on a local model, so the paper goes
   * `ready`/`ingested` (chat + compare unlock) well before `has_report` flips.
   * Gating "Open report" on `status` instead would offer a report that does not
   * exist yet, and that click makes GET /report start a second, competing
   * agent run.
   *
   * Derived server-side from the existence of the report row, so it cannot go
   * stale — but it is a snapshot: refetch `GET /papers` to see it flip.
   */
  has_report: boolean;
  thumbnail_path?: string | null;
  page_count?: number | null;
  figure_count?: number | null;
  error_message?: string | null;
  created_at: string; // ISO datetime
  updated_at: string; // ISO datetime
}

// --------------------------------------------------------------------------
// Job / ingestion
// --------------------------------------------------------------------------

export interface LogEntry {
  t: string; // 'HH:MM:SS'
  m: string;
  stage?: JobStage | null;
}

export interface StageProgress {
  stage: JobStage;
  status: StageStatus;
  detail?: string | null;
  /**
   * The stage finished (`status: "done"`) but not at full quality.
   *
   * Set by "Generate report": an LLM call that exceeds the local model's
   * timeout is swallowed by the agents' draft-fallback path and returns the
   * deterministic extractive draft, so the report is produced with no error
   * anywhere — `done` alone cannot tell a model-written report from a mostly
   * extractive one. Optional because job rows persisted before this flag
   * existed have no such key.
   */
  degraded?: boolean | null;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface IngestJob {
  id: string;
  paper_id: string;
  arxiv_id: string;
  state: JobState;
  current_stage?: JobStage | null;
  percent: number; // 0-100
  stages: StageProgress[];
  log: LogEntry[];
  error_message?: string | null;
  created_at: string;
  updated_at: string;
}

// --------------------------------------------------------------------------
// Report
// --------------------------------------------------------------------------

export interface Citation {
  id: string;
  marker: number;
  source: string;
  page: number;
  page_label: string;
  page_image_path?: string | null;
  snippet: string;
  bbox?: BBox | null;
  chunk_id?: string | null;
}

export interface MediaRef {
  id: string;
  kind: string; // 'figure' | 'table' (plain string on the wire)
  label: string;
  caption: string;
  provenance: Provenance;
  image_path?: string | null;
  thumbnail_path?: string | null;
  filename?: string | null;
  page?: number | null;
  chunk_id?: string | null;
}

export interface ReportStats {
  pages: number;
  figures: number;
  citations_extracted: number;
  reading_time_min: number;
}

export interface Section {
  id: string;
  name: SectionName;
  /**
   * The one question this section answers — SECTION_QUESTIONS in
   * deepvision/models/report.py. Rendered under the section heading. Backfilled
   * by normalize_sections() on load, so it is only null for a section the
   * backend does not recognise.
   */
  question?: string | null;
  body_markdown: string;
  deep_dive_markdown?: string | null;
  badge?: string | null;
  provenance: Provenance[];
  citations: Citation[];
  media: MediaRef[];
  confidence?: number | null;
  /**
   * True when an LLM call for this section fell back to the deterministic
   * extractive draft — the body is quoted from the PDF, not model-written.
   * Per-section counterpart of StageProgress.degraded.
   */
  degraded: boolean;
  default_open: boolean;
}

export interface Report {
  id: string;
  paper_id: string;
  paper?: PaperMeta | null;
  stats: ReportStats;
  sections: Section[];
  generated_at: string;
  model_used?: string | null;
  /** 'paper' for the whole-document report, 'chapter' for a per-chapter one. */
  scope: ReportScope;
  /** Chapter.id — set iff scope === 'chapter'. */
  chapter_id?: string | null;
  /** Header line: `Reporting on: ${chapter_title}`. Set iff scope === 'chapter'. */
  chapter_title?: string | null;
  chapter_page_start?: number | null;
  chapter_page_end?: number | null;
}

// --------------------------------------------------------------------------
// Chapters (per-chapter reports)
// --------------------------------------------------------------------------

/**
 * One selectable chapter — Chapter in deepvision/models/report.py.
 *
 * A chapter IS a contiguous page range; that range is what scopes retrieval on
 * the backend. `id` is stable across re-derivations (it keys the persisted
 * chapter report), so always send back the `id` you were given — the picker is
 * a dropdown, never a text input.
 *
 * Render as: `title`, indented by (`level` - 1), with `pages {page_start}–{page_end}`
 * as the secondary line.
 */
export interface Chapter {
  id: string;
  /** 0-based position in the list (display order). */
  index: number;
  title: string;
  /** Outline depth: 1 = top-level chapter, 2 = subsection. Always 1 for 'headings'. */
  level: number;
  page_start: number;
  /** Inclusive; the last chapter's page_end is the document's page count. */
  page_end: number;
  source: ChapterSource;
}

/**
 * `GET /api/papers/{paper_id}/chapters` — cheap, no model calls, safe to call
 * on page load. `detected === false` (with an empty `chapters`) means the PDF
 * has no usable chapter structure: disable the picker, don't show one option.
 */
export interface ChaptersResponse {
  paper_id: string;
  chapters: Chapter[];
  source?: ChapterSource | null;
  page_count: number;
  detected: boolean;
}

/** `POST /api/report/{paper_id}/chapter` request body. */
export interface ChapterReportRequest {
  /** A Chapter.id from ChaptersResponse — never a typed title. */
  chapter_id: string;
  /** Re-generate even if a chapter report already exists. */
  force?: boolean;
  settings_override?: AppSettings | null;
}

/**
 * `POST /api/report/{paper_id}/chapter` response.
 *
 * Exactly one of `report` / `job` is non-null:
 * - `cached: true` (200) → `report` is ready, render it now.
 * - `cached: false` (202) → `job` was queued; `pollJob(job.id)` (the same helper
 *   ingestion uses), then `getChapterReport(paper_id, chapter_id)`.
 *
 * A chapter job has `stages: []` and `current_stage: null` — the seven
 * ingestion JobStage values are meaningless here. Show `percent` and the last `log` line.
 */
export interface ChapterReportResponse {
  paper_id: string;
  chapter_id: string;
  cached: boolean;
  job?: IngestJob | null;
  report?: Report | null;
}

// --------------------------------------------------------------------------
// Chat
// --------------------------------------------------------------------------

export interface ChatMessage {
  id: string;
  paper_id: string;
  role: ChatRole;
  text: string;
  citations: Citation[];
  figures: MediaRef[];
  created_at: string;
  /** null for a user/system message, or for an assistant message predating this field. */
  answer_kind?: AnswerKind | null;
}

export interface ChatRequest {
  paper_id: string;
  message: string;
  session_id?: string | null;
  history?: ChatMessage[];
  top_k?: number;
}

export interface ChatResponse {
  session_id: string;
  message: ChatMessage;
  retrieved_chunk_ids: string[];
}


// --------------------------------------------------------------------------
// Settings
// --------------------------------------------------------------------------

export interface ProviderKeys {
  llm_api_key?: string | null;
  vision_api_key?: string | null;
  embedding_api_key?: string | null;
}

export interface AppSettings {
  llm_mode: ProviderMode;
  vision_mode: ProviderMode;
  embedding_mode: ProviderMode;
  /** ModelChoice.id from model_catalog.llm; null = backend default for the mode. */
  llm_model?: string | null;
  /** ModelChoice.id from model_catalog.vision. */
  vision_model?: string | null;
  /** ModelChoice.id from model_catalog.embedding. */
  embedding_model?: string | null;
  keys: ProviderKeys;
  ocr_language: OCRLanguage;
  chunk_size: number; // 256-2048
  chunk_overlap: number; // 0-400, must be < chunk_size
  image_embedding: boolean;
  report_detail: ReportDetail;
}

export type ReportDetail = "concise" | "standard" | "detailed";

/** Vendor of a catalog entry — decides which API key it needs. */
export type ModelVendor = "local" | "anthropic" | "openai";

/**
 * One selectable model in a Settings dropdown — ModelChoice in
 * deepvision/api/schemas.py. Selecting a choice must set BOTH
 * `<stage>_model = choice.id` AND `<stage>_mode = choice.mode`.
 */
export interface ModelChoice {
  id: string;
  label: string;
  mode: ProviderMode;
  vendor: ModelVendor;
  note: string;
  needs_key: boolean;
}

/**
 * Backend-published model catalog — ModelCatalog / MODEL_CATALOG in
 * deepvision/api/schemas.py. Never hard-code model lists in the UI; render
 * these. Note there is no Anthropic embedding option (no such API exists).
 */
export interface ModelCatalog {
  llm: ModelChoice[];
  vision: ModelChoice[];
  embedding: ModelChoice[];
}

/**
 * A report-detail option and the length it actually promises — DetailLevelInfo
 * in deepvision/api/schemas.py, sourced from DETAIL_LEVEL_INFO in
 * deepvision/models/settings.py (the same constant the summarizer targets).
 */
export interface DetailLevelInfo {
  value: ReportDetail;
  label: string;
  blurb: string;
}

export interface SettingsResponse {
  settings: AppSettings; // keys redacted (values null) on GET
  needs_keys: boolean;
  keys_present: { llm: boolean; vision: boolean; embedding: boolean } & Record<string, boolean>;
  vision_available: boolean;
  vision_status_text: string;
  /** Static reference data: what the model dropdowns offer, per stage. */
  model_catalog: ModelCatalog;
  /** Static reference data: detail options + their length blurbs, in order. */
  detail_levels: DetailLevelInfo[];
}

export interface UpdateSettingsRequest {
  settings: AppSettings;
}

// --------------------------------------------------------------------------
// Search
// --------------------------------------------------------------------------

export interface SearchResultItem {
  paper: PaperMeta;
  ingested: boolean;
  in_library: boolean;
}

export interface SearchResponse {
  query: string;
  results: SearchResultItem[];
  count: number;
}

// --------------------------------------------------------------------------
// Ingest / jobs / papers
// --------------------------------------------------------------------------

export interface IngestRequest {
  arxiv_id: string;
  settings_override?: AppSettings | null;
}

/**
 * Response of `POST /api/ingest` AND of `POST /api/ingest/upload`.
 *
 * The upload route is multipart/form-data (`file`: the PDF, optional `title`),
 * so it has no JSON request type — but it returns this same shape on purpose,
 * so the existing pollJob / IngestionModal flow works unchanged for uploads.
 * For an uploaded paper, `PaperMeta.arxiv_id` is the placeholder
 * `"upload:<paper_id>"` and `arxiv_label` is `"Uploaded PDF"`.
 */
export interface IngestResponse {
  job: IngestJob;
  paper_id: string;
}

export interface JobStatusResponse {
  job: IngestJob;
}

export interface PapersListResponse {
  papers: PaperMeta[];
  count: number;
}

export interface DeletePaperResponse {
  paper_id: string;
  deleted: boolean;
}

export interface RerunResponse {
  job: IngestJob;
  paper_id: string;
}

// --------------------------------------------------------------------------
// Media / health
// --------------------------------------------------------------------------


export interface MediaMetadata {
  chunk_id: string;
  paper_id: string;
  kind: string;
  label?: string | null;
  caption?: string | null;
  provenance?: Provenance | null;
  page?: number | null;
  content_type: string;
  width?: number | null;
  height?: number | null;
  url: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  llm_online: boolean;
  vision_available: boolean;
  db_ok: boolean;
}

/* ==========================================================================
   STUDY — the interactive study layer.

   Mirrors deepvision/models/study.py + the Study block in
   deepvision/api/schemas.py. Enum string values are byte-identical to the
   Python ones.

   Study is its OWN top-level nav item (between Report and Chat), not a report
   section, for three structural reasons:
     1. Report sections are regenerated wholesale (report_generator overwrites
        ReportRow.sections), which would destroy study progress.
     2. Spaced repetition is only useful across ALL papers — the due queue is
        cross-paper, which a per-report widget cannot express.
     3. Reports export to Markdown/PDF, where an interactive widget cannot
        render.

   The Report page shows LAUNCH CARDS only (live counts from PaperStudyStats,
   deep-linking to /study?paper=…), never the widget itself.
   ========================================================================== */

// --------------------------------------------------------------------------
// Study — enums
// --------------------------------------------------------------------------

/** The three review buttons — Rating in deepvision/models/study.py. */
export type Rating = "again" | "almost" | "got_it";

export const RATINGS: Rating[] = ["again", "almost", "got_it"];

/** What the generator built a card from — FlashcardOrigin. */
export type FlashcardOrigin =
  | "glossary"
  | "key_concept"
  | "takeaway"
  | "fact"
  | "manual";

/** Question type — QuestionKind. Each renders a different input. */
export type QuestionKind = "multiple_choice" | "true_false" | "short_answer";

/** Difficulty tier — Difficulty. A good quiz spans all three. */
export type Difficulty = "recall" | "understand" | "apply";

export const DIFFICULTIES: Difficulty[] = ["recall", "understand", "apply"];

/**
 * How an attempt was started — AttemptMode.
 * `retry_missed` scores over a SUBSET of questions, so its score is NOT
 * comparable to a `full` attempt's — label it in the UI, never chart them
 * on one line.
 */
export type AttemptMode = "full" | "retry_missed";

/** Which study tool a launch card / filter refers to — StudyItemKind. */
export type StudyItemKind = "flashcards" | "quiz";

// --------------------------------------------------------------------------
// Study — SM-2 constants (mirror of deepvision/models/study.py)
//
// Here ONLY so the UI can preview "Good → 3d" on the buttons before the user
// commits. The SERVER computes the real schedule and its answer is
// authoritative: never send a computed interval, and always render
// FlashcardReviewResponse.interval_days / due_in_minutes after the call.
// --------------------------------------------------------------------------


// --------------------------------------------------------------------------
// Study — flashcards
// --------------------------------------------------------------------------

/**
 * What the review log says about one card — CardStrength.
 *
 * Derived server-side on every read, never stored: there is no scheduling
 * column anywhere any more. See describe_scheduler_contract() §B.
 */
export interface CardStrength {
  /** Consecutive non-'again' ratings, counting back from the newest review. */
  strength: number;
  reviews: number;
  last_reviewed_at?: string | null;
  /** Reviewed at least once during the viewer's LOCAL day. */
  seen_today: boolean;
}

export interface Flashcard {
  id: string;
  paper_id: string;
  front: string;
  back: string;
  hint?: string | null;
  origin: FlashcardOrigin;
  /** Stable dedupe key; regeneration upserts on (paper_id, content_key). */
  content_key: string;
  source_section?: SectionName | null;
  source_page?: number | null;
  chunk_id?: string | null;
  tags: string[];
  starred: boolean;
  strength: CardStrength;
  created_at: string;
  updated_at: string;
}

/** One row of the append-only review log — FlashcardReview. */
export interface FlashcardReview {
  id: string;
  card_id: string;
  paper_id: string;
  /**
   * A plain string, not Rating: rows written by the retired SM-2 scheduler
   * still carry 'hard' | 'good' | 'easy' and the log is append-only.
   */
  rating: string;
  elapsed_ms?: number | null;
  reviewed_at: string;
}

/** A card in the cross-paper due queue, with its paper's context. */
export interface DueCard {
  card: Flashcard;
  paper_title: string;
  paper_label: string;
}

export interface DeckProgress {
  total: number;
  /** Still to do today: total minus distinct cards seen this LOCAL day. */
  due_count: number;
  starred_count: number;
  reviews_today: number;
}

// --------------------------------------------------------------------------
// Study — quiz
// --------------------------------------------------------------------------

/** Where a question's answer is grounded — shown with the explanation. */
export interface QuizCitation {
  section: SectionName;
  page?: number | null;
  chunk_id?: string | null;
  snippet: string;
}

/**
 * One selectable option. Deliberately has NO `is_correct` field — this is what
 * ships to the browser before the user answers.
 */
export interface QuizOption {
  /** 'a' | 'b' | 'c' | 'd' for multiple_choice; 'true' | 'false' for true_false. */
  id: string;
  text: string;
}

/**
 * The HIDDEN-UNTIL-ANSWERED wire shape — QuizQuestionPublic. This is what
 * `GET /api/quiz/{quiz_id}` returns. There is structurally no field here that
 * could carry the answer or the explanation; they arrive only inside a
 * GradedAnswer, after submission. Do not add one.
 */
export interface QuizQuestionPublic {
  id: string;
  index: number;
  prompt: string;
  kind: QuestionKind;
  difficulty: Difficulty;
  /** 4 for multiple_choice, the fixed True/False pair for true_false, [] for short_answer. */
  options: QuizOption[];
  /**
   * A POINTER only: `section` / `page` / `chunk_id` are populated, `snippet` is
   * always "" here. The snippet is a verbatim quote of the source the question
   * came from and would often contain the answer outright, so the server blanks
   * it before the user commits. The real snippet arrives in
   * `GradedAnswer.citation`. Render the source chip from a GradedAnswer, not
   * from this.
   */
  citation?: QuizCitation | null;
}

/** `GET /api/quiz/{quiz_id}` and `POST /api/quiz/{quiz_id}/retry` response. */
export interface QuizPublic {
  id: string;
  paper_id: string;
  paper_title: string;
  title: string;
  questions: QuizQuestionPublic[];
  /** Counts per Difficulty value, e.g. { recall: 4, understand: 4, apply: 2 }. */
  difficulty_mix: Record<string, number>;
  /** Counts per QuestionKind value. */
  kind_mix: Record<string, number>;
  generated_at: string;
  /** 'retry_missed' when this payload is a scoped retry. */
  mode: AttemptMode;
  retry_of_attempt_id?: string | null;
}

/** One row in a paper's quiz list — no questions, so it is cheap. */
export interface QuizSummary {
  id: string;
  paper_id: string;
  title: string;
  question_count: number;
  difficulty_mix: Record<string, number>;
  generated_at: string;
  attempt_count: number;
  /** Best attempt score in [0,1]; null if never attempted. */
  best_score?: number | null;
  last_attempted_at?: string | null;
}

/**
 * One answer the client submits. Never carries a correctness claim — grading is
 * server-side and the answer key never leaves the backend.
 */
export interface SubmittedAnswer {
  question_id: string;
  /** For multiple_choice / true_false. */
  selected_option_id?: string | null;
  /** For short_answer. */
  answer_text?: string | null;
  elapsed_ms?: number | null;
}

/**
 * One graded answer — the ONLY shape that reveals the correct answer and the
 * explanation, and it only exists inside a QuizAttempt response.
 */
export interface GradedAnswer {
  question_id: string;
  index: number;
  prompt: string;
  kind: QuestionKind;
  difficulty: Difficulty;
  options: QuizOption[];
  selected_option_id?: string | null;
  answer_text?: string | null;
  /** The server's verdict. Authoritative — never recompute it in the UI. */
  is_correct: boolean;
  correct_option_id?: string | null;
  correct_answer_text?: string | null;
  /** Why the right answer is right (and the tempting wrong one wrong). */
  explanation: string;
  citation?: QuizCitation | null;
}

/** A graded run — QuizAttempt. Response of POST /api/quiz/{quiz_id}/attempt. */
export interface QuizAttempt {
  id: string;
  quiz_id: string;
  paper_id: string;
  mode: AttemptMode;
  retry_of_attempt_id?: string | null;
  question_ids: string[];
  answers: GradedAnswer[];
  total: number;
  correct: number;
  /** correct / total, in [0,1]. Denominator differs by mode — see AttemptMode. */
  score: number;
  started_at: string;
  finished_at: string;
  /** Exact input to a "retry only what I missed" run. */
  missed_question_ids: string[];
}

export interface QuizAttemptSummary {
  id: string;
  quiz_id: string;
  paper_id: string;
  mode: AttemptMode;
  total: number;
  correct: number;
  score: number;
  missed_count: number;
  started_at: string;
  finished_at: string;
}

// --------------------------------------------------------------------------
// Study — cross-paper surface + launch cards
// --------------------------------------------------------------------------

/**
 * Live per-paper study state — PaperStudyStats. EXACTLY what the two Report
 * launch cards render:
 *   "Flashcards · {cards_total} cards"
 *      → /study?paper={paper_id}&tab=flashcards
 *   "Quiz · {quiz_count} quizzes · best {quiz_best_score}"
 *      → /study?paper={paper_id}&tab=quiz
 * `has_deck: false` / `quiz_count: 0` is the "Generate" state, not an error.
 */
export interface PaperStudyStats {
  paper_id: string;
  paper_title: string;
  paper_label: string;
  has_deck: boolean;
  cards_total: number;
  cards_starred: number;
  deck_generated_at?: string | null;
  quiz_count: number;
  quiz_attempt_count: number;
  /**
   * Best score across this paper's FULL attempts (falls back to any attempt
   * only when there has never been a full one). A `retry_missed` run is scored
   * over a smaller, self-selected denominator, so it is deliberately not
   * allowed to outrank a real full attempt here. Same rule as
   * `QuizSummary.best_score` and `QuizAttemptHistoryResponse.best_score`.
   */
  quiz_best_score?: number | null;
  quiz_last_attempted_at?: string | null;
}

/** `GET /api/study/overview` — the Study screen's header numbers. */
export interface StudyOverviewResponse {
  /** Cards still to do today across every paper. No due dates exist. */
  due: number;
  /**
   * Quizzes never attempted. NOT a daily figure like `due` — a quiz is a
   * one-off assessment, so this is a backlog that does not reset at midnight.
   */
  quizzes_due: number;
  totals: DeckProgress;
  papers_with_decks: number;
  papers_with_quizzes: number;
  quiz_count: number;
  quiz_attempt_count: number;
  /** Review counts per day, oldest→newest, length 7 (sparkline). */
  reviews_last_7_days: number[];
  streak_days: number;
}

/**
 * `GET /api/study/due?limit=&paper_id=&include_new=&shuffle=` — the cross-paper
 * review queue, ordered most-overdue-first. `total_due` may exceed `count`
 * (which is this page).
 */
export interface DueQueueResponse {
  cards: DueCard[];
  count: number;
  total_due: number;
  paper_id?: string | null;
  shuffled: boolean;
}

/** `GET /api/study/papers` — per-paper stats in bulk (one call, not N). */
export interface StudyPapersResponse {
  papers: PaperStudyStats[];
  count: number;
}

// --------------------------------------------------------------------------
// Study — flashcard endpoints
// --------------------------------------------------------------------------

/** `POST /api/flashcards/generate` request. */
export interface FlashcardGenerateRequest {
  paper_id: string;
  /** 5-60, default 20. */
  count?: number;
  /** Regenerate; upserts and PRESERVES the SRS state of surviving cards. */
  force?: boolean;
  settings_override?: AppSettings | null;
}

/**
 * `POST /api/flashcards/generate` response — same two-shape contract as
 * ChapterReportResponse, so the existing pollJob flow is reused verbatim:
 * - `cached: true` (200) → `deck` is ready, render it now.
 * - `cached: false` (202) → `pollJob(job.id)`, then `getDeck(paper_id)`.
 * Like a chapter job, this job has `stages: []` and `current_stage: null`;
 * show `percent` and the last `log` line.
 */
export interface FlashcardGenerateResponse {
  paper_id: string;
  cached: boolean;
  job?: IngestJob | null;
  deck?: Flashcard[] | null;
  progress?: DeckProgress | null;
}

/** `GET /api/flashcards/deck/{paper_id}?state=&starred=&due_only=&shuffle=`. */
export interface FlashcardDeckResponse {
  paper_id: string;
  paper_title: string;
  cards: Flashcard[];
  progress: DeckProgress;
  generated_at?: string | null;
  shuffled: boolean;
}

/**
 * `POST /api/flashcards/{card_id}/review` request. Send the rating only — the
 * SERVER computes the schedule. Any locally previewed interval is a hint for
 * the button label, never something you transmit.
 */
export interface FlashcardReviewRequest {
  rating: Rating;
  elapsed_ms?: number | null;
}

export interface FlashcardReviewResponse {
  card_id: string;
  rating: string;
  review_id: string;
  /**
   * How many OTHER cards to place before this one comes back, or null when the
   * card retires from this session ("got it").
   *
   * The 6/12 gaps deliberately have NO copy here — they live in
   * SESSION_REQUEUE_GAPS in deepvision/models/study.py and travel over the
   * wire, so there is nothing on this side that can drift out of sync.
   *
   * If fewer cards remain than this, put the card at the END of the session.
   * Never drop it: a card you just failed must be seen again before you stop.
   */
  requeue_after: number | null;
  /** Consecutive non-'again' ratings after this review. */
  strength: number;
  /** Distinct cards reviewed during the LOCAL day. */
  seen_today: number;
  /** Cards still to do today in the current scope. */
  due: number;
  progress: DeckProgress;
}

/** `POST /api/flashcards/{card_id}/star` — SET, not toggle (idempotent). */
export interface FlashcardStarRequest {
  starred: boolean;
}

export interface FlashcardStarResponse {
  card_id: string;
  starred: boolean;
  starred_count: number;
}

/**
 * `POST /api/flashcards/deck/{paper_id}/reset` — every card back to 'new'.
 * Card text is kept and the append-only review log is NOT deleted.
 */
export interface FlashcardResetResponse {
  paper_id: string;
  reset_count: number;
  progress: DeckProgress;
}

// --------------------------------------------------------------------------
// Study — quiz endpoints
// --------------------------------------------------------------------------

/** `POST /api/quiz/generate` request. */
export interface QuizGenerateRequest {
  paper_id: string;
  /** 3-30, default 10. */
  question_count?: number;
  /** Default: all three tiers. */
  difficulties?: Difficulty[];
  /** Default: all three kinds. */
  kinds?: QuestionKind[];
  /** Optional Chapter.id — scope the quiz to one chapter's page range. */
  chapter_id?: string | null;
  settings_override?: AppSettings | null;
}

/**
 * `POST /api/quiz/generate` response — always 202. There is no cache
 * short-circuit: generating makes a NEW quiz so past attempt history stays
 * meaningful. `pollJob(job.id)`, then fetch `quiz_id`.
 */
export interface QuizGenerateResponse {
  paper_id: string;
  /** The id the quiz WILL have; resolvable once the job reaches 'done'. */
  quiz_id: string;
  job: IngestJob;
}

/** `GET /api/quiz/paper/{paper_id}` — newest first. */
export interface QuizListResponse {
  paper_id: string;
  quizzes: QuizSummary[];
  count: number;
}

/**
 * `POST /api/quiz/{quiz_id}/attempt` request. Grading is server-side: send
 * answers, get back a QuizAttempt whose GradedAnswer[] finally carries the
 * correct answers and explanations. Omitted questions grade as incorrect.
 */
export interface QuizAttemptRequest {
  answers: SubmittedAnswer[];
  mode?: AttemptMode;
  /** Required iff mode === 'retry_missed'; the server validates the subset. */
  retry_of_attempt_id?: string | null;
  started_at?: string | null;
}

/**
 * `POST /api/quiz/{quiz_id}/check` — grade ONE answer, persist NOTHING.
 *
 * This is how the runner gives immediate per-question feedback.
 * `QuizQuestionPublic` has no answer field (deliberately), so knowing whether
 * an answer is right always costs a server round trip — and making those round
 * trips against `/attempt` would record one attempt PER QUESTION, so a
 * ten-question run would show up as ten attempts in history. Use `/check` while
 * the user is answering, then `/attempt` exactly once at the end.
 *
 * Both endpoints share one grader, so the live verdict always matches the
 * recorded one.
 */
export interface QuizCheckRequest {
  answer: SubmittedAnswer;
}

/** `POST /api/quiz/{quiz_id}/check` response — one graded answer, unrecorded. */
export interface QuizCheckResponse {
  quiz_id: string;
  graded: GradedAnswer;
}

/**
 * `POST /api/quiz/{quiz_id}/retry` — build the "only what I missed" run.
 * Returns a QuizPublic with just those questions, answers still hidden.
 */
export interface QuizRetryRequest {
  source_attempt_id: string;
}

/** `GET /api/quiz/{quiz_id}/attempts` — history, newest first, summaries only. */
export interface QuizAttemptHistoryResponse {
  quiz_id: string;
  paper_id: string;
  attempts: QuizAttemptSummary[];
  count: number;
  best_score?: number | null;
}

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

export interface ErrorResponse {
  detail: string;
  code?: string | null;
}
