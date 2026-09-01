/* ==========================================================================
   DeepVision API client — the ONLY place that knows fetch URLs. All pages
   must call these functions instead of hardcoding fetch() calls.
   Base URL: '/api' (proxied to http://localhost:8000 in dev, see
   vite.config.ts). CORS on the backend already allows http://localhost:5173.
   ========================================================================== */

import type {
  AppSettings,
  AttemptMode,
  ChaptersResponse,
  ChapterReportResponse,
  ChatRequest,
  ChatResponse,
  DeletePaperResponse,
  Difficulty,
  DueQueueResponse,
  ErrorResponse,
  FlashcardDeckResponse,
  FlashcardGenerateResponse,
  FlashcardResetResponse,
  FlashcardReviewResponse,
  FlashcardStarResponse,
  HealthResponse,
  IngestJob,
  IngestResponse,
  JobStatusResponse,
  MediaMetadata,
  PapersListResponse,
  PaperStudyStats,
  QuestionKind,
  QuizAttempt,
  QuizAttemptHistoryResponse,
  QuizCheckResponse,
  QuizGenerateResponse,
  QuizListResponse,
  QuizPublic,
  Rating,
  Report,
  RerunResponse,
  SearchDateRange,
  SearchResponse,
  SearchSort,
  SettingsResponse,
  StudyOverviewResponse,
  StudyPapersResponse,
  SubmittedAnswer,
} from "./types";

const BASE_URL = "/api";

/** Thrown for any non-2xx response; carries the parsed detail when available. */
export class ApiError extends Error {
  status: number;
  code?: string | null;

  constructor(status: number, detail: string, code?: string | null) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.code = code ?? null;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    // FormData bodies (multipart uploads) must NOT get a JSON content-type —
    // the browser needs to set its own `multipart/form-data; boundary=…`.
    const isFormData = typeof FormData !== "undefined" && init?.body instanceof FormData;
    res = await fetch(`${BASE_URL}${path}`, {
      headers:
        init?.body !== undefined && !isFormData ? { "Content-Type": "application/json" } : undefined,
      ...init,
    });
  } catch (err) {
    throw new ApiError(0, "Network error — is the backend running on :8000?");
  }

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    let code: string | null | undefined;
    try {
      const body = (await res.json()) as ErrorResponse;
      if (body?.detail) detail = body.detail;
      code = body?.code;
    } catch {
      // response wasn't JSON — keep the generic status text
    }
    throw new ApiError(res.status, detail, code);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

function qs(params: object): string {
  const usp = new URLSearchParams();
  for (const [k, v] of Object.entries(params as Record<string, unknown>)) {
    if (v !== undefined && v !== null && v !== "") usp.set(k, String(v));
  }
  const s = usp.toString();
  return s ? `?${s}` : "";
}

// --------------------------------------------------------------------------
// 1. GET /api/health
// --------------------------------------------------------------------------
export function health(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

// --------------------------------------------------------------------------
// 2. GET /api/search
// --------------------------------------------------------------------------
export interface SearchParams {
  q: string;
  category?: string;
  date_range?: SearchDateRange;
  sort?: SearchSort;
  limit?: number;
}

export function search(params: SearchParams): Promise<SearchResponse> {
  return request<SearchResponse>(`/search${qs(params)}`);
}

// --------------------------------------------------------------------------
// 3. POST /api/ingest
// --------------------------------------------------------------------------
export function ingest(
  arxivId: string,
  settingsOverride?: AppSettings
): Promise<IngestResponse> {
  return request<IngestResponse>("/ingest", {
    method: "POST",
    body: JSON.stringify({
      arxiv_id: arxivId,
      settings_override: settingsOverride ?? null,
    }),
  });
}

// --------------------------------------------------------------------------
// 3b. POST /api/ingest/upload — multipart PDF upload (Library "Upload PDF").
// Same IngestResponse shape as POST /ingest; feeds the same openIngest() /
// pollJob() flow used by Search.tsx. Field name MUST be "file" (optional
// "title" falls back to the filename stem on the backend).
// --------------------------------------------------------------------------
export function uploadPdf(file: File, title?: string): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  if (title) form.append("title", title);
  return request<IngestResponse>("/ingest/upload", {
    method: "POST",
    body: form,
  });
}

// --------------------------------------------------------------------------
// 4. GET /api/jobs/{job_id}  — unwrapped to the IngestJob for convenience
// --------------------------------------------------------------------------
export async function getJob(jobId: string): Promise<IngestJob> {
  const res = await request<JobStatusResponse>(`/jobs/${encodeURIComponent(jobId)}`);
  return res.job;
}

// --------------------------------------------------------------------------
// 5. GET /api/papers
// --------------------------------------------------------------------------
export function listPapers(): Promise<PapersListResponse> {
  return request<PapersListResponse>("/papers");
}

// --------------------------------------------------------------------------
// 6. DELETE /api/papers/{paper_id}
// --------------------------------------------------------------------------
export function deletePaper(paperId: string): Promise<DeletePaperResponse> {
  return request<DeletePaperResponse>(`/papers/${encodeURIComponent(paperId)}`, {
    method: "DELETE",
  });
}

// --------------------------------------------------------------------------
// 7. POST /api/papers/{paper_id}/rerun
// --------------------------------------------------------------------------
export function rerunPaper(paperId: string): Promise<RerunResponse> {
  return request<RerunResponse>(`/papers/${encodeURIComponent(paperId)}/rerun`, {
    method: "POST",
  });
}

// --------------------------------------------------------------------------
// 8. GET /api/report/{paper_id}
// --------------------------------------------------------------------------
export function getReport(paperId: string): Promise<Report> {
  return request<Report>(`/report/${encodeURIComponent(paperId)}`);
}

// --------------------------------------------------------------------------
// 8b. GET /api/papers/{paper_id}/chapters
// Cheap (no model calls) — safe to fetch as soon as the Report page mounts.
// --------------------------------------------------------------------------
export function getChapters(paperId: string): Promise<ChaptersResponse> {
  return request<ChaptersResponse>(`/papers/${encodeURIComponent(paperId)}/chapters`);
}

// --------------------------------------------------------------------------
// 8c. POST /api/report/{paper_id}/chapter
// 200 -> { cached: true, report } when already persisted (and force is not
// set); 202 -> { cached: false, job } when queued. Never resolves by title —
// chapterId must be a Chapter.id from getChapters().
// --------------------------------------------------------------------------
export function generateChapterReport(
  paperId: string,
  chapterId: string,
  force = false
): Promise<ChapterReportResponse> {
  return request<ChapterReportResponse>(`/report/${encodeURIComponent(paperId)}/chapter`, {
    method: "POST",
    body: JSON.stringify({ chapter_id: chapterId, force, settings_override: null }),
  });
}

// --------------------------------------------------------------------------
// 8d. GET /api/report/{paper_id}/chapter/{chapter_id}
// --------------------------------------------------------------------------
export function getChapterReport(paperId: string, chapterId: string): Promise<Report> {
  return request<Report>(
    `/report/${encodeURIComponent(paperId)}/chapter/${encodeURIComponent(chapterId)}`
  );
}

// --------------------------------------------------------------------------
// 9. POST /api/chat
// --------------------------------------------------------------------------
export function chat(req: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// --------------------------------------------------------------------------
// 11. GET /api/media/{chunk_id} — byte-serving URL (use directly as <img src>)
// --------------------------------------------------------------------------
export function mediaUrl(chunkId: string): string {
  return `${BASE_URL}/media/${encodeURIComponent(chunkId)}`;
}

// --------------------------------------------------------------------------
// 12. GET /api/media/{chunk_id}?meta=1
// --------------------------------------------------------------------------
export function mediaMeta(chunkId: string): Promise<MediaMetadata> {
  return request<MediaMetadata>(`/media/${encodeURIComponent(chunkId)}?meta=1`);
}

// --------------------------------------------------------------------------
// 13. GET /api/settings
// --------------------------------------------------------------------------
export function getSettings(): Promise<SettingsResponse> {
  return request<SettingsResponse>("/settings");
}

// --------------------------------------------------------------------------
// 14. PUT /api/settings
// --------------------------------------------------------------------------
export function putSettings(settings: AppSettings): Promise<SettingsResponse> {
  return request<SettingsResponse>("/settings", {
    method: "PUT",
    body: JSON.stringify({ settings }),
  });
}

// --------------------------------------------------------------------------
// Job-polling helper — polls GET /jobs/{id} until state is 'done' or 'error'.
// Used by the IngestionModal overlay; also available to page agents that
// need to track a job outside the modal (e.g. a library "processing" card).
// --------------------------------------------------------------------------
export interface PollJobOptions {
  /** Poll interval in ms. Default 900ms. */
  intervalMs?: number;
  /** Called with every fetched snapshot (including the final one). */
  onUpdate?: (job: IngestJob) => void;
  /** Abort polling early (e.g. on unmount). */
  signal?: AbortSignal;
}

/**
 * Polls `getJob(jobId)` on an interval until `state` is 'done' or 'error',
 * or until `signal` aborts. Resolves with the final job snapshot; rejects
 * only on repeated network failure or an aborted signal.
 */
export function pollJob(jobId: string, options: PollJobOptions = {}): Promise<IngestJob> {
  const { intervalMs = 900, onUpdate, signal } = options;

  return new Promise<IngestJob>((resolve, reject) => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const onAbort = () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      reject(new DOMException("Polling aborted", "AbortError"));
    };
    if (signal) {
      if (signal.aborted) return onAbort();
      signal.addEventListener("abort", onAbort, { once: true });
    }

    const tick = async () => {
      if (cancelled) return;
      try {
        const job = await getJob(jobId);
        if (cancelled) return;
        onUpdate?.(job);
        if (job.state === "done" || job.state === "error") {
          if (signal) signal.removeEventListener("abort", onAbort);
          resolve(job);
          return;
        }
        timer = setTimeout(tick, intervalMs);
      } catch (err) {
        if (cancelled) return;
        // Transient network hiccup — keep polling rather than failing hard.
        timer = setTimeout(tick, intervalMs);
      }
    };

    tick();
  });
}

/* ==========================================================================
   STUDY — the interactive study layer (Study nav item, between Report and
   Chat). Mirrors the routes documented in deepvision/api/routers/study.py,
   flashcards.py and quiz.py. See frontend/src/api/types.ts's STUDY block for
   the wire shapes.
   ========================================================================== */

// --------------------------------------------------------------------------
// Study — cross-paper surface
// --------------------------------------------------------------------------

/** GET /api/study/overview — the Study screen's header numbers. */
export function getStudyOverview(): Promise<StudyOverviewResponse> {
  return request<StudyOverviewResponse>("/study/overview");
}

export interface DueQueueParams {
  /** Default 50, max 200 (server-enforced). */
  limit?: number;
  /** Scope the queue to one paper — what the Report launch card deep-links to. */
  paper_id?: string | null;
  /** Default true. */
  include_new?: boolean;
  /** Applies AFTER the server's due_at ASC ordering + limit. Default false. */
  shuffle?: boolean;
}

/**
 * GET /api/study/due — THE cross-paper review queue, most overdue first.
 * `total_due` may exceed the number of cards actually returned (`count`).
 */
export function getDueQueue(params: DueQueueParams = {}): Promise<DueQueueResponse> {
  return request<DueQueueResponse>(`/study/due${qs(params)}`);
}

/** GET /api/study/papers — per-paper study stats in bulk (one call, not N). */
export function getStudyPapers(): Promise<StudyPapersResponse> {
  return request<StudyPapersResponse>("/study/papers");
}

/**
 * GET /api/study/papers/{paper_id}/stats — exactly what the Report page's two
 * launch cards render ("Flashcards · 24 cards · 6 due"). Exported under this
 * name deliberately — the Report-page builder imports it directly.
 */
export function getPaperStudyStats(paperId: string): Promise<PaperStudyStats> {
  return request<PaperStudyStats>(`/study/papers/${encodeURIComponent(paperId)}/stats`);
}

// --------------------------------------------------------------------------
// Study — flashcards
// --------------------------------------------------------------------------

export interface GenerateFlashcardsOptions {
  /** 5-60, default 20. */
  count?: number;
  /** Regenerate; upserts and PRESERVES surviving cards' SRS state. */
  force?: boolean;
  settingsOverride?: AppSettings;
}

/**
 * POST /api/flashcards/generate — same two-shape contract as
 * generateChapterReport: `cached: true` (200) -> `deck` is ready now;
 * `cached: false` (202) -> `pollJob(job.id)`, then `getDeck(paperId)`. Like a
 * chapter job, `stages` is [] and `current_stage` is null — show `percent`
 * and the last `log` line.
 */
export function generateFlashcards(
  paperId: string,
  options: GenerateFlashcardsOptions = {}
): Promise<FlashcardGenerateResponse> {
  return request<FlashcardGenerateResponse>("/flashcards/generate", {
    method: "POST",
    body: JSON.stringify({
      paper_id: paperId,
      count: options.count,
      force: options.force ?? false,
      settings_override: options.settingsOverride ?? null,
    }),
  });
}

export interface GetDeckParams {
  starred?: boolean;
  shuffle?: boolean;
}

/** GET /api/flashcards/deck/{paper_id} */
export function getDeck(paperId: string, params: GetDeckParams = {}): Promise<FlashcardDeckResponse> {
  return request<FlashcardDeckResponse>(
    `/flashcards/deck/${encodeURIComponent(paperId)}${qs(params)}`
  );
}

/**
 * POST /api/flashcards/{card_id}/review — SYNCHRONOUS, no model call. Send
 * only the rating; the SERVER computes the SM-2 schedule. Any locally
 * previewed interval (see types.ts's SM-2 constants) is a button-label hint
 * only — always render the response's `interval_days` / `due_in_minutes` as
 * the authoritative result.
 */
export function reviewCard(
  cardId: string,
  rating: Rating,
  elapsedMs?: number | null
): Promise<FlashcardReviewResponse> {
  return request<FlashcardReviewResponse>(`/flashcards/${encodeURIComponent(cardId)}/review`, {
    method: "POST",
    body: JSON.stringify({ rating, elapsed_ms: elapsedMs ?? null }),
  });
}

/** POST /api/flashcards/{card_id}/star — SETS the flag, does not toggle. */
export function starCard(cardId: string, starred: boolean): Promise<FlashcardStarResponse> {
  return request<FlashcardStarResponse>(`/flashcards/${encodeURIComponent(cardId)}/star`, {
    method: "POST",
    body: JSON.stringify({ starred }),
  });
}

/** POST /api/flashcards/deck/{paper_id}/reset — every card back to 'new'. */
export function resetDeck(paperId: string): Promise<FlashcardResetResponse> {
  return request<FlashcardResetResponse>(`/flashcards/deck/${encodeURIComponent(paperId)}/reset`, {
    method: "POST",
  });
}

// --------------------------------------------------------------------------
// Study — quiz
// --------------------------------------------------------------------------

export interface GenerateQuizOptions {
  /** 3-30, default 10. */
  questionCount?: number;
  /** Default: all three tiers. */
  difficulties?: Difficulty[];
  /** Default: all three kinds. */
  kinds?: QuestionKind[];
  /** Optional Chapter.id — scope the quiz to one chapter's page range. */
  chapterId?: string | null;
  settingsOverride?: AppSettings;
}

/**
 * POST /api/quiz/generate — ALWAYS 202 (no cache short-circuit: every call
 * makes a new quiz so past attempt history stays meaningful).
 * `pollJob(job.id)`, then `getQuiz(response.quiz_id)`.
 */
export function generateQuiz(
  paperId: string,
  options: GenerateQuizOptions = {}
): Promise<QuizGenerateResponse> {
  return request<QuizGenerateResponse>("/quiz/generate", {
    method: "POST",
    body: JSON.stringify({
      paper_id: paperId,
      question_count: options.questionCount,
      difficulties: options.difficulties,
      kinds: options.kinds,
      chapter_id: options.chapterId ?? null,
      settings_override: options.settingsOverride ?? null,
    }),
  });
}

/** GET /api/quiz/paper/{paper_id} — newest first, summaries only (cheap). */
export function listQuizzes(paperId: string): Promise<QuizListResponse> {
  return request<QuizListResponse>(`/quiz/paper/${encodeURIComponent(paperId)}`);
}

/** GET /api/quiz/{quiz_id} — every answer/explanation field stripped. */
export function getQuiz(quizId: string): Promise<QuizPublic> {
  return request<QuizPublic>(`/quiz/${encodeURIComponent(quizId)}`);
}

/**
 * POST /api/quiz/{quiz_id}/check — grade ONE answer, RECORD NOTHING.
 *
 * Use this for immediate per-question feedback while the user is still working
 * through the quiz, then call `submitQuizAttempt` ONCE at the end with the full
 * answer list. Do not drive live feedback off `submitQuizAttempt`: it persists
 * an attempt every call, so a ten-question run would leave ten rows in history.
 * Both endpoints run the same grader, so the verdict shown here is exactly the
 * verdict the recorded attempt will produce.
 */
export function checkQuizAnswer(
  quizId: string,
  answer: SubmittedAnswer
): Promise<QuizCheckResponse> {
  return request<QuizCheckResponse>(`/quiz/${encodeURIComponent(quizId)}/check`, {
    method: "POST",
    body: JSON.stringify({ answer }),
  });
}

/**
 * POST /api/quiz/{quiz_id}/attempt — graded SERVER-SIDE, synchronously, and
 * PERSISTED as one row of attempt history. Call it once per finished run.
 * The response's `answers` (GradedAnswer[]) is the only place correct
 * answers/explanations ever appear.
 */
export function submitQuizAttempt(
  quizId: string,
  answers: SubmittedAnswer[],
  options: { mode?: AttemptMode; retryOfAttemptId?: string | null; startedAt?: string } = {}
): Promise<QuizAttempt> {
  return request<QuizAttempt>(`/quiz/${encodeURIComponent(quizId)}/attempt`, {
    method: "POST",
    body: JSON.stringify({
      answers,
      mode: options.mode ?? "full",
      retry_of_attempt_id: options.retryOfAttemptId ?? null,
      started_at: options.startedAt ?? null,
    }),
  });
}

/**
 * POST /api/quiz/{quiz_id}/retry — "retry only what I missed"; returns a
 * QuizPublic scoped to `sourceAttemptId`'s missed questions, answers hidden.
 */
export function retryQuizMissed(quizId: string, sourceAttemptId: string): Promise<QuizPublic> {
  return request<QuizPublic>(`/quiz/${encodeURIComponent(quizId)}/retry`, {
    method: "POST",
    body: JSON.stringify({ source_attempt_id: sourceAttemptId }),
  });
}

/** GET /api/quiz/{quiz_id}/attempts — history, newest first, summaries only. */
export function getQuizAttempts(quizId: string): Promise<QuizAttemptHistoryResponse> {
  return request<QuizAttemptHistoryResponse>(`/quiz/${encodeURIComponent(quizId)}/attempts`);
}

/** GET /api/quiz/attempts/{attempt_id} — the full graded attempt, with explanations. */
export function getQuizAttempt(attemptId: string): Promise<QuizAttempt> {
  return request<QuizAttempt>(`/quiz/attempts/${encodeURIComponent(attemptId)}`);
}
