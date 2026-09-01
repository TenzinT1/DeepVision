import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  ApiError,
  generateFlashcards,
  generateQuiz,
  getQuiz,
  getQuizAttempts,
  getStudyOverview,
  getStudyPapers,
  listPapers,
  listQuizzes,
  pollJob,
} from "../api/client";
import { parseServerDate } from "../api/types";
import type {
  IngestJob,
  PaperMeta,
  PaperStudyStats,
  QuizAttempt,
  QuizAttemptSummary,
  QuizPublic,
  QuizSummary,
  StudyOverviewResponse,
} from "../api/types";
import { DueQueue } from "./study/DueQueue";
import { FlashcardDrill } from "./study/FlashcardDrill";
import QuizResults from "./study/QuizResults";
import QuizRunner from "./study/QuizRunner";

/* --------------------------------------------------------------------------
   Study — the top-level study nav item. Three sections: the cross-paper Due
   Today queue (the reason Study is top-level at all), per-paper Decks, and
   per-paper Quizzes. `?paper=<id>` (from a Report page launch card)
   pre-filters; `?tab=flashcards|quiz` picks the starting tab.
   -------------------------------------------------------------------------- */

type StudyTab = "due" | "decks" | "quizzes";

function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Something went wrong.";
}

function tabToParam(tab: StudyTab): string {
  return tab === "decks" ? "flashcards" : tab === "quizzes" ? "quiz" : "due";
}

function paramToTab(raw: string | null, hasPaperFilter: boolean): StudyTab {
  if (raw === "flashcards" || raw === "decks") return "decks";
  if (raw === "quiz" || raw === "quizzes") return "quizzes";
  if (raw === "due") return "due";
  return hasPaperFilter ? "decks" : "due";
}

function zeroStats(p: PaperMeta): PaperStudyStats {
  return {
    paper_id: p.id,
    paper_title: p.title,
    paper_label: p.arxiv_label,
    has_deck: false,
    cards_total: 0,
    cards_starred: 0,
    deck_generated_at: null,
    quiz_count: 0,
    quiz_attempt_count: 0,
    quiz_best_score: null,
    quiz_last_attempted_at: null,
  };
}

function formatWhen(iso?: string | null): string | null {
  if (!iso) return null;
  // parseServerDate, not `new Date`: the backend's timestamps are naive UTC and
  // JS would read an offset-less ISO string as local time (wrong day off-UTC).
  const d = parseServerDate(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function Spinner() {
  return (
    <span
      style={{
        display: "inline-block",
        width: 14,
        height: 14,
        borderRadius: "50%",
        border: "2px solid var(--border-strong)",
        borderTopColor: "var(--accent)",
        animation: "dv-spin .7s linear infinite",
      }}
    />
  );
}

interface GenState {
  percent: number;
  logLine: string | null;
  error: string | null;
}

function GenerationBar({ gen }: { gen: GenState }) {
  return (
    <div style={{ marginTop: 10 }}>
      {gen.error ? (
        <div style={{ fontSize: 12, color: "var(--red)" }}>{gen.error}</div>
      ) : (
        <>
          <div style={{ height: 6, borderRadius: 20, background: "var(--surface-3)", overflow: "hidden" }}>
            <div
              style={{
                height: "100%",
                borderRadius: 20,
                background: "var(--accent)",
                width: `${gen.percent}%`,
                transition: "width .3s ease",
              }}
            />
          </div>
          <div className="dv-mono" style={{ fontSize: 10.5, color: "var(--fg-muted)", marginTop: 5 }}>
            {gen.percent}%{gen.logLine ? ` — ${gen.logLine}` : ""}
          </div>
        </>
      )}
    </div>
  );
}

const PAGE_STYLE: React.CSSProperties = { maxWidth: 980, margin: "0 auto", padding: "34px 28px 80px" };

const cardBase: React.CSSProperties = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: 14,
  padding: "18px 20px",
};

const primaryBtn: React.CSSProperties = {
  fontSize: 12.5,
  fontWeight: 600,
  color: "var(--accent-fg)",
  background: "var(--accent)",
  border: "none",
  borderRadius: 9,
  padding: "9px 16px",
};

const ghostBtn: React.CSSProperties = {
  fontSize: 12.5,
  fontWeight: 500,
  color: "var(--fg-muted)",
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  borderRadius: 9,
  padding: "9px 16px",
};

export default function StudyPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const paperFilter = searchParams.get("paper");
  const tab = paramToTab(searchParams.get("tab"), !!paperFilter);

  const setTab = useCallback(
    (next: StudyTab) => {
      setSearchParams(
        (prev) => {
          const p = new URLSearchParams(prev);
          p.set("tab", tabToParam(next));
          return p;
        },
        { replace: true }
      );
    },
    [setSearchParams]
  );

  const clearPaperFilter = useCallback(() => {
    setSearchParams(
      (prev) => {
        const p = new URLSearchParams(prev);
        p.delete("paper");
        return p;
      },
      { replace: true }
    );
  }, [setSearchParams]);

  // -- overview + library data --------------------------------------------
  const [overview, setOverview] = useState<StudyOverviewResponse | null>(null);
  const [papers, setPapers] = useState<PaperMeta[]>([]);
  const [studyStats, setStudyStats] = useState<Map<string, PaperStudyStats>>(new Map());
  const [loadError, setLoadError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  const refresh = useCallback(() => {
    setLoadError(null);
    Promise.all([getStudyOverview(), listPapers(), getStudyPapers()])
      .then(([ov, papersRes, statsRes]) => {
        setOverview(ov);
        setPapers(papersRes.papers.filter((p) => p.status === "ready"));
        setStudyStats(new Map(statsRes.papers.map((s) => [s.paper_id, s])));
      })
      .catch((err) => setLoadError(errorMessage(err)));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh, reloadKey]);

  const readyPapers: PaperStudyStats[] = useMemo(
    () => papers.map((p) => studyStats.get(p.id) ?? zeroStats(p)),
    [papers, studyStats]
  );

  const paperFilterLabel = useMemo(() => {
    if (!paperFilter) return null;
    return readyPapers.find((s) => s.paper_id === paperFilter)?.paper_title ?? paperFilter;
  }, [paperFilter, readyPapers]);

  const decksToShow = paperFilter ? readyPapers.filter((s) => s.paper_id === paperFilter) : readyPapers;
  const quizzesToShow = paperFilter ? readyPapers.filter((s) => s.paper_id === paperFilter) : readyPapers;

  // -- Due Today session toggles + drill -----------------------------------
  const [shuffle, setShuffle] = useState(false);
  const [includeNew, setIncludeNew] = useState(true);
  const [drilling, setDrilling] = useState(false);

  const exitDrill = useCallback(() => {
    setDrilling(false);
    setReloadKey((k) => k + 1);
  }, []);

  /** "Review due (N)" on a deck card — filter the URL to this paper, then drop into the drill. */
  const reviewPaperDeck = useCallback(
    (paperId: string) => {
      setSearchParams(
        (prev) => {
          const p = new URLSearchParams(prev);
          p.set("paper", paperId);
          p.set("tab", "due");
          return p;
        },
        { replace: true }
      );
      setDrilling(true);
    },
    [setSearchParams]
  );

  // -- deck / quiz generation (job-based, inline progress) -----------------
  const [deckGen, setDeckGen] = useState<Record<string, GenState>>({});
  const [quizGen, setQuizGen] = useState<Record<string, GenState>>({});
  const genOpRef = useRef<Record<string, number>>({});

  const startDeckGeneration = useCallback(
    (paperId: string, force = false) => {
      const opId = (genOpRef.current[`deck:${paperId}`] = (genOpRef.current[`deck:${paperId}`] ?? 0) + 1);
      setDeckGen((s) => ({ ...s, [paperId]: { percent: 0, logLine: null, error: null } }));
      generateFlashcards(paperId, { force })
        .then(async (res) => {
          if (genOpRef.current[`deck:${paperId}`] !== opId) return;
          if (res.cached) {
            setDeckGen((s) => {
              const n = { ...s };
              delete n[paperId];
              return n;
            });
            setReloadKey((k) => k + 1);
            return;
          }
          if (res.job) {
            const finalJob: IngestJob = await pollJob(res.job.id, {
              onUpdate: (job) => {
                if (genOpRef.current[`deck:${paperId}`] !== opId) return;
                setDeckGen((s) => ({
                  ...s,
                  [paperId]: {
                    percent: job.percent,
                    logLine: job.log.length ? job.log[job.log.length - 1].m : null,
                    error: null,
                  },
                }));
              },
            });
            if (genOpRef.current[`deck:${paperId}`] !== opId) return;
            if (finalJob.state === "error") {
              throw new Error(finalJob.error_message || "Flashcard generation failed.");
            }
            setDeckGen((s) => {
              const n = { ...s };
              delete n[paperId];
              return n;
            });
            setReloadKey((k) => k + 1);
          }
        })
        .catch((err) => {
          if (genOpRef.current[`deck:${paperId}`] !== opId) return;
          setDeckGen((s) => ({ ...s, [paperId]: { percent: 100, logLine: null, error: errorMessage(err) } }));
        });
    },
    []
  );

  const startQuizGeneration = useCallback((paperId: string) => {
    const opId = (genOpRef.current[`quiz:${paperId}`] = (genOpRef.current[`quiz:${paperId}`] ?? 0) + 1);
    setQuizGen((s) => ({ ...s, [paperId]: { percent: 0, logLine: null, error: null } }));
    generateQuiz(paperId)
      .then(async (res) => {
        if (genOpRef.current[`quiz:${paperId}`] !== opId) return;
        const finalJob: IngestJob = await pollJob(res.job.id, {
          onUpdate: (job) => {
            if (genOpRef.current[`quiz:${paperId}`] !== opId) return;
            setQuizGen((s) => ({
              ...s,
              [paperId]: {
                percent: job.percent,
                logLine: job.log.length ? job.log[job.log.length - 1].m : null,
                error: null,
              },
            }));
          },
        });
        if (genOpRef.current[`quiz:${paperId}`] !== opId) return;
        if (finalJob.state === "error") {
          throw new Error(finalJob.error_message || "Quiz generation failed.");
        }
        setQuizGen((s) => {
          const n = { ...s };
          delete n[paperId];
          return n;
        });
        // Drop the cached quiz list so the quiz just generated actually shows
        // up in an already-expanded panel.
        setQuizzesByPaper((s) => {
          const n = { ...s };
          delete n[paperId];
          return n;
        });
        setReloadKey((k) => k + 1);
      })
      .catch((err) => {
        if (genOpRef.current[`quiz:${paperId}`] !== opId) return;
        setQuizGen((s) => ({ ...s, [paperId]: { percent: 100, logLine: null, error: errorMessage(err) } }));
      });
  }, []);

  // -- quiz list / attempt-history expansion (read-only browse) -----------
  const [expandedQuizPaper, setExpandedQuizPaper] = useState<string | null>(null);
  const [quizzesByPaper, setQuizzesByPaper] = useState<Record<string, QuizSummary[]>>({});
  const [quizListLoading, setQuizListLoading] = useState<string | null>(null);
  const [expandedQuiz, setExpandedQuiz] = useState<string | null>(null);
  const [attemptsByQuiz, setAttemptsByQuiz] = useState<Record<string, QuizAttemptSummary[]>>({});

  const toggleQuizPaper = useCallback(
    (paperId: string) => {
      if (expandedQuizPaper === paperId) {
        setExpandedQuizPaper(null);
        return;
      }
      setExpandedQuizPaper(paperId);
      setExpandedQuiz(null);
      if (!quizzesByPaper[paperId]) {
        setQuizListLoading(paperId);
        listQuizzes(paperId)
          .then((res) => setQuizzesByPaper((s) => ({ ...s, [paperId]: res.quizzes })))
          .catch((err) => setLoadError(errorMessage(err)))
          .finally(() => setQuizListLoading(null));
      }
    },
    [expandedQuizPaper, quizzesByPaper]
  );

  /* -- taking a quiz ------------------------------------------------------
     QuizRunner and QuizResults are pure presentational components: they take
     an already-fetched QuizPublic / QuizAttempt and never fetch by id. This
     page owns the session, which is a three-phase state machine:
       loading  -> getQuiz(id) is in flight
       running  -> QuizRunner (per-question /check, one /attempt at the end)
       results  -> QuizResults, which can hand back a scoped retry QuizPublic
                   and put us straight back into `running`.
     Without this, a generated quiz could be listed but never taken.
     ---------------------------------------------------------------------- */
  type QuizSession =
    | { phase: "loading"; quizId: string }
    | { phase: "running"; quiz: QuizPublic }
    | { phase: "results"; quiz: QuizPublic; attempt: QuizAttempt };

  const [quizSession, setQuizSession] = useState<QuizSession | null>(null);
  const [quizSessionError, setQuizSessionError] = useState<string | null>(null);

  const startQuiz = useCallback((quizId: string) => {
    setQuizSessionError(null);
    setQuizSession({ phase: "loading", quizId });
    getQuiz(quizId)
      .then((quiz) => setQuizSession({ phase: "running", quiz }))
      .catch((err) => {
        setQuizSession(null);
        setQuizSessionError(errorMessage(err));
      });
  }, []);

  const exitQuiz = useCallback((quizId: string | null) => {
    setQuizSession(null);
    if (quizId) {
      // An attempt may have landed — drop the cached history for this quiz and
      // refresh the per-paper stats so counts/best score are not stale.
      setAttemptsByQuiz((s) => {
        const n = { ...s };
        delete n[quizId];
        return n;
      });
      setQuizzesByPaper({});
    }
    setReloadKey((k) => k + 1);
  }, []);

  const toggleQuizAttempts = useCallback(
    (quizId: string) => {
      if (expandedQuiz === quizId) {
        setExpandedQuiz(null);
        return;
      }
      setExpandedQuiz(quizId);
      if (!attemptsByQuiz[quizId]) {
        getQuizAttempts(quizId)
          .then((res) => setAttemptsByQuiz((s) => ({ ...s, [quizId]: res.attempts })))
          .catch((err) => setLoadError(errorMessage(err)));
      }
    },
    [expandedQuiz, attemptsByQuiz]
  );

  if (quizSession) {
    const quizId =
      quizSession.phase === "loading" ? quizSession.quizId : quizSession.quiz.id;
    return (
      <div style={PAGE_STYLE}>
        {quizSession.phase === "loading" && (
          <div style={{ display: "flex", alignItems: "center", gap: 9, fontSize: 13, color: "var(--fg-muted)" }}>
            <Spinner /> Loading quiz…
          </div>
        )}
        {quizSession.phase === "running" && (
          <QuizRunner
            quiz={quizSession.quiz}
            onComplete={(attempt) =>
              setQuizSession({ phase: "results", quiz: quizSession.quiz, attempt })
            }
            onExit={() => exitQuiz(quizId)}
          />
        )}
        {quizSession.phase === "results" && (
          <QuizResults
            attempt={quizSession.attempt}
            quiz={quizSession.quiz}
            onRetryMissed={(retryQuiz) => setQuizSession({ phase: "running", quiz: retryQuiz })}
            onExit={() => exitQuiz(quizId)}
          />
        )}
      </div>
    );
  }

  if (drilling) {
    return (
      <FlashcardDrill
        paperId={paperFilter}
        shuffle={shuffle}
        includeNew={includeNew}
        onExit={exitDrill}
      />
    );
  }

  return (
    <div style={PAGE_STYLE}>
      {overview && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
            gap: 12,
            marginBottom: 24,
          }}
        >
          {[
            // "Cards due", not bare "Due": it sits next to "Quizzes due" and
            // the two count different things — cards reset daily, quizzes are
            // a backlog of ones never attempted.
            { label: "Cards due", value: overview.due },
            { label: "Quizzes due", value: overview.quizzes_due },
            { label: "Decks", value: overview.papers_with_decks },
            { label: "Quizzes", value: overview.quiz_count },
          ].map((s) => (
            <div key={s.label} style={{ ...cardBase, padding: "14px 16px" }}>
              <div className="dv-mono" style={{ fontSize: 10.5, color: "var(--fg-subtle)", letterSpacing: ".03em" }}>
                {s.label.toUpperCase()}
              </div>
              <div style={{ fontSize: 22, fontWeight: 700, marginTop: 4 }}>{s.value}</div>
            </div>
          ))}
        </div>
      )}

      {loadError && (
        <div
          style={{
            fontSize: 12.5,
            color: "var(--red)",
            background: "var(--red-soft)",
            border: "1px solid var(--red)",
            borderRadius: 10,
            padding: "10px 14px",
            marginBottom: 18,
          }}
        >
          {loadError}
        </div>
      )}

      {quizSessionError && (
        <div
          style={{
            fontSize: 12.5,
            color: "var(--red)",
            background: "var(--red-soft)",
            border: "1px solid var(--red)",
            borderRadius: 10,
            padding: "10px 14px",
            marginBottom: 18,
          }}
        >
          {quizSessionError}
        </div>
      )}

      {/* tabs */}
      <div style={{ display: "flex", gap: 6, borderBottom: "1px solid var(--border)", marginBottom: 22 }}>
        {(
          [
            { key: "due", label: "Due Today" },
            { key: "decks", label: "Decks" },
            { key: "quizzes", label: "Quizzes" },
          ] as { key: StudyTab; label: string }[]
        ).map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            style={{
              fontSize: 13.5,
              fontWeight: tab === t.key ? 600 : 500,
              color: tab === t.key ? "var(--accent)" : "var(--fg-muted)",
              background: "transparent",
              border: "none",
              borderBottom: tab === t.key ? "2px solid var(--accent)" : "2px solid transparent",
              padding: "10px 4px",
              marginRight: 20,
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "due" && (
        <DueQueue
          paperFilter={paperFilter}
          paperFilterLabel={paperFilterLabel}
          onClearPaperFilter={clearPaperFilter}
          shuffle={shuffle}
          includeNew={includeNew}
          onShuffleChange={setShuffle}
          onIncludeNewChange={setIncludeNew}
          onStartReview={() => setDrilling(true)}
          reloadKey={reloadKey}
        />
      )}

      {tab === "decks" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {paperFilter && (
            <FilterChip label={paperFilterLabel ?? paperFilter} onClear={clearPaperFilter} />
          )}
          {decksToShow.length === 0 && (
            <div style={{ fontSize: 13, color: "var(--fg-subtle)" }}>No ingested papers yet.</div>
          )}
          {decksToShow.map((s) => {
            const gen = deckGen[s.paper_id];
            const generatedLabel = formatWhen(s.deck_generated_at);
            return (
              <div key={s.paper_id} style={cardBase}>
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontSize: 14.5, fontWeight: 600 }}>{s.paper_title}</div>
                    <div className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)", marginTop: 3 }}>
                      {generatedLabel ? `generated ${generatedLabel}` : ""}
                    </div>
                  </div>
                  <div style={{ flex: "none", display: "flex", gap: 8 }}>
                    {s.has_deck && (
                      <button
                        style={ghostBtn}
                        disabled={!!gen}
                        onClick={() => startDeckGeneration(s.paper_id, true)}
                      >
                        Regenerate
                      </button>
                    )}
                    <button
                      style={primaryBtn}
                      disabled={!!gen || (s.has_deck && s.cards_total === 0)}
                      onClick={() =>
                        s.has_deck ? reviewPaperDeck(s.paper_id) : startDeckGeneration(s.paper_id, false)
                      }
                    >
                      {s.has_deck ? `Study (${s.cards_total})` : "Generate flashcards"}
                    </button>
                  </div>
                </div>

                {gen && <GenerationBar gen={gen} />}

                {s.has_deck && !gen && (
                  <div style={{ display: "flex", gap: 18, marginTop: 14 }}>
                    <StatMini label="Total" value={s.cards_total} accent />
                    <StatMini label="Starred" value={s.cards_starred} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {tab === "quizzes" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {paperFilter && (
            <FilterChip label={paperFilterLabel ?? paperFilter} onClear={clearPaperFilter} />
          )}
          {quizzesToShow.length === 0 && (
            <div style={{ fontSize: 13, color: "var(--fg-subtle)" }}>No ingested papers yet.</div>
          )}
          {quizzesToShow.map((s) => {
            const gen = quizGen[s.paper_id];
            const expanded = expandedQuizPaper === s.paper_id;
            return (
              <div key={s.paper_id} style={cardBase}>
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontSize: 14.5, fontWeight: 600 }}>{s.paper_title}</div>
                    <div className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)", marginTop: 3 }}>
                      {s.quiz_count > 0
                        ? `${s.quiz_count} quiz${s.quiz_count === 1 ? "" : "zes"}${
                            s.quiz_best_score != null ? ` · best ${Math.round(s.quiz_best_score * 100)}%` : ""
                          }`
                        : ""}
                    </div>
                  </div>
                  <div style={{ flex: "none", display: "flex", gap: 8 }}>
                    {s.quiz_count > 0 && (
                      <button style={ghostBtn} onClick={() => toggleQuizPaper(s.paper_id)}>
                        {expanded ? "Hide quizzes" : "View quizzes"}
                      </button>
                    )}
                    <button style={primaryBtn} disabled={!!gen} onClick={() => startQuizGeneration(s.paper_id)}>
                      {s.quiz_count > 0 ? "Generate new quiz" : "Generate quiz"}
                    </button>
                  </div>
                </div>

                {gen && <GenerationBar gen={gen} />}

                {expanded && (
                  <div style={{ marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
                    {quizListLoading === s.paper_id && (
                      <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, color: "var(--fg-muted)" }}>
                        <Spinner /> Loading quizzes…
                      </div>
                    )}
                    {(quizzesByPaper[s.paper_id] ?? []).map((q) => (
                      <div key={q.id} style={{ padding: "10px 0", borderBottom: "1px solid var(--border)" }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                          <div style={{ minWidth: 0 }}>
                            <div style={{ fontSize: 13, fontWeight: 500 }}>{q.title}</div>
                            <div className="dv-mono" style={{ fontSize: 10.5, color: "var(--fg-subtle)", marginTop: 2 }}>
                              {q.question_count} questions · {q.attempt_count} attempt{q.attempt_count === 1 ? "" : "s"}
                              {q.best_score != null ? ` · best ${Math.round(q.best_score * 100)}%` : " · not attempted"}
                            </div>
                          </div>
                          <div style={{ flex: "none", display: "flex", gap: 8 }}>
                            {q.attempt_count > 0 && (
                              <button
                                style={{ ...ghostBtn, padding: "6px 12px", fontSize: 11.5 }}
                                onClick={() => toggleQuizAttempts(q.id)}
                              >
                                {expandedQuiz === q.id ? "Hide history" : "History"}
                              </button>
                            )}
                            <button
                              style={{ ...primaryBtn, padding: "6px 14px", fontSize: 11.5 }}
                              onClick={() => startQuiz(q.id)}
                            >
                              {q.attempt_count > 0 ? "Retake" : "Take quiz"}
                            </button>
                          </div>
                        </div>
                        {expandedQuiz === q.id && (
                          <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
                            {(attemptsByQuiz[q.id] ?? []).map((a) => (
                              <div
                                key={a.id}
                                style={{
                                  display: "flex",
                                  justifyContent: "space-between",
                                  fontSize: 11.5,
                                  color: "var(--fg-muted)",
                                  background: "var(--surface-2)",
                                  borderRadius: 7,
                                  padding: "6px 10px",
                                }}
                              >
                                <span>
                                  {a.correct}/{a.total} ({Math.round(a.score * 100)}%)
                                  {a.mode === "retry_missed" ? " · retry" : ""}
                                </span>
                                <span className="dv-mono">{formatWhen(a.finished_at)}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function FilterChip({ label, onClear }: { label: string; onClear: () => void }) {
  return (
    <div
      style={{
        alignSelf: "flex-start",
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        fontSize: 12.5,
        color: "var(--accent)",
        background: "var(--accent-soft)",
        border: "1px solid var(--accent-border)",
        borderRadius: 20,
        padding: "5px 6px 5px 12px",
      }}
    >
      Filtered to <strong style={{ fontWeight: 600 }}>{label}</strong>
      <button
        onClick={onClear}
        aria-label="Clear paper filter"
        style={{
          width: 20,
          height: 20,
          borderRadius: "50%",
          border: "none",
          background: "var(--surface)",
          color: "var(--accent)",
          fontSize: 12,
          lineHeight: 1,
        }}
      >
        ✕
      </button>
    </div>
  );
}

function StatMini({ label, value, accent }: { label: string; value: number; accent?: boolean }) {
  return (
    <div>
      <div style={{ fontSize: 17, fontWeight: 700, color: accent ? "var(--accent)" : "var(--fg)" }}>{value}</div>
      <div className="dv-mono" style={{ fontSize: 10, color: "var(--fg-subtle)", marginTop: 2 }}>
        {label.toUpperCase()}
      </div>
    </div>
  );
}
