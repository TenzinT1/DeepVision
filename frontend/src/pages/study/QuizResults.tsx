import React, { useEffect, useMemo, useState } from "react";
import { ApiError, getQuizAttempts, retryQuizMissed } from "../../api/client";
import type { AttemptMode, Difficulty, QuizAttempt, QuizAttemptSummary, QuizPublic } from "../../api/types";
import { DIFFICULTIES, parseServerDate } from "../../api/types";
import { renderInline } from "../report/inlineMarkdown";
import { CitationChip, DifficultyBadge, formatKind } from "./QuizRunner";

/** Same rationale as QuizRunner's `md()`: quiz text has `**bold**` markers
 *  but no `Citation[]` of its own, so `[n]`-shaped text falls back to
 *  literal text rather than a dead clickable marker. */
function md(text: string): React.ReactNode {
  return renderInline(text, [], () => {});
}

function formatMode(mode: AttemptMode): string {
  return mode === "retry_missed" ? "Retry · missed only" : "Full attempt";
}

function pct(score: number): string {
  return `${Math.round(score * 100)}%`;
}

function formatDate(iso: string): string {
  try {
    // parseServerDate, not `new Date`: attempt timestamps are naive UTC, and JS
    // reads an offset-less ISO string as local time — an attempt finished a
    // minute ago would render hours (and often a day) off.
    return parseServerDate(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function Spinner() {
  return (
    <span
      style={{
        display: "inline-block",
        width: 13,
        height: 13,
        borderRadius: "50%",
        border: "2px solid var(--border-strong)",
        borderTopColor: "var(--accent)",
        animation: "dv-spin .7s linear infinite",
      }}
    />
  );
}

function ScoreRing({ score }: { score: number }) {
  const p = Math.max(0, Math.min(1, score));
  const color = p >= 0.8 ? "var(--green)" : p >= 0.5 ? "var(--amber)" : "var(--red)";
  const size = 84;
  const stroke = 8;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke="var(--surface-3)"
        strokeWidth={stroke}
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeDasharray={c}
        strokeDashoffset={c * (1 - p)}
        strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ transition: "stroke-dashoffset .4s ease" }}
      />
      <text
        x="50%"
        y="52%"
        textAnchor="middle"
        dominantBaseline="middle"
        style={{ fontSize: 18, fontWeight: 700, fill: "var(--fg)" }}
      >
        {pct(p)}
      </text>
    </svg>
  );
}

export interface QuizResultsProps {
  attempt: QuizAttempt;
  quiz: QuizPublic;
  /** Called with the new scoped QuizPublic once "retry missed" has been started. */
  onRetryMissed: (retryQuiz: QuizPublic) => void;
  onExit?: () => void;
}

export default function QuizResults({ attempt, quiz, onRetryMissed, onExit }: QuizResultsProps) {
  const [history, setHistory] = useState<QuizAttemptSummary[] | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);
  const [retryError, setRetryError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getQuizAttempts(quiz.id)
      .then((res) => {
        if (cancelled) return;
        setHistory(res.attempts);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setHistoryError(err instanceof ApiError ? err.message : "Couldn't load attempt history.");
      });
    return () => {
      cancelled = true;
    };
  }, [quiz.id, attempt.id]);

  const byDifficulty = useMemo(() => {
    const map: Record<Difficulty, { correct: number; total: number }> = {
      recall: { correct: 0, total: 0 },
      understand: { correct: 0, total: 0 },
      apply: { correct: 0, total: 0 },
    };
    for (const a of attempt.answers) {
      map[a.difficulty].total += 1;
      if (a.is_correct) map[a.difficulty].correct += 1;
    }
    return map;
  }, [attempt]);

  const missed = useMemo(() => attempt.answers.filter((a) => !a.is_correct), [attempt]);

  async function handleRetryMissed() {
    if (retrying || missed.length === 0) return;
    setRetrying(true);
    setRetryError(null);
    try {
      const scoped = await retryQuizMissed(quiz.id, attempt.id);
      onRetryMissed(scoped);
    } catch (err) {
      setRetryError(err instanceof ApiError ? err.message : "Couldn't start the retry.");
    } finally {
      setRetrying(false);
    }
  }

  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 20,
          padding: "20px 22px",
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 13,
        }}
      >
        <ScoreRing score={attempt.score} />
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 16, fontWeight: 700 }}>
            {attempt.correct} / {attempt.total} correct
          </div>
          <div style={{ marginTop: 4, fontSize: 12.5, color: "var(--fg-subtle)" }}>
            {quiz.title} · {formatMode(attempt.mode)}
          </div>
          {attempt.mode === "retry_missed" && (
            <div style={{ marginTop: 4, fontSize: 11.5, color: "var(--amber)" }}>
              Scored over the missed subset only — not comparable to a full attempt.
            </div>
          )}
        </div>
        {missed.length > 0 ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6 }}>
            <button
              onClick={handleRetryMissed}
              disabled={retrying}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontSize: 13,
                fontWeight: 600,
                color: "var(--accent-fg)",
                background: "var(--accent)",
                border: "none",
                borderRadius: 9,
                padding: "9px 16px",
                opacity: retrying ? 0.6 : 1,
              }}
            >
              {retrying && <Spinner />}
              Retry only what I missed ({missed.length})
            </button>
            {retryError && <div style={{ fontSize: 11.5, color: "var(--red)" }}>{retryError}</div>}
          </div>
        ) : (
          <div
            className="dv-mono"
            style={{
              fontSize: 11.5,
              fontWeight: 700,
              color: "var(--green)",
              background: "var(--green-soft)",
              padding: "6px 12px",
              borderRadius: 8,
            }}
          >
            PERFECT
          </div>
        )}
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
        {DIFFICULTIES.map((d) => {
          const stat = byDifficulty[d];
          return (
            <div
              key={d}
              style={{
                flex: 1,
                padding: "12px 14px",
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 11,
              }}
            >
              <DifficultyBadge difficulty={d} />
              <div style={{ marginTop: 8, fontSize: 17, fontWeight: 700 }}>
                {stat.total === 0 ? "—" : `${stat.correct}/${stat.total}`}
              </div>
            </div>
          );
        })}
      </div>

      {missed.length > 0 && (
        <div style={{ marginTop: 22 }}>
          <div style={{ fontSize: 13.5, fontWeight: 700, marginBottom: 10 }}>
            Missed questions ({missed.length})
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {missed.map((a) => {
              // Short-answer "your answer" is the user's own typed text, not
              // server content — render it verbatim, never as markdown.
              // Every other string here (prompt, option text, correct
              // answer, explanation) comes from the quiz/answer key, so it
              // gets the same **bold** treatment as the running quiz.
              const isShortAnswer = a.kind === "short_answer";
              const yourAnswer = isShortAnswer
                ? a.answer_text || "(no answer)"
                : a.options.find((o) => o.id === a.selected_option_id)?.text || "(no answer)";
              const correctAnswer = isShortAnswer
                ? a.correct_answer_text
                : a.options.find((o) => o.id === a.correct_option_id)?.text;
              return (
                <div
                  key={a.question_id}
                  style={{
                    padding: "14px 16px",
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    borderRadius: 11,
                  }}
                >
                  <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
                    <DifficultyBadge difficulty={a.difficulty} />
                    <span className="dv-mono" style={{ fontSize: 10.5, color: "var(--fg-subtle)" }}>
                      {formatKind(a.kind).toUpperCase()}
                    </span>
                  </div>
                  <div style={{ fontSize: 14.5, fontWeight: 500, lineHeight: 1.5 }}>{md(a.prompt)}</div>
                  <div style={{ marginTop: 8, fontSize: 13, color: "var(--red)" }}>
                    Your answer: <strong>{isShortAnswer ? yourAnswer : md(yourAnswer)}</strong>
                  </div>
                  {correctAnswer && (
                    <div style={{ marginTop: 3, fontSize: 13, color: "var(--green)" }}>
                      Correct answer: <strong>{md(correctAnswer)}</strong>
                    </div>
                  )}
                  {a.explanation && (
                    <div style={{ marginTop: 8, fontSize: 13, lineHeight: 1.6, color: "var(--fg-muted)" }}>
                      {md(a.explanation)}
                    </div>
                  )}
                  {a.citation && <CitationChip citation={a.citation} />}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div style={{ marginTop: 22 }}>
        <div style={{ fontSize: 13.5, fontWeight: 700, marginBottom: 10 }}>Attempt history</div>
        {historyError ? (
          <div style={{ fontSize: 12.5, color: "var(--red)" }}>{historyError}</div>
        ) : history === null ? (
          <div style={{ fontSize: 12.5, color: "var(--fg-subtle)", display: "flex", alignItems: "center", gap: 8 }}>
            <Spinner /> Loading history…
          </div>
        ) : history.length === 0 ? (
          <div style={{ fontSize: 12.5, color: "var(--fg-subtle)" }}>No previous attempts.</div>
        ) : (
          <div
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 11,
              overflow: "hidden",
            }}
          >
            {history.map((h, i) => (
              <div
                key={h.id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: "10px 14px",
                  borderTop: i === 0 ? "none" : "1px solid var(--border)",
                  background: h.id === attempt.id ? "var(--accent-soft)" : "transparent",
                }}
              >
                <span className="dv-mono" style={{ fontSize: 11.5, color: "var(--fg-subtle)", width: 128 }}>
                  {formatDate(h.finished_at)}
                </span>
                <span
                  className="dv-mono"
                  style={{
                    fontSize: 10.5,
                    fontWeight: 600,
                    color: h.mode === "retry_missed" ? "var(--amber)" : "var(--fg-muted)",
                  }}
                >
                  {formatMode(h.mode).toUpperCase()}
                </span>
                <span style={{ marginLeft: "auto", fontSize: 13, fontWeight: 700 }}>
                  {h.correct}/{h.total} · {pct(h.score)}
                </span>
                {h.missed_count > 0 && (
                  <span style={{ fontSize: 11.5, color: "var(--fg-subtle)" }}>{h.missed_count} missed</span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {onExit && (
        <div style={{ marginTop: 22 }}>
          <button
            onClick={onExit}
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: "var(--fg)",
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              borderRadius: 9,
              padding: "9px 16px",
            }}
          >
            Back to Study
          </button>
        </div>
      )}
    </div>
  );
}
