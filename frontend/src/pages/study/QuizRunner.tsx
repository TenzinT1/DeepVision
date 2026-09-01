import React, { useRef, useState } from "react";
import { ApiError, checkQuizAnswer, submitQuizAttempt } from "../../api/client";
import { renderInline } from "../report/inlineMarkdown";
import type {
  Difficulty,
  GradedAnswer,
  QuestionKind,
  QuizAttempt,
  QuizCitation,
  QuizPublic,
  QuizQuestionPublic,
  SubmittedAnswer,
} from "../../api/types";

/**
 * One question at a time, server-graded, WITH immediate per-question
 * feedback. `QuizQuestionPublic` never carries the answer key (see
 * types.ts), so "immediate feedback" can only come from a server round
 * trip — there is no client-side way to know if an answer is right.
 *
 * TWO ENDPOINTS, ON PURPOSE:
 *   - `checkQuizAnswer` (POST /quiz/{id}/check) grades the question the
 *     user just answered and records NOTHING. One call per question.
 *   - `submitQuizAttempt` (POST /quiz/{id}/attempt) grades the whole run
 *     and writes ONE row of attempt history. One call per finished quiz.
 *
 * Driving the live feedback off /attempt instead would persist an attempt
 * per question — a ten-question run would land in history as ten attempts,
 * nine of them junk partial scores. Both endpoints share the server's one
 * grader, so the per-question verdict always matches the final attempt.
 */

/** Quiz prompts/options/explanations carry `**bold**` markers but have no
 *  `Citation[]` of their own — reuse the report's inline renderer with an
 *  empty citation list so any `[n]`-shaped text falls back to literal text
 *  instead of a dead clickable marker (see inlineMarkdown.tsx). */
function md(text: string): React.ReactNode {
  return renderInline(text, [], () => {});
}

const DIFFICULTY_STYLE: Record<Difficulty, { label: string; color: string; bg: string }> = {
  recall: { label: "Recall", color: "var(--teal)", bg: "var(--teal-soft)" },
  understand: { label: "Understand", color: "var(--accent)", bg: "var(--accent-soft)" },
  apply: { label: "Apply", color: "var(--amber)", bg: "var(--amber-soft)" },
};

export function DifficultyBadge({ difficulty }: { difficulty: Difficulty }) {
  const s = DIFFICULTY_STYLE[difficulty];
  return (
    <span
      className="dv-mono"
      style={{
        fontSize: 10.5,
        fontWeight: 600,
        letterSpacing: ".02em",
        color: s.color,
        background: s.bg,
        padding: "2px 7px",
        borderRadius: 5,
        lineHeight: 1.6,
      }}
    >
      {s.label.toUpperCase()}
    </span>
  );
}

export function formatKind(kind: QuestionKind): string {
  switch (kind) {
    case "multiple_choice":
      return "Multiple choice";
    case "true_false":
      return "True / False";
    case "short_answer":
      return "Short answer";
    default:
      return kind;
  }
}

export function CitationChip({ citation }: { citation: QuizCitation }) {
  return (
    <div
      style={{
        marginTop: 12,
        padding: "10px 12px",
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        borderRadius: 9,
      }}
    >
      <div className="dv-mono" style={{ fontSize: 10.5, color: "var(--fg-subtle)", marginBottom: 4 }}>
        SOURCE · {citation.section}
        {citation.page != null ? ` · p.${citation.page}` : ""}
      </div>
      {citation.snippet && (
        <div style={{ fontSize: 12.5, color: "var(--fg-muted)", fontStyle: "italic", lineHeight: 1.5 }}>
          &ldquo;{citation.snippet}&rdquo;
        </div>
      )}
    </div>
  );
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

function buildAnswer(
  q: QuizQuestionPublic,
  selectedOptionId: string | null,
  answerText: string,
  elapsedMs: number
): SubmittedAnswer {
  const isShortAnswer = q.kind === "short_answer";
  return {
    question_id: q.id,
    selected_option_id: isShortAnswer ? null : selectedOptionId,
    answer_text: isShortAnswer ? answerText.trim() : null,
    elapsed_ms: elapsedMs,
  };
}

export interface QuizRunnerProps {
  /** The quiz to run. Answers are always hidden here by design (QuizPublic). */
  quiz: QuizPublic;
  /** Called once with the finished, fully-graded attempt (all questions answered, none omitted). */
  onComplete: (attempt: QuizAttempt) => void;
  /** "Exit without finishing" — optional, only shown when provided. */
  onExit?: () => void;
}

export default function QuizRunner({ quiz, onComplete, onExit }: QuizRunnerProps) {
  const total = quiz.questions.length;
  const [index, setIndex] = useState(0);
  const [answeredList, setAnsweredList] = useState<SubmittedAnswer[]>([]);
  const [selectedOptionId, setSelectedOptionId] = useState<string | null>(null);
  const [answerText, setAnswerText] = useState("");
  const [feedback, setFeedback] = useState<GradedAnswer | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [finishing, setFinishing] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const attemptStartedAtRef = useRef<string>(new Date().toISOString());
  const questionStartRef = useRef<number>(Date.now());

  if (total === 0) {
    return (
      <div
        style={{
          padding: "28px 24px",
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 13,
          textAlign: "center",
        }}
      >
        <div style={{ fontSize: 14, fontWeight: 600 }}>This quiz has no questions.</div>
        {onExit && (
          <button
            onClick={onExit}
            style={{
              marginTop: 14,
              fontSize: 13,
              fontWeight: 600,
              color: "var(--fg)",
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              borderRadius: 9,
              padding: "8px 16px",
            }}
          >
            Back
          </button>
        )}
      </div>
    );
  }

  const q = quiz.questions[index];
  const isLast = index === total - 1;
  const hasAnswer = q.kind === "short_answer" ? answerText.trim().length > 0 : !!selectedOptionId;

  /** Grade the current question only — no attempt is recorded by this call. */
  async function handleCheck() {
    if (submitting || feedback) return;
    const elapsed = Date.now() - questionStartRef.current;
    const answer = buildAnswer(q, selectedOptionId, answerText, elapsed);
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await checkQuizAnswer(quiz.id, answer);
      setAnsweredList((prev) => [...prev, answer]);
      setFeedback(res.graded);
    } catch (err) {
      setSubmitError(err instanceof ApiError ? err.message : "Couldn't check your answer. Try again.");
    } finally {
      setSubmitting(false);
    }
  }

  /** On the last question this is the ONLY call that records an attempt. */
  async function handleAdvance() {
    if (!isLast) {
      setIndex((i) => i + 1);
      setSelectedOptionId(null);
      setAnswerText("");
      setFeedback(null);
      setSubmitError(null);
      questionStartRef.current = Date.now();
      return;
    }
    if (finishing) return;
    setFinishing(true);
    setSubmitError(null);
    try {
      const attempt: QuizAttempt = await submitQuizAttempt(quiz.id, answeredList, {
        mode: quiz.mode,
        retryOfAttemptId: quiz.retry_of_attempt_id ?? null,
        startedAt: attemptStartedAtRef.current,
      });
      onComplete(attempt);
    } catch (err) {
      setSubmitError(
        err instanceof ApiError ? err.message : "Couldn't save your attempt. Try again."
      );
      setFinishing(false);
    }
  }

  const showOptions = q.kind === "multiple_choice" || q.kind === "true_false";

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <div style={{ flex: 1 }}>
          <div className="dv-mono" style={{ fontSize: 11.5, color: "var(--fg-subtle)" }}>
            Question {index + 1} of {total}
          </div>
          <div
            style={{
              marginTop: 6,
              height: 5,
              borderRadius: 20,
              background: "var(--surface-3)",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                borderRadius: 20,
                background: "var(--accent)",
                width: `${((index + (feedback ? 1 : 0)) / total) * 100}%`,
                transition: "width .25s ease",
              }}
            />
          </div>
        </div>
        {quiz.mode === "retry_missed" && (
          <span
            className="dv-mono"
            style={{
              fontSize: 10.5,
              fontWeight: 600,
              color: "var(--amber)",
              background: "var(--amber-soft)",
              padding: "3px 8px",
              borderRadius: 6,
            }}
          >
            RETRY · MISSED ONLY
          </span>
        )}
        {onExit && (
          <button
            onClick={onExit}
            aria-label="Exit quiz"
            title="Exit quiz"
            style={{
              fontSize: 12,
              color: "var(--fg-subtle)",
              background: "transparent",
              border: "none",
              padding: "4px 6px",
            }}
          >
            ✕
          </button>
        )}
      </div>

      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 13,
          padding: "20px 22px",
        }}
      >
        <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
          <DifficultyBadge difficulty={q.difficulty} />
          <span className="dv-mono" style={{ fontSize: 10.5, color: "var(--fg-subtle)" }}>
            {formatKind(q.kind).toUpperCase()}
          </span>
        </div>

        <div className="dv-serif" style={{ fontSize: 17, lineHeight: 1.55, fontWeight: 500 }}>
          {md(q.prompt)}
        </div>

        {showOptions && (
          <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 9 }}>
            {q.options.map((opt) => {
              const isSelected = selectedOptionId === opt.id;
              const isCorrectOpt = feedback?.correct_option_id === opt.id;
              const isWrongSelected = !!feedback && isSelected && !feedback.is_correct;

              let bg = "var(--surface)";
              let border = "var(--border)";
              let color = "var(--fg)";
              if (feedback) {
                if (isCorrectOpt) {
                  bg = "var(--green-soft)";
                  border = "var(--green)";
                  color = "var(--green)";
                } else if (isWrongSelected) {
                  bg = "var(--red-soft)";
                  border = "var(--red)";
                  color = "var(--red)";
                }
              } else if (isSelected) {
                bg = "var(--accent-soft)";
                border = "var(--accent)";
                color = "var(--accent)";
              }

              return (
                <button
                  key={opt.id}
                  disabled={!!feedback || submitting}
                  onClick={() => setSelectedOptionId(opt.id)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    width: "100%",
                    textAlign: "left",
                    fontSize: 14,
                    fontWeight: isSelected || isCorrectOpt ? 600 : 500,
                    color,
                    background: bg,
                    border: `1.5px solid ${border}`,
                    borderRadius: 10,
                    padding: "11px 14px",
                    cursor: feedback ? "default" : "pointer",
                  }}
                >
                  <span>{md(opt.text)}</span>
                  {feedback && isCorrectOpt && <span style={{ marginLeft: "auto" }}>✓</span>}
                  {feedback && isWrongSelected && <span style={{ marginLeft: "auto" }}>✕</span>}
                </button>
              );
            })}
          </div>
        )}

        {q.kind === "short_answer" && (
          <div style={{ marginTop: 16 }}>
            <input
              type="text"
              value={answerText}
              disabled={!!feedback || submitting}
              onChange={(e) => setAnswerText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && hasAnswer && !feedback) handleCheck();
              }}
              placeholder="Type your answer…"
              style={{
                width: "100%",
                fontSize: 14,
                color: "var(--fg)",
                background: feedback ? "var(--surface-2)" : "var(--surface)",
                border: "1.5px solid var(--border)",
                borderRadius: 10,
                padding: "11px 14px",
              }}
            />
            {feedback && !feedback.is_correct && feedback.correct_answer_text && (
              <div style={{ marginTop: 8, fontSize: 12.5, color: "var(--green)" }}>
                Accepted answer: <strong>{md(feedback.correct_answer_text)}</strong>
              </div>
            )}
          </div>
        )}

        {submitError && (
          <div style={{ marginTop: 12, fontSize: 12.5, color: "var(--red)" }}>{submitError}</div>
        )}

        {feedback && (
          <div
            style={{
              marginTop: 16,
              padding: "14px 16px",
              borderRadius: 11,
              background: feedback.is_correct ? "var(--green-soft)" : "var(--red-soft)",
              border: `1px solid ${feedback.is_correct ? "var(--green)" : "var(--red)"}`,
            }}
          >
            <div
              style={{
                fontSize: 13.5,
                fontWeight: 700,
                color: feedback.is_correct ? "var(--green)" : "var(--red)",
              }}
            >
              {feedback.is_correct ? "Correct" : "Not quite"}
            </div>
            {feedback.explanation && (
              <div style={{ marginTop: 6, fontSize: 13.5, lineHeight: 1.6, color: "var(--fg)" }}>
                {md(feedback.explanation)}
              </div>
            )}
            {feedback.citation && <CitationChip citation={feedback.citation} />}
          </div>
        )}

        <div style={{ marginTop: 18, display: "flex", justifyContent: "flex-end" }}>
          {!feedback ? (
            <button
              onClick={handleCheck}
              disabled={!hasAnswer || submitting}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontSize: 13.5,
                fontWeight: 600,
                color: "var(--accent-fg)",
                background: "var(--accent)",
                border: "none",
                borderRadius: 9,
                padding: "9px 18px",
                opacity: !hasAnswer || submitting ? 0.55 : 1,
                cursor: !hasAnswer || submitting ? "not-allowed" : "pointer",
              }}
            >
              {submitting && <Spinner />}
              Check answer
            </button>
          ) : (
            <button
              onClick={handleAdvance}
              disabled={finishing}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontSize: 13.5,
                fontWeight: 600,
                color: "var(--accent-fg)",
                background: "var(--accent)",
                border: "none",
                borderRadius: 9,
                padding: "9px 18px",
                opacity: finishing ? 0.55 : 1,
                cursor: finishing ? "not-allowed" : "pointer",
              }}
            >
              {finishing && <Spinner />}
              {isLast ? "See results →" : "Next question →"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
