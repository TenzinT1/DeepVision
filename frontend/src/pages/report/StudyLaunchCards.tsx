import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, getPaperStudyStats } from "../../api/client";
import type { PaperStudyStats } from "../../api/types";

/**
 * Two compact launch cards for the Study layer, shown at the bottom of the
 * Report page. LAUNCH CARDS ONLY — they show live counts from
 * PaperStudyStats and deep-link into the top-level Study screen; they never
 * embed the flashcard drill or the quiz runner here (see types.ts's
 * PaperStudyStats doc comment for the exact template these follow).
 */

function pct(score: number): string {
  return `${Math.round(score * 100)}%`;
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

function LaunchCard({
  kicker,
  title,
  subtitle,
  detail,
  cta,
  onClick,
}: {
  kicker: string;
  title: string;
  subtitle: string;
  detail?: string;
  cta: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
        textAlign: "left",
        gap: 4,
        padding: "16px 18px",
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 13,
        cursor: "pointer",
      }}
    >
      <span className="dv-mono" style={{ fontSize: 10.5, fontWeight: 600, letterSpacing: ".04em", color: "var(--accent)" }}>
        {kicker}
      </span>
      <span style={{ fontSize: 15.5, fontWeight: 700, color: "var(--fg)" }}>{title}</span>
      <span style={{ fontSize: 12.5, color: "var(--fg-muted)" }}>{subtitle}</span>
      {detail && (
        <span style={{ fontSize: 11.5, color: "var(--fg-subtle)" }}>{detail}</span>
      )}
      <span style={{ marginTop: 8, fontSize: 12.5, fontWeight: 600, color: "var(--accent)" }}>{cta} →</span>
    </button>
  );
}

export function StudyLaunchCards({ paperId }: { paperId: string }) {
  const navigate = useNavigate();
  const [stats, setStats] = useState<PaperStudyStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStats(null);
    setError(null);
    getPaperStudyStats(paperId)
      .then((s) => {
        if (!cancelled) setStats(s);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Couldn't load study stats.");
      });
    return () => {
      cancelled = true;
    };
  }, [paperId]);

  if (error) {
    // Non-blocking — the report itself rendered fine, so this must not look
    // like a failed page. But hiding the cards entirely would also remove the
    // only route from Report into Study, so keep them: the deep links work
    // whether or not the stats call did. The counts are what's missing, and we
    // say so rather than showing a confident "0 cards".
    return (
      <div style={{ marginTop: 28 }}>
        <div style={{ display: "flex", gap: 12 }}>
          <LaunchCard
            kicker="STUDY"
            title="Flashcards"
            subtitle="Counts unavailable"
            cta="Open"
            onClick={() => navigate(`/study?paper=${encodeURIComponent(paperId)}&tab=flashcards`)}
          />
          <LaunchCard
            kicker="STUDY"
            title="Quiz"
            subtitle="Counts unavailable"
            cta="Open"
            onClick={() => navigate(`/study?paper=${encodeURIComponent(paperId)}&tab=quiz`)}
          />
        </div>
        <div style={{ marginTop: 8, fontSize: 11.5, color: "var(--fg-subtle)" }}>
          Couldn&rsquo;t load study stats — {error}
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div
        style={{
          marginTop: 28,
          display: "flex",
          alignItems: "center",
          gap: 8,
          fontSize: 12.5,
          color: "var(--fg-subtle)",
        }}
      >
        <Spinner /> Loading study tools…
      </div>
    );
  }

  const flashcardsSub = stats.has_deck
    ? `${stats.cards_total} card${stats.cards_total === 1 ? "" : "s"}`
    : "No flashcards yet";
  const flashcardsDetail = stats.has_deck
    ? `${stats.cards_starred} starred`
    : undefined;

  const quizSub =
    stats.quiz_count > 0
      ? `${stats.quiz_count} quiz${stats.quiz_count === 1 ? "" : "zes"}`
      : "No quiz yet";
  const quizDetail =
    stats.quiz_count > 0
      ? stats.quiz_best_score != null
        ? `best ${pct(stats.quiz_best_score)} · ${stats.quiz_attempt_count} attempt${stats.quiz_attempt_count === 1 ? "" : "s"}`
        : "not attempted yet"
      : undefined;

  return (
    <div style={{ marginTop: 28, display: "flex", gap: 12 }}>
      <LaunchCard
        kicker="STUDY"
        title="Flashcards"
        subtitle={flashcardsSub}
        detail={flashcardsDetail}
        cta={stats.has_deck ? "Review" : "Generate"}
        onClick={() => navigate(`/study?paper=${encodeURIComponent(paperId)}&tab=flashcards`)}
      />
      <LaunchCard
        kicker="STUDY"
        title="Quiz"
        subtitle={quizSub}
        detail={quizDetail}
        cta={stats.quiz_count > 0 ? "Practice" : "Generate"}
        onClick={() => navigate(`/study?paper=${encodeURIComponent(paperId)}&tab=quiz`)}
      />
    </div>
  );
}

export default StudyLaunchCards;
