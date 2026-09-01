import React, { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, getDueQueue, reviewCard, starCard } from "../../api/client";
import { RATINGS } from "../../api/types";
import type { DueCard, Flashcard, Rating } from "../../api/types";

/* --------------------------------------------------------------------------
   FlashcardDrill — the actual drill. Fetches one batch of due cards up
   front (via the same GET /study/due the DueQueue preview used), then walks
   through them front -> flip -> rate, posting each rating synchronously and
   advancing. Ends with a session summary.

   Scheduling is session-based: there are no dates and no intervals. A rating
   decides only where the card goes in THIS session — the server returns
   `requeue_after` (how many other cards to put in front of it) or null to
   retire it for the sitting. Nothing is persisted but the review log.

   Keyboard: space flips the current card; 1-3 rate Again/Almost/Got it
   (only once flipped).
   -------------------------------------------------------------------------- */

export interface FlashcardDrillProps {
  /** Scope the session to one paper, or null for the cross-paper queue. */
  paperId?: string | null;
  shuffle?: boolean;
  includeNew?: boolean;
  onExit: () => void;
}

function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Something went wrong.";
}

const RATING_META: Record<Rating, { label: string; key: string; color: string; soft: string }> = {
  again: { label: "Again", key: "1", color: "var(--red)", soft: "var(--red-soft)" },
  almost: { label: "Almost", key: "2", color: "var(--amber)", soft: "var(--amber-soft)" },
  got_it: { label: "Got it", key: "3", color: "var(--green)", soft: "var(--green-soft)" },
};

function Spinner() {
  return (
    <span
      style={{
        display: "inline-block",
        width: 16,
        height: 16,
        borderRadius: "50%",
        border: "2px solid var(--border-strong)",
        borderTopColor: "var(--accent)",
        animation: "dv-spin .7s linear infinite",
      }}
    />
  );
}

const CENTER_WRAP: React.CSSProperties = {
  maxWidth: 720,
  margin: "0 auto",
  padding: "34px 24px 60px",
  display: "flex",
  flexDirection: "column",
  minHeight: "calc(100vh - 120px)",
};

export function FlashcardDrill({ paperId = null, shuffle = false, includeNew = true, onExit }: FlashcardDrillProps) {
  const [queue, setQueue] = useState<DueCard[] | null>(null);
  const [totalDue, setTotalDue] = useState(0);
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [lastResult, setLastResult] = useState<{ rating: Rating; text: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starBusy, setStarBusy] = useState(false);
  const [counts, setCounts] = useState<Record<Rating, number>>({ again: 0, almost: 0, got_it: 0 });
  const cardStartRef = useRef<number>(Date.now());
  /** Queue length when the session began — the denominator for "retired". */
  const startSizeRef = useRef<number>(0);
  const advanceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const load = useCallback(() => {
    setQueue(null);
    setIndex(0);
    setFlipped(false);
    setLastResult(null);
    setError(null);
    getDueQueue({ paper_id: paperId ?? undefined, include_new: includeNew, shuffle, limit: 50 })
      .then((res) => {
        setQueue(res.cards);
        startSizeRef.current = res.cards.length;
        setTotalDue(res.total_due);
      })
      .catch((err) => {
        setQueue([]);
        setTotalDue(0);
        setError(errorMessage(err));
      });
  }, [paperId, includeNew, shuffle]);

  useEffect(() => {
    load();
    return () => {
      if (advanceTimerRef.current) clearTimeout(advanceTimerRef.current);
    };
  }, [load]);

  useEffect(() => {
    cardStartRef.current = Date.now();
  }, [index]);

  const current: DueCard | null = queue && index < queue.length ? queue[index] : null;

  const rate = useCallback(
    (rating: Rating) => {
      if (!current || submitting) return;
      setSubmitting(true);
      const elapsed = Date.now() - cardStartRef.current;
      reviewCard(current.card.id, rating, elapsed)
        .then((res) => {
          setCounts((c) => ({ ...c, [rating]: c[rating] + 1 }));
          setLastResult({
            rating,
            text:
              res.requeue_after === null
                ? "done for now"
                : `back in ~${res.requeue_after}`,
          });
          setQueue((q) => {
            if (!q) return q;
            const next = q.slice();
            const [card] = next.splice(index, 1);
            if (card && res.requeue_after !== null) {
              // Reinsert `requeue_after` OTHER cards ahead. Splicing the card
              // out first means `index` already points at the next card, so
              // the gap is counted from there. Clamped to the end of the
              // queue: with fewer cards left than the gap the card still has
              // to come back — dropping one you just failed is backwards.
              const target = Math.min(index + res.requeue_after, next.length);
              next.splice(target, 0, card);
            }
            return next;
          });
          advanceTimerRef.current = setTimeout(() => {
            // The queue shrank or reordered under us; `index` now addresses
            // the next card without incrementing.
            setFlipped(false);
            setLastResult(null);
            setSubmitting(false);
          }, 650);
        })
        .catch((err) => {
          setSubmitting(false);
          setError(errorMessage(err));
        });
    },
    [current, submitting, index]
  );

  const toggleStar = useCallback(() => {
    if (!current || starBusy || !queue) return;
    const next = !current.card.starred;
    setStarBusy(true);
    starCard(current.card.id, next)
      .then(() => {
        setQueue((q) => {
          if (!q) return q;
          const copy = q.slice();
          copy[index] = { ...copy[index], card: { ...copy[index].card, starred: next } };
          return copy;
        });
      })
      .catch((err) => setError(errorMessage(err)))
      .finally(() => setStarBusy(false));
  }, [current, starBusy, queue, index]);

  // Keyboard: space to flip, 1-4 to rate (only once flipped, not mid-submit).
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (!current) return;
      if (e.code === "Space") {
        e.preventDefault();
        setFlipped((f) => !f);
        return;
      }
      if (!flipped || submitting) return;
      const rating = RATINGS.find((r) => RATING_META[r].key === e.key);
      if (rating) {
        e.preventDefault();
        rate(rating);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [current, flipped, submitting, rate]);

  const totalReviewed = counts.again + counts.almost + counts.got_it;
  // "N of M" is meaningless once cards come back: rating Again reinserts the
  // card, so the index does not advance and the counter would sit at 1/50
  // while you worked. What actually moves is how many cards are still in the
  // session, and how many you have retired out of the ones it started with.
  const cardsLeft = queue?.length ?? 0;
  const retired = Math.max(0, startSizeRef.current - cardsLeft);

  // --- loading ---
  if (queue === null) {
    return (
      <div style={CENTER_WRAP}>
        <div style={{ margin: "auto", display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
          <Spinner />
          <div style={{ fontSize: 13.5, color: "var(--fg-muted)" }}>Loading due cards…</div>
        </div>
      </div>
    );
  }

  // --- empty queue ---
  if (queue.length === 0) {
    return (
      <div style={CENTER_WRAP}>
        <div style={{ margin: "auto", textAlign: "center", maxWidth: 380 }}>
          <div style={{ fontSize: 40, marginBottom: 10 }}>✓</div>
          <div style={{ fontSize: 17, fontWeight: 600 }}>Nothing due — come back later</div>
          <div style={{ fontSize: 13.5, color: "var(--fg-muted)", marginTop: 8, lineHeight: 1.6 }}>
            {error ?? "All caught up. New reviews will appear here as cards come due."}
          </div>
          <button
            onClick={onExit}
            style={{
              marginTop: 22,
              fontSize: 13.5,
              fontWeight: 600,
              color: "var(--accent-fg)",
              background: "var(--accent)",
              border: "none",
              borderRadius: 10,
              padding: "10px 22px",
            }}
          >
            Back to Study
          </button>
        </div>
      </div>
    );
  }

  // --- session summary ---
  if (index >= queue.length) {
    const moreLeft = totalDue - queue.length;
    return (
      <div style={CENTER_WRAP}>
        <div style={{ margin: "auto", textAlign: "center", maxWidth: 460, width: "100%" }}>
          <div style={{ fontSize: 40, marginBottom: 6 }}>🎉</div>
          <div style={{ fontSize: 19, fontWeight: 700 }}>Session complete</div>
          <div style={{ fontSize: 13.5, color: "var(--fg-muted)", marginTop: 6 }}>
            {totalReviewed} card{totalReviewed === 1 ? "" : "s"} reviewed
          </div>
          <div style={{ display: "flex", gap: 10, marginTop: 24 }}>
            {RATINGS.map((r) => (
              <div
                key={r}
                style={{
                  flex: 1,
                  background: RATING_META[r].soft,
                  borderRadius: 12,
                  padding: "14px 8px",
                }}
              >
                <div style={{ fontSize: 22, fontWeight: 700, color: RATING_META[r].color }}>{counts[r]}</div>
                <div className="dv-mono" style={{ fontSize: 10.5, color: "var(--fg-muted)", marginTop: 3 }}>
                  {RATING_META[r].label.toUpperCase()}
                </div>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: 10, marginTop: 26, justifyContent: "center" }}>
            <button
              onClick={onExit}
              style={{
                fontSize: 13.5,
                fontWeight: 500,
                color: "var(--fg-muted)",
                background: "transparent",
                border: "1px solid var(--border)",
                borderRadius: 10,
                padding: "11px 20px",
              }}
            >
              Back to Study
            </button>
            {moreLeft > 0 && (
              <button
                onClick={load}
                style={{
                  fontSize: 13.5,
                  fontWeight: 600,
                  color: "var(--accent-fg)",
                  background: "var(--accent)",
                  border: "none",
                  borderRadius: 10,
                  padding: "11px 20px",
                }}
              >
                Continue — {moreLeft} more due
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  // --- active drill ---
  const card: Flashcard = current!.card;
  const provenance = [card.source_section, card.source_page ? `p. ${card.source_page}` : null]
    .filter(Boolean)
    .join(" · ");

  return (
    <div style={CENTER_WRAP}>
      {/* progress */}
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 22 }}>
        <button
          onClick={onExit}
          aria-label="Exit drill"
          style={{
            flex: "none",
            fontSize: 13,
            color: "var(--fg-muted)",
            background: "transparent",
            border: "none",
            padding: "4px 0",
          }}
        >
          ← Exit
        </button>
        <div style={{ flex: 1, height: 6, borderRadius: 20, background: "var(--surface-3)", overflow: "hidden" }}>
          <div
            style={{
              height: "100%",
              borderRadius: 20,
              background: "var(--accent)",
              width: `${startSizeRef.current ? (retired / startSizeRef.current) * 100 : 0}%`,
              transition: "width .25s ease",
            }}
          />
        </div>
        <span className="dv-mono" style={{ fontSize: 12, color: "var(--fg-subtle)", flex: "none" }}>
          {cardsLeft} left
        </span>
      </div>

      {error && (
        <div
          style={{
            fontSize: 12.5,
            color: "var(--red)",
            background: "var(--red-soft)",
            border: "1px solid var(--red)",
            borderRadius: 10,
            padding: "10px 14px",
            marginBottom: 16,
          }}
        >
          {error}
        </div>
      )}

      {/* card */}
      <div
        onClick={() => setFlipped((f) => !f)}
        style={{
          flex: 1,
          minHeight: 260,
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 18,
          boxShadow: "var(--shadow)",
          padding: "36px 34px",
          display: "flex",
          flexDirection: "column",
          cursor: "pointer",
          position: "relative",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span
            className="dv-mono"
            style={{
              fontSize: 10.5,
              fontWeight: 600,
              letterSpacing: ".04em",
              color: flipped ? "var(--teal)" : "var(--accent)",
              background: flipped ? "var(--teal-soft)" : "var(--accent-soft)",
              padding: "3px 9px",
              borderRadius: 6,
            }}
          >
            {flipped ? "ANSWER" : "QUESTION"}
          </span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              toggleStar();
            }}
            disabled={starBusy}
            aria-label={card.starred ? "Unstar card" : "Star card"}
            style={{
              background: "transparent",
              border: "none",
              fontSize: 20,
              lineHeight: 1,
              color: card.starred ? "var(--amber)" : "var(--fg-subtle)",
              padding: 4,
            }}
          >
            {card.starred ? "★" : "☆"}
          </button>
        </div>

        <div
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            textAlign: "center",
            padding: "20px 0",
          }}
        >
          <div style={{ fontSize: 21, fontWeight: 500, lineHeight: 1.55, color: "var(--fg)" }}>
            {flipped ? card.back : card.front}
          </div>
        </div>

        {!flipped && card.hint && (
          <div style={{ textAlign: "center", fontSize: 12.5, color: "var(--fg-subtle)", fontStyle: "italic" }}>
            hint: {card.hint}
          </div>
        )}

        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            fontSize: 11.5,
            color: "var(--fg-subtle)",
            marginTop: 10,
          }}
        >
          <span className="dv-mono">{provenance || "—"}</span>
          {!flipped && <span>space / click to flip</span>}
        </div>
      </div>

      {/* result toast */}
      {lastResult && (
        <div
          style={{
            textAlign: "center",
            marginTop: 12,
            fontSize: 12.5,
            fontWeight: 600,
            color: RATING_META[lastResult.rating].color,
          }}
        >
          {RATING_META[lastResult.rating].label} — next review in {lastResult.text}
        </div>
      )}

      {/* rating buttons */}
      <div style={{ display: "flex", gap: 10, marginTop: 18 }}>
        {RATINGS.map((r) => (
          <button
            key={r}
            onClick={() => rate(r)}
            disabled={!flipped || submitting}
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 4,
              padding: "13px 6px",
              borderRadius: 12,
              border: `1px solid ${flipped ? RATING_META[r].color : "var(--border)"}`,
              background: flipped ? RATING_META[r].soft : "var(--surface-2)",
              color: flipped ? RATING_META[r].color : "var(--fg-subtle)",
              opacity: !flipped || submitting ? 0.55 : 1,
              cursor: !flipped || submitting ? "default" : "pointer",
            }}
          >
            <span style={{ fontSize: 13.5, fontWeight: 700 }}>{RATING_META[r].label}</span>
            <span className="dv-mono" style={{ fontSize: 10.5 }}>
              {RATING_META[r].key}
            </span>
          </button>
        ))}
      </div>
      {!flipped && (
        <div style={{ textAlign: "center", fontSize: 11.5, color: "var(--fg-subtle)", marginTop: 8 }}>
          Flip the card to rate it.
        </div>
      )}
    </div>
  );
}
