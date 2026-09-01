import React, { useEffect, useState } from "react";
import { ApiError, getDueQueue } from "../../api/client";
import type { DueCard } from "../../api/types";

/* --------------------------------------------------------------------------
   DueQueue — the "Due Today" tab body. This is the cross-paper review
   queue front and centre: a big due count, an optional per-paper filter
   chip, a couple of session toggles, a per-paper breakdown, and the big
   "Start review" button that hands off to FlashcardDrill.

   This component only PREVIEWS the queue (count + breakdown) — it does not
   drill. FlashcardDrill does its own fetch with the same params when the
   session actually starts, so the numbers shown here match what the drill
   will pull.
   -------------------------------------------------------------------------- */

export interface DueQueueProps {
  /** Scope to one paper (from ?paper=<id>), or null for the full library. */
  paperFilter: string | null;
  paperFilterLabel?: string | null;
  onClearPaperFilter: () => void;
  shuffle: boolean;
  includeNew: boolean;
  onShuffleChange: (v: boolean) => void;
  onIncludeNewChange: (v: boolean) => void;
  onStartReview: () => void;
  /** Bump to force a refetch (e.g. after a drill session ends). */
  reloadKey: number;
}

function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Something went wrong.";
}

interface PaperBucket {
  paperId: string;
  label: string;
  title: string;
  count: number;
}

function bucketByPaper(cards: DueCard[]): PaperBucket[] {
  const map = new Map<string, PaperBucket>();
  for (const dc of cards) {
    const key = dc.card.paper_id;
    const existing = map.get(key);
    if (existing) {
      existing.count += 1;
    } else {
      map.set(key, {
        paperId: key,
        label: dc.paper_label,
        title: dc.paper_title,
        count: 1,
      });
    }
  }
  return Array.from(map.values()).sort((a, b) => b.count - a.count);
}

const toggleStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 7,
  fontSize: 12.5,
  color: "var(--fg-muted)",
  fontWeight: 500,
  cursor: "pointer",
  userSelect: "none",
};

export function DueQueue({
  paperFilter,
  paperFilterLabel,
  onClearPaperFilter,
  shuffle,
  includeNew,
  onShuffleChange,
  onIncludeNewChange,
  onStartReview,
  reloadKey,
}: DueQueueProps) {
  const [cards, setCards] = useState<DueCard[] | null>(null);
  const [totalDue, setTotalDue] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getDueQueue({ paper_id: paperFilter ?? undefined, include_new: includeNew, shuffle, limit: 100 })
      .then((res) => {
        if (cancelled) return;
        setCards(res.cards);
        setTotalDue(res.total_due);
      })
      .catch((err) => {
        if (cancelled) return;
        setCards([]);
        setTotalDue(0);
        setError(errorMessage(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [paperFilter, includeNew, shuffle, reloadKey]);

  const buckets = cards ? bucketByPaper(cards) : [];
  const shownCount = cards?.length ?? 0;
  const hasMore = totalDue > shownCount;

  return (
    <div>
      {paperFilter && (
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            fontSize: 12.5,
            color: "var(--accent)",
            background: "var(--accent-soft)",
            border: "1px solid var(--accent-border)",
            borderRadius: 20,
            padding: "5px 6px 5px 12px",
            marginBottom: 18,
          }}
        >
          Filtered to <strong style={{ fontWeight: 600 }}>{paperFilterLabel ?? paperFilter}</strong>
          <button
            onClick={onClearPaperFilter}
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
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            ✕
          </button>
        </div>
      )}

      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 16,
          padding: "30px 28px",
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: 28,
        }}
      >
        <div style={{ flex: "none" }}>
          <div className="dv-mono" style={{ fontSize: 12, color: "var(--fg-subtle)", letterSpacing: ".03em" }}>
            DUE NOW
          </div>
          <div style={{ fontSize: 46, fontWeight: 700, lineHeight: 1.1, marginTop: 4 }}>
            {loading ? "…" : totalDue}
          </div>
          <div style={{ fontSize: 13, color: "var(--fg-muted)", marginTop: 4 }}>
            {loading
              ? "loading…"
              : totalDue === 0
                ? "nothing due — come back later"
                : `across ${buckets.length} paper${buckets.length === 1 ? "" : "s"}${hasMore ? " (showing top " + shownCount + ")" : ""}`}
          </div>
        </div>

        <div style={{ flex: "none", marginLeft: "auto", display: "flex", flexDirection: "column", gap: 10 }}>
          <label style={toggleStyle}>
            <input type="checkbox" checked={includeNew} onChange={(e) => onIncludeNewChange(e.target.checked)} />
            Include new cards
          </label>
          <label style={toggleStyle}>
            <input type="checkbox" checked={shuffle} onChange={(e) => onShuffleChange(e.target.checked)} />
            Shuffle order
          </label>
        </div>

        <button
          onClick={onStartReview}
          disabled={loading || totalDue === 0}
          style={{
            flex: "none",
            fontSize: 15,
            fontWeight: 600,
            color: "var(--accent-fg)",
            background: totalDue === 0 || loading ? "var(--fg-subtle)" : "var(--accent)",
            border: "none",
            borderRadius: 12,
            padding: "16px 30px",
            opacity: totalDue === 0 || loading ? 0.6 : 1,
            cursor: totalDue === 0 || loading ? "default" : "pointer",
          }}
        >
          Start review →
        </button>
      </div>

      {error && (
        <div
          style={{
            marginTop: 14,
            fontSize: 12.5,
            color: "var(--red)",
            background: "var(--red-soft)",
            border: "1px solid var(--red)",
            borderRadius: 10,
            padding: "10px 14px",
          }}
        >
          {error}
        </div>
      )}

      {!loading && buckets.length > 0 && (
        <div style={{ marginTop: 22 }}>
          <div className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)", marginBottom: 10, letterSpacing: ".03em" }}>
            BY PAPER
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
            {buckets.map((b) => (
              <div
                key={b.paperId}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: "12px 4px",
                  borderBottom: "1px solid var(--border)",
                }}
              >
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div
                    style={{
                      fontSize: 13.5,
                      fontWeight: 500,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {b.title}
                  </div>
                </div>
                <span
                  className="dv-mono"
                  style={{
                    flex: "none",
                    fontSize: 12,
                    fontWeight: 600,
                    color: "var(--accent)",
                    background: "var(--accent-soft)",
                    padding: "3px 10px",
                    borderRadius: 20,
                  }}
                >
                  {b.count} due
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
