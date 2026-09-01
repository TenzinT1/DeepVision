import React, { useEffect, useState } from "react";
import { mediaUrl } from "../../api/client";
import { useOverlays } from "./OverlaysContext";

export function CitationPopover() {
  const { citation, closeCitation } = useOverlays();
  const [imgFailed, setImgFailed] = useState(false);

  useEffect(() => {
    setImgFailed(false);
  }, [citation?.id]);

  useEffect(() => {
    if (!citation) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeCitation();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [citation, closeCitation]);

  if (!citation) return null;

  const src = citation.chunk_id ? mediaUrl(citation.chunk_id) : null;

  return (
    <div
      onClick={closeCitation}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 75,
        background: "rgba(15,18,28,.4)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        animation: "dv-fade .12s ease",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 520,
          maxWidth: "100%",
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 14,
          boxShadow: "var(--shadow-lg)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            padding: "15px 18px",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ fontSize: 13.5, fontWeight: 600 }}>Source · {citation.source}</div>
          <button
            onClick={closeCitation}
            style={{
              color: "var(--fg-subtle)",
              background: "var(--surface-2)",
              border: "none",
              borderRadius: 7,
              width: 28,
              height: 28,
            }}
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div style={{ display: "flex", gap: 14, padding: 18 }}>
          <div style={{ flex: "none", width: 110 }}>
            <div
              style={{
                height: 150,
                border: "1px solid var(--border)",
                borderRadius: 8,
                overflow: "hidden",
                background: src && !imgFailed ? "var(--surface)" : undefined,
                backgroundImage:
                  src && !imgFailed
                    ? undefined
                    : "repeating-linear-gradient(0deg,var(--surface-2) 0 7px,var(--surface-3) 7px 8px)",
                display: "flex",
                alignItems: "flex-end",
                justifyContent: "center",
                padding: src && !imgFailed ? 0 : "0 0 8px",
              }}
            >
              {src && !imgFailed ? (
                <img
                  src={src}
                  alt={citation.page_label}
                  onError={() => setImgFailed(true)}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                />
              ) : (
                <span
                  className="dv-mono"
                  style={{
                    fontSize: 9.5,
                    color: "var(--fg-subtle)",
                    background: "var(--surface)",
                    padding: "2px 5px",
                    borderRadius: 4,
                  }}
                >
                  {citation.page_label}
                </span>
              )}
            </div>
          </div>
          <div style={{ flex: 1 }}>
            <div className="dv-mono" style={{ fontSize: 10.5, color: "var(--fg-subtle)", letterSpacing: ".03em" }}>
              EXACT SNIPPET
            </div>
            <p
              className="dv-serif"
              style={{
                fontSize: 14,
                lineHeight: 1.65,
                color: "var(--fg)",
                margin: "8px 0 0",
                padding: "12px 14px",
                background: "var(--surface-2)",
                borderLeft: "3px solid var(--accent)",
                borderRadius: "0 8px 8px 0",
              }}
            >
              {citation.snippet}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
