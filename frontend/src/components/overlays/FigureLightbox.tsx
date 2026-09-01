import React, { useEffect, useState } from "react";
import { mediaUrl } from "../../api/client";
import { useOverlays } from "./OverlaysContext";

export function FigureLightbox() {
  const { lightbox, closeLightbox } = useOverlays();
  const [imgFailed, setImgFailed] = useState(false);

  useEffect(() => {
    setImgFailed(false);
  }, [lightbox?.id]);

  useEffect(() => {
    if (!lightbox) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeLightbox();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox, closeLightbox]);

  if (!lightbox) return null;

  const isOcr = lightbox.provenance === "ocr";
  const badgeText = lightbox.provenance.toUpperCase();
  const badgeStyle: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    fontFamily: "'IBM Plex Mono'",
    fontSize: 10,
    fontWeight: 600,
    color: isOcr ? "var(--amber)" : "var(--teal)",
    background: isOcr ? "var(--amber-soft)" : "var(--teal-soft)",
    padding: "2px 7px",
    borderRadius: 5,
  };

  const src = lightbox.chunk_id ? mediaUrl(lightbox.chunk_id) : null;

  return (
    <div
      onClick={closeLightbox}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 70,
        background: "rgba(10,12,20,.8)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 40,
        animation: "dv-fade .15s ease",
      }}
    >
      <div style={{ maxWidth: 820, width: "100%" }} onClick={(e) => e.stopPropagation()}>
        <div
          style={{
            background: "var(--surface)",
            borderRadius: 14,
            overflow: "hidden",
            boxShadow: "var(--shadow-lg)",
          }}
        >
          <div
            style={{
              height: 420,
              backgroundImage:
                "repeating-linear-gradient(135deg,var(--surface-2) 0 18px,var(--surface-3) 18px 36px)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {src && !imgFailed ? (
              <img
                src={src}
                alt={lightbox.label}
                onError={() => setImgFailed(true)}
                style={{ maxWidth: "100%", maxHeight: 420, objectFit: "contain" }}
              />
            ) : (
              <span
                className="dv-mono"
                style={{
                  fontSize: 13,
                  color: "var(--fg-subtle)",
                  background: "var(--surface)",
                  padding: "6px 12px",
                  borderRadius: 7,
                  border: "1px solid var(--border)",
                }}
              >
                {lightbox.filename || lightbox.label}
              </span>
            )}
          </div>
          <div style={{ padding: "16px 20px", display: "flex", alignItems: "center", gap: 12 }}>
            <span style={badgeStyle}>{badgeText}</span>
            <span style={{ fontSize: 13.5, color: "var(--fg)" }}>{lightbox.caption || lightbox.label}</span>
          </div>
        </div>
        <div style={{ textAlign: "center", marginTop: 14 }}>
          <button
            onClick={closeLightbox}
            style={{
              fontSize: 13,
              fontWeight: 500,
              color: "#fff",
              background: "rgba(255,255,255,.12)",
              border: "1px solid rgba(255,255,255,.25)",
              borderRadius: 9,
              padding: "8px 18px",
            }}
          >
            Close (Esc)
          </button>
        </div>
      </div>
    </div>
  );
}
