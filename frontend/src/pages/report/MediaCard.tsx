import React, { useEffect, useState } from "react";
import { mediaUrl } from "../../api/client";
import type { MediaRef } from "../../api/types";

/** One figure/table thumbnail card in a section's media grid — matches the
 *  prototype's Figures grid tile exactly (placeholder pattern, VISION/OCR
 *  badge, caption). Opens the shared lightbox overlay on click. */
export function MediaCard({
  media,
  span,
  onOpen,
}: {
  media: MediaRef;
  span?: boolean;
  onOpen: (media: MediaRef) => void;
}) {
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
  }, [media.id]);

  const src = media.chunk_id ? mediaUrl(media.chunk_id) : null;
  const showImage = !!src && !failed;
  const isOcr = media.provenance === "ocr";
  const isVision = media.provenance === "vision";

  return (
    <div
      onClick={() => onOpen(media)}
      style={{
        cursor: "zoom-in",
        border: "1px solid var(--border)",
        borderRadius: 10,
        overflow: "hidden",
        gridColumn: span ? "span 2" : undefined,
        background: "var(--surface)",
      }}
    >
      <div
        style={{
          height: 130,
          backgroundImage: showImage
            ? undefined
            : "repeating-linear-gradient(135deg,var(--surface-2) 0 12px,var(--surface-3) 12px 24px)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          overflow: "hidden",
        }}
      >
        {showImage ? (
          <img
            src={src as string}
            alt={media.label}
            onError={() => setFailed(true)}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <span className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)" }}>
            {media.filename || media.label}
          </span>
        )}
      </div>
      <div style={{ padding: "9px 11px" }}>
        {(isOcr || isVision) && (
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 4,
              fontFamily: "'IBM Plex Mono'",
              fontSize: 10,
              fontWeight: 600,
              color: isOcr ? "var(--amber)" : "var(--teal)",
              background: isOcr ? "var(--amber-soft)" : "var(--teal-soft)",
              padding: "2px 6px",
              borderRadius: 5,
              lineHeight: 1,
            }}
          >
            {isOcr ? "OCR" : "VISION"}
          </span>
        )}
        <div style={{ fontSize: 12.5, color: "var(--fg-muted)", marginTop: 6, lineHeight: 1.5 }}>
          {media.label} · {media.caption}
        </div>
      </div>
    </div>
  );
}
