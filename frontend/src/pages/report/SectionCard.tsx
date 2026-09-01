import React from "react";
import type { Citation, MediaRef, Section } from "../../api/types";
import { renderParagraphs } from "./inlineMarkdown";
import { MediaCard } from "./MediaCard";

const BODY_STYLE: React.CSSProperties = {
  fontSize: 16,
  lineHeight: 1.72,
  color: "var(--fg)",
};

const DEEP_STYLE: React.CSSProperties = {
  fontSize: 15,
  lineHeight: 1.7,
  color: "var(--fg-muted)",
  paddingTop: 14,
  borderTop: "1px dashed var(--border-strong)",
};

function ProvenanceBadge({ kind }: { kind: "vision" | "ocr" }) {
  const isOcr = kind === "ocr";
  return (
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
  );
}

/** Header pill shown when `section.degraded` — the body is quoted straight
 *  from the PDF because the LLM call for this section fell back to the
 *  deterministic extractive draft. Reuses the amber OCR treatment (same
 *  "this wasn't fully model-produced" register) but is a distinct label so
 *  it never reads as a provenance badge. Sits next to `section.badge`,
 *  never replacing it. */
function DegradedPill() {
  return (
    <span
      className="dv-mono"
      title="Quoted from the PDF — the language model did not write this section."
      style={{
        display: "inline-flex",
        alignItems: "center",
        fontSize: 10.5,
        fontWeight: 600,
        color: "var(--amber)",
        background: "var(--amber-soft)",
        padding: "2px 7px",
        borderRadius: 5,
        lineHeight: 1,
        cursor: "help",
      }}
    >
      EXTRACTED
    </span>
  );
}

export function SectionCard({
  section,
  isOpen,
  onToggleOpen,
  isDeepOpen,
  onToggleDeep,
  figuresTotal,
  onCite,
  onOpenMedia,
  isFirst,
}: {
  section: Section;
  isOpen: boolean;
  onToggleOpen: () => void;
  isDeepOpen: boolean;
  onToggleDeep: () => void;
  /** Only meaningful for the Figures section — total figure count from report.stats. */
  figuresTotal: number;
  onCite: (c: Citation) => void;
  onOpenMedia: (m: MediaRef) => void;
  /** First card in the list gets the larger top gap below the stats bar (22px vs 12px), matching the prototype. */
  isFirst?: boolean;
}) {
  const isFigures = section.name === "Figures";
  const hasDeepDive = !!section.deep_dive_markdown;
  const nonTextProvenance = section.provenance.filter(
    (p): p is "vision" | "ocr" => p === "vision" || p === "ocr"
  );
  const isEmpty = !section.body_markdown && section.media.length === 0;

  return (
    <div
      style={{
        marginTop: isFirst ? 22 : 12,
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 13,
        overflow: "hidden",
      }}
    >
      <button
        onClick={onToggleOpen}
        style={{
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "stretch",
          gap: 4,
          padding: "16px 18px",
          background: "transparent",
          border: "none",
          textAlign: "left",
        }}
      >
        <span style={{ display: "flex", alignItems: "center", gap: 11 }}>
          <span
            style={{
              color: "var(--fg-subtle)",
              fontSize: 12,
              transition: "transform .15s",
            }}
          >
            {isOpen ? "▾" : "▸"}
          </span>
          <span style={{ fontSize: 16, fontWeight: 600, letterSpacing: "-.01em", color: "var(--fg)" }}>
            {section.name}
          </span>
          <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
            {isFigures && (
              <span className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)" }}>
                {section.media.length} of {figuresTotal} shown
              </span>
            )}
            {!isFigures && section.badge && (
              <span
                className="dv-mono"
                style={{
                  fontSize: 10.5,
                  fontWeight: 600,
                  color: section.badge.toUpperCase().includes("OCR") ? "var(--amber)" : "var(--accent)",
                  background: section.badge.toUpperCase().includes("OCR")
                    ? "var(--amber-soft)"
                    : "var(--accent-soft)",
                  padding: "2px 7px",
                  borderRadius: 5,
                }}
              >
                {section.badge}
              </span>
            )}
            {section.degraded && <DegradedPill />}
          </span>
        </span>
        {section.question && (
          <span
            style={{
              marginLeft: 23,
              fontSize: 12.5,
              fontWeight: 400,
              color: "var(--fg-subtle)",
              fontStyle: "italic",
            }}
          >
            {section.question}
          </span>
        )}
      </button>

      {isOpen && (
        <div style={{ padding: "0 20px 18px 47px" }}>
          {isEmpty && (
            <div style={{ fontSize: 13.5, color: "var(--fg-subtle)" }}>
              No content was extracted for this section.
            </div>
          )}

          {nonTextProvenance.length > 0 && section.body_markdown && (
            <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
              {nonTextProvenance.map((p) => (
                <ProvenanceBadge key={p} kind={p} />
              ))}
            </div>
          )}

          {section.body_markdown && (
            <div className="dv-serif">
              {renderParagraphs(section.body_markdown, section.citations, onCite, BODY_STYLE)}
            </div>
          )}

          {section.media.length > 0 && (
            <div style={{ marginTop: section.body_markdown ? 14 : 0 }}>
              {!isFigures && (
                <div
                  className="dv-mono"
                  style={{ fontSize: 11, color: "var(--fg-subtle)", marginBottom: 8 }}
                >
                  Figures referenced in this section
                </div>
              )}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 12,
                }}
              >
                {section.media.map((m, i) => (
                  <MediaCard
                    key={m.id}
                    media={m}
                    span={i === section.media.length - 1 && section.media.length % 2 === 1 && section.media.length > 1}
                    onOpen={onOpenMedia}
                  />
                ))}
              </div>
            </div>
          )}

          {isDeepOpen && section.deep_dive_markdown && (
            <div className="dv-serif" style={{ marginTop: 14 }}>
              {renderParagraphs(section.deep_dive_markdown, section.citations, onCite, DEEP_STYLE)}
            </div>
          )}

          {hasDeepDive && (
            <button
              onClick={onToggleDeep}
              style={{
                marginTop: 14,
                fontSize: 12.5,
                fontWeight: 600,
                color: "var(--accent)",
                background: "transparent",
                border: "none",
                padding: 0,
                fontFamily: "'IBM Plex Sans'",
              }}
            >
              {isDeepOpen ? "− Hide detail" : "+ Deep dive"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
