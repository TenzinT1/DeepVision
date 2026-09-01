import React from "react";
import type { Citation } from "../../api/types";

/**
 * Tiny inline-markdown renderer for Section.body_markdown / deep_dive_markdown.
 * Supports **bold**, *italic* and inline [n] citation markers (resolved
 * against the section's own `citations` list by `marker` and wired to
 * useOverlays().openCitation). Anything else is left as literal text —
 * the backend only ever emits this small subset.
 */

type Segment =
  | { type: "text"; value: string }
  | { type: "bold"; value: string }
  | { type: "italic"; value: string }
  | { type: "citation"; marker: number; raw: string };

const TOKEN_RE = /(\*\*[^*]+\*\*|\*[^*]+\*|\[\d+\])/g;

function tokenize(text: string): Segment[] {
  const segments: Segment[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  TOKEN_RE.lastIndex = 0;
  while ((match = TOKEN_RE.exec(text))) {
    if (match.index > lastIndex) {
      segments.push({ type: "text", value: text.slice(lastIndex, match.index) });
    }
    const token = match[0];
    if (token.startsWith("**")) {
      segments.push({ type: "bold", value: token.slice(2, -2) });
    } else if (token.startsWith("[")) {
      segments.push({ type: "citation", marker: Number(token.slice(1, -1)), raw: token });
    } else {
      segments.push({ type: "italic", value: token.slice(1, -1) });
    }
    lastIndex = TOKEN_RE.lastIndex;
  }
  if (lastIndex < text.length) segments.push({ type: "text", value: text.slice(lastIndex) });
  return segments;
}

/**
 * Exported separately from `renderParagraphs` for callers that need inline
 * formatting (`**bold**` / `*italic*` / `[n]`) without paragraph/list-block
 * splitting — e.g. quiz prompts, options and explanations (`pages/study/`),
 * which are single strings, not multi-paragraph section bodies. Pass `[]`
 * for `citations` when the caller has no `Citation[]` of its own (quiz text
 * uses a different, unrelated `QuizCitation` shape): a `[n]` marker then
 * finds no match and falls back to literal text (see the `citation` case
 * below), never a dead/broken clickable link.
 */
export function renderInline(
  text: string,
  citations: Citation[],
  onCite: (c: Citation) => void
): React.ReactNode[] {
  return tokenize(text).map((seg, i) => {
    switch (seg.type) {
      case "bold":
        return <strong key={i}>{seg.value}</strong>;
      case "italic":
        return <em key={i}>{seg.value}</em>;
      case "citation": {
        const citation = citations.find((c) => c.marker === seg.marker);
        if (!citation) return <React.Fragment key={i}>{seg.raw}</React.Fragment>;
        return (
          <sup
            key={i}
            onClick={() => onCite(citation)}
            style={{
              cursor: "pointer",
              color: "var(--accent)",
              fontWeight: 600,
              fontFamily: "'IBM Plex Sans'",
              fontSize: 11,
              padding: "0 1px",
            }}
          >
            [{seg.marker}]
          </sup>
        );
      }
      default:
        return <React.Fragment key={i}>{seg.value}</React.Fragment>;
    }
  });
}

/**
 * List-line detection for the eleven-section report's four fixed-shape
 * sections (see `Section`'s docstring in `deepvision/models/report.py` for
 * the exact per-section conventions):
 *   - At a Glance    — five `**Label** — value` lines (AT_A_GLANCE_FIELDS).
 *   - Key Concepts   — `**Term** — explanation from zero.` per item
 *     (absorbed the former Glossary, which used the identical shape).
 *   - Key Takeaways  — `**Lead-in** — claim [n].` per item.
 *   - Study Questions — `**N. Question text?**` per item, numbered.
 * All four are "one item per line" bodies rather than "one item per
 * blank-line-separated paragraph". Without this detection, a block's `\n`s
 * collapse under normal CSS white-space handling inside a single <p>, so the
 * five At a Glance fields (or any of the other three) would visually run
 * together into one paragraph instead of reading as distinct lines. A block
 * whose lines all look like list items — bold-led (`**...**`), or a
 * "- "/"1. " marker — is rendered as stacked rows (a lightweight definition
 * list) instead. This is a single generic check, not a per-section switch:
 * BOLD_LEAD_RE alone covers all four shapes above, since every one of them
 * puts a `**...**` span first on the line (Study Questions bold-wraps the
 * ordinal itself, e.g. `**1. …**`, rather than leading with a bare `1. `).
 */
const UNORDERED_RE = /^[-*]\s+/;
const ORDERED_RE = /^(\d+)[.)]\s+/;
const BOLD_LEAD_RE = /^\*\*[^*]+\*\*/;

function isListLine(line: string): boolean {
  return UNORDERED_RE.test(line) || ORDERED_RE.test(line) || BOLD_LEAD_RE.test(line);
}

/** Strips a leading "- "/"* "/"1. " marker, returning a bullet label (or null
 *  for a bare bold-led "**Term** — definition." line, which needs none). */
function splitListMarker(line: string): { bullet: string | null; text: string } {
  const ordered = line.match(ORDERED_RE);
  if (ordered) return { bullet: `${ordered[1]}.`, text: line.slice(ordered[0].length) };
  const unordered = line.match(UNORDERED_RE);
  if (unordered) return { bullet: "•", text: line.slice(unordered[0].length) };
  return { bullet: null, text: line };
}

function renderListBlock(
  key: number,
  lines: string[],
  citations: Citation[],
  onCite: (c: Citation) => void,
  paragraphStyle: React.CSSProperties,
  marginTop: number
): React.ReactNode {
  return (
    <div key={key} style={{ marginTop }}>
      {lines.map((line, i) => {
        const { bullet, text } = splitListMarker(line);
        return (
          <div
            key={i}
            style={{
              ...paragraphStyle,
              margin: i === 0 ? 0 : "8px 0 0",
              display: "flex",
              gap: 8,
              alignItems: "baseline",
            }}
          >
            {bullet && (
              <span
                className="dv-mono"
                style={{ flex: "none", fontSize: "0.8em", color: "var(--fg-subtle)" }}
              >
                {bullet}
              </span>
            )}
            <span style={{ flex: 1 }}>{renderInline(text, citations, onCite)}</span>
          </div>
        );
      })}
    </div>
  );
}

/** Splits on blank lines and renders each paragraph with inline formatting.
 *  A paragraph whose lines are ALL list-shaped (markdown "- "/"1. " markers,
 *  or "**Term** — …" definition lines) renders as stacked rows instead of a
 *  single flattened paragraph — see renderListBlock above. */
export function renderParagraphs(
  markdown: string | null | undefined,
  citations: Citation[],
  onCite: (c: Citation) => void,
  paragraphStyle: React.CSSProperties
): React.ReactNode {
  if (!markdown) return null;
  const blocks = markdown
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter(Boolean);
  if (blocks.length === 0) return null;
  return blocks.map((block, i) => {
    const marginTop = i === 0 ? 0 : 14;
    const lines = block
      .split(/\n/)
      .map((l) => l.trim())
      .filter(Boolean);
    if (lines.length > 0 && lines.every(isListLine)) {
      return renderListBlock(i, lines, citations, onCite, paragraphStyle, marginTop);
    }
    return (
      <p key={i} style={{ ...paragraphStyle, margin: marginTop === 0 ? 0 : `${marginTop}px 0 0` }}>
        {renderInline(block, citations, onCite)}
      </p>
    );
  });
}
