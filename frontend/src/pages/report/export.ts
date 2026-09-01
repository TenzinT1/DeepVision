import type { Report } from "../../api/types";

/**
 * to_pdf) is an internal Domain-D interface, not one of the 14 routes wired
 * into src/api/client.ts — there is no `POST /export` endpoint to call. So
 * "Export as Markdown" / "Export as PDF" are produced entirely in the
 * browser from the already-fetched `Report`, matching the prototype's
 * Export ▾ dropdown without inventing a backend call.
 */

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function reportToMarkdown(report: Report): string {
  const paper = report.paper;
  const lines: string[] = [];

  if (paper) {
    lines.push(`# ${paper.title}`);
    lines.push("");
    const label = `${paper.arxiv_label}${paper.version ?? ""}`;
    const cats = paper.categories.join(" / ");
    lines.push([label, cats].filter(Boolean).join("  ·  "));
    lines.push("");
    if (paper.authors.length) lines.push(`**Authors:** ${paper.authors.join(", ")}`);
    if (paper.published) lines.push(`**Published:** ${paper.published}`);
    if (paper.abs_url) lines.push(`**arXiv:** ${paper.abs_url}`);
    lines.push("");
  } else {
    lines.push("# Report");
    lines.push("");
  }

  lines.push(
    `Pages: ${report.stats.pages} · Figures: ${report.stats.figures} · Citations extracted: ${report.stats.citations_extracted} · Reading time: ${report.stats.reading_time_min} min`
  );
  lines.push("");
  lines.push("---");

  for (const section of report.sections) {
    lines.push("");
    lines.push(`## ${section.name}${section.badge ? ` (${section.badge})` : ""}`);
    lines.push("");
    if (section.body_markdown) lines.push(section.body_markdown);
    if (section.deep_dive_markdown) {
      lines.push("");
      lines.push("**Deep dive**");
      lines.push("");
      lines.push(`> ${section.deep_dive_markdown.replace(/\n/g, "\n> ")}`);
    }
    if (section.media.length) {
      lines.push("");
      for (const m of section.media) {
        lines.push(`- **${m.label}** — ${m.caption}${m.filename ? ` (${m.filename})` : ""}`);
      }
    }
    if (section.citations.length) {
      lines.push("");
      lines.push("Sources:");
      for (const c of section.citations) {
        lines.push(`${c.marker}. ${c.source}, ${c.page_label} — "${c.snippet}"`);
      }
    }
  }

  lines.push("");
  lines.push("---");
  lines.push(
    `_Generated ${report.generated_at}${report.model_used ? ` · model: ${report.model_used}` : ""}_`
  );

  return lines.join("\n");
}

export function downloadTextFile(filename: string, content: string, mime: string): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Opens a print-formatted window and invokes the browser's print dialog
 *  (which offers "Save as PDF") — no PDF-generation dependency required. */
export function printReportAsPdf(report: Report): void {
  const paper = report.paper;
  const title = paper?.title ?? report.paper_id;
  const win = window.open("", "_blank", "noopener,noreferrer");
  if (!win) return;

  const sectionsHtml = report.sections
    .map((s) => {
      const bodyHtml = s.body_markdown
        ? `<p>${escapeHtml(s.body_markdown)
            .split(/\n{2,}/)
            .join("</p><p>")}</p>`
        : "";
      const deepHtml = s.deep_dive_markdown
        ? `<blockquote>${escapeHtml(s.deep_dive_markdown)}</blockquote>`
        : "";
      return `<h2>${escapeHtml(s.name)}</h2>${bodyHtml}${deepHtml}`;
    })
    .join("\n");

  const metaLine = paper
    ? `${escapeHtml(paper.arxiv_label)}${escapeHtml(paper.version ?? "")} · ${escapeHtml(
        paper.authors.join(", ")
      )}`
    : "";

  win.document.write(`<!doctype html><html><head><title>${escapeHtml(title)}</title>
<meta charset="utf-8" />
<style>
  body { font-family: Georgia, 'Source Serif 4', serif; max-width: 720px; margin: 40px auto; color: #171a21; line-height: 1.65; padding: 0 24px; }
  h1 { font-size: 26px; margin-bottom: 4px; }
  .meta { font-family: ui-monospace, monospace; font-size: 12px; color: #5a6472; margin-bottom: 24px; }
  h2 { font-size: 17px; margin-top: 30px; border-bottom: 1px solid #e3e6ec; padding-bottom: 6px; }
  p { font-size: 14px; }
  blockquote { color: #5a6472; border-left: 3px solid #d3d8e0; padding-left: 12px; margin-left: 0; font-size: 13.5px; }
</style>
</head><body>
<h1>${escapeHtml(title)}</h1>
<div class="meta">${metaLine}</div>
${sectionsHtml}
</body></html>`);
  win.document.close();
  win.focus();
  win.print();
}
