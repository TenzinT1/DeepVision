"""Exporters — Markdown and PDF export of a Report.

Renders a :class:`Report` to a self-contained Markdown string (figures
embedded as base64 data URIs so the file has no external dependencies) and to
PDF via WeasyPrint. The WeasyPrint import is guarded so this module — and
anything that imports it — still imports cleanly when the optional
``weasyprint`` package (and its native cairo/pango deps) isn't installed;
``to_pdf`` raises a clear, actionable error in that case instead of crashing
at import time.
"""

from __future__ import annotations

import abc
import base64
import html as html_lib
import mimetypes
import re
from typing import Optional

from deepvision.config import get_config
from deepvision.models import MediaRef, Report, Section
from deepvision.utils import get_logger

__all__ = ["ReportExporter", "DefaultReportExporter"]

log = get_logger(__name__)

try:  # pragma: no cover - exercised only when the optional dep is installed
    import weasyprint  # type: ignore

    _HAS_WEASYPRINT = True
except Exception:  # noqa: BLE001 - any failure means "treat as unavailable"
    weasyprint = None  # type: ignore[assignment]
    _HAS_WEASYPRINT = False

try:  # pragma: no cover - optional, nicer markdown->HTML than the fallback below
    from markdown_it import MarkdownIt  # type: ignore

    _MD = MarkdownIt("commonmark")
except Exception:  # noqa: BLE001
    _MD = None


class ReportExporter(abc.ABC):
    """Serializes a report to downloadable formats."""

    @abc.abstractmethod
    def to_markdown(self, report: Report) -> str:
        """Render ``report`` to a Markdown string."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_pdf(self, report: Report) -> bytes:
        """Render ``report`` to PDF bytes (via WeasyPrint)."""
        raise NotImplementedError


def _embed_data_uri(rel_path: Optional[str]) -> Optional[str]:
    """Read ``rel_path`` (relative to ``data_dir``) and return a base64 data URI."""
    if not rel_path:
        return None
    abs_path = get_config().data_dir / rel_path
    try:
        data = abs_path.read_bytes()
    except OSError:
        log.warning("export: media file missing on disk", extra={"path": str(abs_path)})
        return None
    mime = mimetypes.guess_type(str(abs_path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _markdown_to_html(markdown: str) -> str:
    """Render ``markdown`` to HTML for PDF export.

    Prefers ``markdown-it-py`` (listed in requirements.txt) when installed;
    falls back to a minimal, dependency-free converter for the subset the
    agents actually emit (paragraphs, **bold**, *italic*, inline ``[n]``
    citation markers) so this module still works without the optional dep.
    """
    if _MD is not None:
        return _MD.render(markdown.strip())
    paragraphs = [p.strip() for p in markdown.strip().split("\n\n") if p.strip()]
    rendered = []
    for para in paragraphs:
        text = html_lib.escape(para)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
        text = re.sub(r"\[(\d+)\]", r"<sup>[\1]</sup>", text)
        text = text.replace("\n", "<br>")
        rendered.append(f"<p>{text}</p>")
    return "\n".join(rendered)


class DefaultReportExporter(ReportExporter):
    """Renders a :class:`Report` to Markdown or PDF, embedding figures + citations."""

    # ---- Markdown ---------------------------------------------------
    def to_markdown(self, report: Report) -> str:
        lines: list[str] = []
        title = report.paper.title if report.paper else report.paper_id
        lines.append(f"# {title}")
        if report.paper:
            if report.paper.authors:
                lines.append(f"*{', '.join(report.paper.authors)}*")
            meta_bits = [
                b
                for b in (
                    report.paper.arxiv_label,
                    ", ".join(report.paper.categories) if report.paper.categories else "",
                    str(report.paper.published) if report.paper.published else "",
                )
                if b
            ]
            if meta_bits:
                lines.append(" · ".join(meta_bits))
        lines.append("")
        s = report.stats
        lines.append(
            f"**{s.pages} pages** · **{s.figures} figures** · "
            f"**{s.citations_extracted} citations** · **{s.reading_time_min} min read**"
        )
        if report.model_used:
            lines.append(f"*Generated with {report.model_used}*")
        lines.append("")
        for section in report.sections:
            lines.extend(self._section_markdown(section))
        return "\n".join(lines).strip() + "\n"

    def _section_markdown(self, section: Section) -> list[str]:
        heading = f"## {section.name.value}"
        if section.badge:
            heading += f" — {section.badge}"
        out = [heading, ""]
        if section.body_markdown:
            out.append(section.body_markdown.strip())
            out.append("")
        if section.deep_dive_markdown:
            out.append("### Deep dive")
            out.append(section.deep_dive_markdown.strip())
            out.append("")
        for media in section.media:
            out.extend(self._media_markdown(media))
        if section.citations:
            out.append("**Citations:**")
            for cit in section.citations:
                out.append(f'[{cit.marker}] {cit.source} ({cit.page_label}): "{cit.snippet}"')
            out.append("")
        return out

    def _media_markdown(self, media: MediaRef) -> list[str]:
        data_uri = _embed_data_uri(media.image_path) or _embed_data_uri(media.thumbnail_path)
        alt = media.caption or media.label
        src = data_uri or (media.image_path or media.thumbnail_path or "")
        out = [f"![{alt}]({src})"]
        caption_line = f"*{media.label}"
        if media.caption:
            caption_line += f": {media.caption}"
        caption_line += "*"
        out.append(caption_line)
        out.append("")
        return out

    # ---- PDF ----------------------------------------------------------
    def to_pdf(self, report: Report) -> bytes:
        if not _HAS_WEASYPRINT:
            raise RuntimeError(
                "PDF export requires the optional 'weasyprint' package (and its "
                "system cairo/pango libraries). Install it with "
                "`pip install weasyprint` to enable this feature."
            )
        html_doc = self._to_html(report)
        return weasyprint.HTML(  # type: ignore[union-attr]
            string=html_doc, base_url=str(get_config().data_dir)
        ).write_pdf()

    def _to_html(self, report: Report) -> str:
        title = report.paper.title if report.paper else report.paper_id
        parts: list[str] = [
            "<html><head><meta charset='utf-8'>",
            f"<title>{html_lib.escape(title)}</title>",
            "<style>"
            "body{font-family:Georgia,serif;max-width:800px;margin:2rem auto;color:#1a1a1a;}"
            "img{max-width:100%;}"
            "h1{font-size:1.6rem;} h2{border-bottom:1px solid #ccc;padding-bottom:4px;margin-top:2rem;}"
            "h3{color:#444;}"
            ".badge{font-size:.65em;color:#666;border:1px solid #ccc;border-radius:4px;"
            "padding:1px 6px;margin-left:8px;vertical-align:middle;}"
            ".stats{color:#555;}"
            ".citation{font-size:.85em;color:#444;margin:4px 0;}"
            "figure{margin:1rem 0;} figcaption{font-size:.85em;color:#555;}"
            "</style></head><body>",
            f"<h1>{html_lib.escape(title)}</h1>",
        ]
        if report.paper and report.paper.authors:
            parts.append(f"<p><em>{html_lib.escape(', '.join(report.paper.authors))}</em></p>")
        s = report.stats
        parts.append(
            f"<p class='stats'>{s.pages} pages · {s.figures} figures · "
            f"{s.citations_extracted} citations · {s.reading_time_min} min read</p>"
        )
        if report.model_used:
            parts.append(f"<p class='stats'><em>Generated with {html_lib.escape(report.model_used)}</em></p>")
        for section in report.sections:
            heading = f"<h2>{html_lib.escape(section.name.value)}"
            if section.badge:
                heading += f"<span class='badge'>{html_lib.escape(section.badge)}</span>"
            heading += "</h2>"
            parts.append(heading)
            if section.body_markdown:
                parts.append(_markdown_to_html(section.body_markdown))
            if section.deep_dive_markdown:
                parts.append("<h3>Deep dive</h3>")
                parts.append(_markdown_to_html(section.deep_dive_markdown))
            for media in section.media:
                data_uri = _embed_data_uri(media.image_path) or _embed_data_uri(media.thumbnail_path)
                if not data_uri:
                    continue
                caption = html_lib.escape(media.caption or "")
                label = html_lib.escape(media.label)
                parts.append(
                    f"<figure><img src='{data_uri}' alt='{label}'>"
                    f"<figcaption>{label} — {caption}</figcaption></figure>"
                )
            if section.citations:
                parts.append("<div class='citations'>")
                for cit in section.citations:
                    parts.append(
                        f"<p class='citation'>[{cit.marker}] {html_lib.escape(cit.source)} "
                        f"({html_lib.escape(cit.page_label)}): "
                        f"&ldquo;{html_lib.escape(cit.snippet)}&rdquo;</p>"
                    )
                parts.append("</div>")
        parts.append("</body></html>")
        return "\n".join(parts)
