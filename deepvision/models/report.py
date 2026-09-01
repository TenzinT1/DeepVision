"""Report model — the interactive multimodal *study guide* for one paper.

This is the shape the Report screen renders and the shape the report/agents
layers must produce. A ``Report`` has a fixed set of eleven named sections (see
:class:`SectionName`), a stats bar, and rich per-section content: markdown body,
a "deep dive" extra body, provenance badges, inline citations, and media
(figure/table) references.

**Design premise: the reader is assumed to have NO background in the topic.**
The section set is a teaching arc, not a paper outline, and it is built on one
rule: *every section answers exactly one question no other section answers*.
That rule is not a comment — it is enforced by :data:`SECTION_QUESTIONS`, which
is fed verbatim into each section's generation prompt and rendered under each
section header in the UI. It exists because the previous section set had three
overlapping "what did they find / what does it mean" sections (Key Results,
Conclusions, Key Takeaways) and two overlapping vocabulary sections (Key
Concepts, Glossary), which produced reports that said the same thing four times.

The arc: **At a Glance** places the paper in five scannable lines · **Overview**
orients in prose · **Background** supplies the before-picture · **Key Concepts**
supplies the vocabulary · **Methods** and **Key Results** cover what was done
and found · **Figures** shows the evidence · **Limitations & Open Questions**
covers what the work does *not* establish · **Why It Matters** covers
consequences · **Key Takeaways** recaps once the reader has actually seen the
material · **Study Questions** lets them self-check.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepvision.models.chunks import BBox, Provenance

__all__ = [
    "SectionName",
    "SECTION_ORDER",
    "SECTION_QUESTIONS",
    "AT_A_GLANCE_FIELDS",
    "Citation",
    "MediaRef",
    "ReportStats",
    "Section",
    "ReportScope",
    "Report",
    "ChapterSource",
    "Chapter",
]


class SectionName(str, Enum):
    """The eleven fixed report sections, in display order.

    Values are the exact headings shown in the UI and are the wire contract —
    the frontend switches on them (``frontend/src/api/types.ts``
    ``SectionName``), so they must never be renamed.

    The set is built for a reader with **no background in the topic**, and reads
    top-to-bottom as a lesson. Each entry below states the section's job; the
    single question it answers (and which no other section may answer) is in
    :data:`SECTION_QUESTIONS`.

    - ``At a Glance`` — the paper placed in five scannable labelled lines
      (:data:`AT_A_GLANCE_FIELDS`: Problem / Approach / Evidence / Headline
      result / Read this if). Fixed-schema and one line per field, so it is a
      *fact card*, not a third prose summary — that fixed shape is what keeps it
      distinct from Overview and Key Takeaways, and it survives full LLM
      degradation because the field labels are generated deterministically.
    - ``Overview`` — orients in flowing prose: what this paper is and why it
      exists. Seeded from the abstract.
    - ``Background`` — the situation before this paper: the problem, prior
      approaches, and why they fell short. Never this paper's own contribution.
    - ``Key Concepts`` — the vocabulary and ideas the reader must hold before
      Methods can make sense, each written ``**Term** — explanation from zero.``
      **This section absorbed the former ``Glossary``**: a separate glossary
      produced near-identical term/definition pairs, so the two were merged and
      the extraction that fed the glossary
      (``deepvision.agents.study_agent.extract_terms``) now anchors this
      section's draft.
    - ``Methods`` — what the authors actually did, in order.
    - ``Key Results`` — what they found: the numbers, the baselines they are
      measured against, and how strong the evidence is. Findings only; what the
      findings *mean* belongs to ``Why It Matters``.
    - ``Figures`` — the paper's figures/tables explained (media cards live
      here). Other sections may also carry media: any figure a section's prose
      references by number is attached to it by
      ``deepvision.report.figure_links.attach_referenced_figures``, so "Figure 6"
      in the body renders as the actual image rather than as a bare caption line.
    - ``Limitations & Open Questions`` — what this work does **not** establish:
      stated limitations, threats to validity, untested conditions, and what the
      authors leave unresolved. Split out of the former ``Conclusions``, which
      used to carry it implicitly and therefore usually dropped it.
    - ``Why It Matters`` — consequences: what changes because of this work, who
      it affects, what it enables next. Replaces ``Conclusions``, whose name
      invited a restatement of Key Results instead of an argument for
      significance.
    - ``Key Takeaways`` — the handful of claims worth remembering a month from
      now, each a self-contained sentence with an inline ``[n]`` citation. It
      sits **after** the evidence on purpose: as the opening section it was a
      preview of Overview, and as a recap it is a genuine synthesis. It remains
      the highest-value seed for auto-generated flashcards and quiz questions
      (see ``deepvision/models/study.py``).
    - ``Study Questions`` — self-check questions over the material above, so the
      reader can verify they actually understood it. Answers live in
      ``deep_dive_markdown``.
    """

    AT_A_GLANCE = "At a Glance"
    OVERVIEW = "Overview"
    BACKGROUND = "Background"
    KEY_CONCEPTS = "Key Concepts"
    METHODS = "Methods"
    KEY_RESULTS = "Key Results"
    FIGURES = "Figures"
    LIMITATIONS = "Limitations & Open Questions"
    WHY_IT_MATTERS = "Why It Matters"
    KEY_TAKEAWAYS = "Key Takeaways"
    STUDY_QUESTIONS = "Study Questions"


#: Canonical section order for rendering and generation. Mirrored verbatim by
#: ``SECTION_ORDER`` in ``frontend/src/api/types.ts``.
SECTION_ORDER: list[SectionName] = [
    SectionName.AT_A_GLANCE,
    SectionName.OVERVIEW,
    SectionName.BACKGROUND,
    SectionName.KEY_CONCEPTS,
    SectionName.METHODS,
    SectionName.KEY_RESULTS,
    SectionName.FIGURES,
    SectionName.LIMITATIONS,
    SectionName.WHY_IT_MATTERS,
    SectionName.KEY_TAKEAWAYS,
    SectionName.STUDY_QUESTIONS,
]


#: The one question each section answers — and which **no other section may
#: answer**. This is the anti-redundancy contract, and it is load-bearing in
#: three places at once:
#:
#: 1. ``agents/summarizer_agent.py`` puts the question at the top of that
#:    section's generation prompt, and the *other* sections' questions in a
#:    "these are handled elsewhere, do not answer them here" list.
#: 2. ``report/interactive_sections.py::build_section`` stamps it onto
#:    :attr:`Section.question`, and ``normalize_sections`` backfills it on load
#:    so reports persisted before this field existed still render it.
#: 3. The Report screen renders it under the section heading, so the reader
#:    knows what each section is for.
#:
#: Every name in :data:`SECTION_ORDER` must have an entry.
SECTION_QUESTIONS: dict[SectionName, str] = {
    SectionName.AT_A_GLANCE: "What is this paper, in five lines?",
    SectionName.OVERVIEW: "What is this paper, and why does it exist?",
    SectionName.BACKGROUND: "What came before this paper, and why wasn't it enough?",
    SectionName.KEY_CONCEPTS: (
        "Which ideas and terms do I need before the methods make sense?"
    ),
    SectionName.METHODS: "What did the authors actually do?",
    SectionName.KEY_RESULTS: "What did they find, and how strong is the evidence?",
    SectionName.FIGURES: "What do the paper's figures and tables show?",
    SectionName.LIMITATIONS: (
        "What does this work not show, and what is still unresolved?"
    ),
    SectionName.WHY_IT_MATTERS: "So what — what changes because of this work?",
    SectionName.KEY_TAKEAWAYS: (
        "If I remember only a few things from this paper, what should they be?"
    ),
    SectionName.STUDY_QUESTIONS: "Can I explain this back without looking?",
}


#: The fixed field labels of the ``At a Glance`` fact card, in order, each with
#: the one-line brief handed to the generator. The labels are emitted
#: deterministically as ``**Label** — value``, so the card keeps its shape even
#: when every model call falls back; only the values are LLM-written. Changing
#: this list changes what the section *is* — keep it to five scannable lines.
AT_A_GLANCE_FIELDS: list[tuple[str, str]] = [
    ("Problem", "the specific problem the paper attacks, in one line"),
    ("Approach", "what the authors propose, in one line"),
    ("Evidence", "what they tested it on — data, benchmark, or setup"),
    ("Headline result", "the single most important number or outcome"),
    ("Read this if", "who benefits from reading this paper, and why"),
]


class Citation(BaseModel):
    """A resolvable citation shown in the citation popover.

    Every inline ``[n]`` marker in section markdown resolves to one of these.
    The popover renders ``source`` + ``page_label`` + the page image (via
    ``page_image_path``) + the exact ``snippet``, highlighting ``bbox``.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(..., description="Citation id (see utils.ids.citation_id).")
    marker: int = Field(
        ..., ge=1, description="The inline number, e.g. 1 for '[1]'."
    )
    source: str = Field(
        ...,
        description="Source label, e.g. 'Section 1 · Introduction'.",
        examples=["Section 1 · Introduction"],
    )
    page: int = Field(..., ge=1, description="1-based page number.")
    page_label: str = Field(
        ..., description="Human page label, e.g. 'page 2'.", examples=["page 2"]
    )
    page_image_path: Optional[str] = Field(
        default=None,
        description="Path (relative to data_dir) to the rendered page image.",
    )
    snippet: str = Field(..., description="Exact quoted snippet from the source.")
    bbox: Optional[BBox] = Field(
        default=None, description="Region of the snippet on the page image."
    )
    chunk_id: Optional[str] = Field(
        default=None, description="The chunk this citation was grounded in."
    )


class MediaRef(BaseModel):
    """A figure or table reference embedded in a section (esp. Figures).

    Renders as a card with the full ``image_path``, a ``thumbnail_path`` preview,
    the ``caption``, and a provenance badge (VISION vs OCR) via ``provenance``.
    Clicking opens the lightbox with the full image.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(..., description="Media ref id (see utils.ids.media_id).")
    kind: str = Field(
        default="figure", description="'figure' or 'table'.", examples=["figure"]
    )
    label: str = Field(
        ..., description="Display label, e.g. 'Fig 2' or 'Table 4'.", examples=["Fig 2"]
    )
    caption: str = Field(default="", description="Caption / description text.")
    provenance: Provenance = Field(
        ..., description="Which processor produced it (VISION or OCR)."
    )
    image_path: Optional[str] = Field(
        default=None, description="Full-size image path, relative to data_dir."
    )
    thumbnail_path: Optional[str] = Field(
        default=None, description="Thumbnail image path, relative to data_dir."
    )
    filename: Optional[str] = Field(
        default=None, description="Original asset filename, e.g. 'scaling-curves.png'."
    )
    page: Optional[int] = Field(default=None, ge=1, description="Source page number.")
    chunk_id: Optional[str] = Field(
        default=None, description="Backing image/vision chunk id (for /media/{id})."
    )


class ReportStats(BaseModel):
    """The stats bar at the top of a report."""

    pages: int = Field(default=0, ge=0, description="Total page count.")
    figures: int = Field(default=0, ge=0, description="Figures extracted.")
    citations_extracted: int = Field(
        default=0, ge=0, description="Number of citations extracted."
    )
    reading_time_min: int = Field(
        default=0, ge=0, description="Estimated reading time in minutes."
    )


class Section(BaseModel):
    """One named, collapsible report section.

    ``body_markdown`` is the primary content (may contain inline ``[n]`` citation
    markers). ``deep_dive_markdown`` is the extra detail revealed by the
    "+ Deep dive" toggle. ``badge`` is an optional header pill (e.g. 'SUMMARY',
    'OCR-DERIVED'); ``provenance`` lists the badge kinds present in the block so
    the UI can colour inline text/OCR/VISION spans. ``media`` holds figure/table
    cards (used mainly by the Figures section). ``confidence`` is the generator's
    self-assessed confidence in ``[0, 1]``.

    ``question`` is the one question this section answers, from
    :data:`SECTION_QUESTIONS`; the UI renders it under the heading. ``degraded``
    is ``True`` when at least one LLM call for this section fell back to the
    deterministic extractive draft — i.e. the text below is quoted from the PDF,
    not written by a model. It is the per-section counterpart of
    ``StageProgress.degraded`` and exists because a report read a day later
    otherwise gives the reader no way to tell polished prose from raw extract.

    Body conventions for the shaped sections (plain markdown, so the existing
    renderer needs no special case):

    - ``At a Glance`` — exactly the five ``**Label** — value`` lines of
      :data:`AT_A_GLANCE_FIELDS`, in that order, one line each. No prose.
    - ``Key Concepts`` — a markdown list, one item per concept, each written as
      ``**Term** — explanation from zero.``
    - ``Key Takeaways`` — 3-8 items, each one a single self-contained claim
      (one sentence, ideally with the number that makes it concrete) carrying an
      inline ``[n]`` citation. No sub-bullets.
    - ``Study Questions`` — a numbered markdown list of questions; put the
      answers in ``deep_dive_markdown`` so the reader can attempt them first and
      reveal answers via the existing "+ Deep dive" toggle.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(..., description="Section id (see utils.ids.section_id).")
    name: SectionName = Field(..., description="One of the eleven fixed section names.")
    question: Optional[str] = Field(
        default=None,
        description=(
            "The one question this section answers (SECTION_QUESTIONS), shown "
            "under the heading. Backfilled on load for older reports."
        ),
    )
    body_markdown: str = Field(default="", description="Primary section body (markdown).")
    deep_dive_markdown: Optional[str] = Field(
        default=None, description="Extra detail for the 'Deep dive' expansion."
    )
    badge: Optional[str] = Field(
        default=None,
        description="Optional header pill label, e.g. 'SUMMARY' or 'OCR-DERIVED'.",
    )
    provenance: list[Provenance] = Field(
        default_factory=list,
        description="Provenance kinds present in this section (for badge colouring).",
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Citations referenced from this section."
    )
    media: list[MediaRef] = Field(
        default_factory=list, description="Figures/tables shown in this section."
    )
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Generator self-confidence."
    )
    degraded: bool = Field(
        default=False,
        description=(
            "True if an LLM call for this section fell back to the extractive "
            "draft — the body is quoted from the PDF, not model-written."
        ),
    )
    default_open: bool = Field(
        default=False, description="Whether the section starts expanded in the UI."
    )


class ChapterSource(str, Enum):
    """Where a :class:`Chapter` list was derived from.

    Values are the wire contract (mirrored in ``frontend/src/api/types.ts``) and
    must never be renamed.

    - ``outline`` — the PDF's own embedded outline/bookmarks
      (``fitz.Document.get_toc()``). Real books, theses and reports have one and
      it is authoritative: exact titles, exact start pages, real nesting depth.
    - ``headings`` — every non-outline derivation: primarily the ingestion
      heading heuristic (``TextChunk.source_ref.section_label``, produced by
      ``ingestion/pdf_parser.py::_looks_like_heading``) taken in page order —
      the arXiv case, since papers almost never ship an outline but do have
      detectable headings — and also the last-resort ``"Pages 1-10"`` buckets
      ``ingestion/chapters.py`` falls back to when nothing else is detectable.
      Always ``level == 1`` (flat).
    """

    OUTLINE = "outline"
    HEADINGS = "headings"


class Chapter(BaseModel):
    """One selectable chapter of a paper's PDF, shown in the report's picker.

    A chapter is a **contiguous page range**, and that is the whole point: the
    page range is what makes retrieval scoping possible (see
    ``PageScopedRetriever`` in ``deepvision/rag/chapter_scope.py``). A chapter's
    ``page_end`` is the page *before* the next chapter at the same-or-shallower
    level starts; the final chapter runs to the document's last page.

    ``id`` must be **stable across re-derivations** — it is half of the
    ``chapter_reports`` primary key, so a chapter report generated yesterday
    must still resolve today. Build it with :meth:`make_id`, never with a random
    id helper.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(
        ...,
        description="Stable chapter id (see Chapter.make_id).",
        examples=["ch-3f1a9c2b7d04"],
    )
    index: int = Field(
        ..., ge=0, description="0-based position in the chapter list (display order)."
    )
    title: str = Field(
        ...,
        description="Chapter title as printed in the PDF.",
        examples=["3 Methods"],
    )
    level: int = Field(
        default=1,
        ge=1,
        description="Outline depth: 1 = top-level chapter, 2 = subsection. Always 1 for 'headings'.",
    )
    page_start: int = Field(..., ge=1, description="1-based first page of the chapter.")
    page_end: int = Field(
        ..., ge=1, description="1-based last page (inclusive); page_count for the last chapter."
    )
    source: ChapterSource = Field(
        ..., description="How this chapter was detected ('outline' | 'headings')."
    )

    @model_validator(mode="after")
    def _check_page_range(self) -> "Chapter":
        if self.page_end < self.page_start:
            raise ValueError(
                f"chapter page_end ({self.page_end}) precedes page_start ({self.page_start})"
            )
        return self

    @property
    def page_count(self) -> int:
        """Number of pages this chapter spans (inclusive)."""
        return self.page_end - self.page_start + 1

    @staticmethod
    def make_id(paper_id: str, title: str, page_start: int, level: int = 1) -> str:
        """Return the deterministic id for a chapter of ``paper_id``.

        Hashes the *normalized* title together with the start page and level, so
        re-deriving the chapter list from the same PDF always yields the same
        ids (and therefore resolves the same persisted ``ChapterReportRow``),
        while two chapters that merely share a title on different pages stay
        distinct. Format: ``ch-<sha1 hex[:12]>``.
        """
        norm = re.sub(r"\s+", " ", (title or "").strip().casefold())
        raw = f"{paper_id}|{level}|{page_start}|{norm}"
        return f"ch-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


class ReportScope(str, Enum):
    """What a :class:`Report` covers — the whole paper, or one chapter of it.

    Values are the wire contract (mirrored in ``frontend/src/api/types.ts``).
    A chapter report has the **same eleven-section shape** as a paper report, on
    purpose: the Report page renders both with the same ``SectionCard``
    components and only swaps the header line ("Reporting on: 3 Methods").
    Never build a parallel render path for chapter reports.
    """

    PAPER = "paper"
    CHAPTER = "chapter"


class Report(BaseModel):
    """The complete interactive report returned by ``GET /report/{paper_id}``.

    Also the shape returned by the per-chapter endpoints
    (``GET /report/{paper_id}/chapter/{chapter_id}``) — identical sections,
    stats and citations, with ``scope == ReportScope.CHAPTER`` and the
    ``chapter_*`` fields populated. Everything scope-related is optional with a
    ``paper``-scoped default, so reports persisted before this feature existed
    validate unchanged.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(..., description="Report id (see utils.ids.report_id).")
    paper_id: str = Field(..., description="Owning paper id.")
    paper: Optional["PaperMeta"] = Field(
        default=None, description="Denormalized paper metadata for the report header."
    )
    stats: ReportStats = Field(
        default_factory=ReportStats, description="Stats bar values."
    )
    sections: list[Section] = Field(
        default_factory=list, description="Sections in SECTION_ORDER."
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: Optional[str] = Field(
        default=None, description="LLM identifier that generated the summary."
    )

    scope: ReportScope = Field(
        default=ReportScope.PAPER,
        description="'paper' (whole document) or 'chapter' (one chapter of it).",
    )
    chapter_id: Optional[str] = Field(
        default=None, description="Chapter.id — set iff scope == 'chapter'."
    )
    chapter_title: Optional[str] = Field(
        default=None,
        description="Chapter title for the 'Reporting on: …' header — set iff scope == 'chapter'.",
    )
    chapter_page_start: Optional[int] = Field(
        default=None, ge=1, description="First page of the chapter — set iff scope == 'chapter'."
    )
    chapter_page_end: Optional[int] = Field(
        default=None, ge=1, description="Last page of the chapter — set iff scope == 'chapter'."
    )

    def section(self, name: SectionName) -> Optional[Section]:
        """Return the section with ``name``, or ``None`` if absent."""
        return next((s for s in self.sections if s.name == name), None)


# Resolve the forward reference to PaperMeta without a hard import cycle at
# module top (report is imported widely; paper imports nothing from report).
from deepvision.models.paper import PaperMeta  # noqa: E402

Report.model_rebuild()
