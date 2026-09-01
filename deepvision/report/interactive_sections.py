"""Interactive sections — section assembly, media serving, citation resolution.

Three responsibilities live here (all builder-owned, REPORT):

1. Section-assembly helpers (:func:`build_section`, :func:`normalize_sections`)
   used by :mod:`report_generator` — and available to the agents layer's agents — to
   build UI-ready :class:`Section` objects with the correct ``badge`` /
   ``default_open`` / ``provenance`` per the fixed report design (Overview and
   Background start expanded; ``provenance`` drives the inline
   text/OCR/VISION colour coding).
2. :class:`DefaultMediaService` — serves figure/table/page image bytes (or
   their :class:`MediaMetadata`) for ``GET /media/{chunk_id}``, resolving the
   chunk id against whichever persisted report references it and reading the
   bytes from the local file store (``config.data_dir``).
3. :class:`DefaultCitationResolver` — resolves a citation id back to its full
   popover payload.
"""

from __future__ import annotations

import abc
import mimetypes
from pathlib import Path
from typing import Any, Optional, Sequence

from sqlalchemy import String, cast
from sqlmodel import select

from deepvision.api.schemas import MediaMetadata
from deepvision.config import get_config
from deepvision.db import session_scope
from deepvision.db.schema import ReportRow
from deepvision.models import (
    SECTION_ORDER,
    Citation,
    MediaRef,
    Provenance,
    Section,
    SectionName,
)
from deepvision.models.report import SECTION_QUESTIONS
from deepvision.utils import get_logger
from deepvision.utils.ids import section_id

__all__ = [
    "MediaService",
    "CitationResolver",
    "DefaultMediaService",
    "DefaultCitationResolver",
    "DEFAULT_OPEN_SECTIONS",
    "SECTION_BADGES",
    "build_section",
    "normalize_sections",
    "sections_from_rows",
    "provenance_from_media",
]

log = get_logger(__name__)

#: Sections that start expanded in the UI, per the report design.
#:
#: The report reads as a lesson for someone with no background in the topic, so
#: the three sections that orient that reader — the fact card (At a Glance),
#: what this paper is (Overview), and what came before it (Background) — are
#: open on arrival and everything else is collapsed. Opening more than that
#: buries the section list itself, which is the reader's map; Key Results in
#: particular is meaningless before Methods, and Key Takeaways is now a *recap*
#: that only lands after the reader has seen the material, so neither starts
#: open.
DEFAULT_OPEN_SECTIONS: frozenset[SectionName] = frozenset(
    {SectionName.AT_A_GLANCE, SectionName.OVERVIEW, SectionName.BACKGROUND}
)

#: Fallback header pill per section when a generator doesn't set one
#: explicitly (``None`` means "no badge"). Covers all eleven names in
#: :data:`SECTION_ORDER`.
SECTION_BADGES: dict[SectionName, Optional[str]] = {
    SectionName.AT_A_GLANCE: "FACT CARD",
    SectionName.OVERVIEW: "SUMMARY",
    SectionName.BACKGROUND: "CONTEXT",
    SectionName.KEY_CONCEPTS: "CONCEPTS",
    SectionName.METHODS: None,
    SectionName.KEY_RESULTS: "KEY RESULTS",
    SectionName.FIGURES: "FIGURES",
    SectionName.LIMITATIONS: "CAVEATS",
    SectionName.WHY_IT_MATTERS: "IMPACT",
    SectionName.KEY_TAKEAWAYS: "RECAP",
    SectionName.STUDY_QUESTIONS: "SELF-CHECK",
}

_UNSET: Any = object()


class MediaService(abc.ABC):
    """Serves figure/table/page image assets from the local store."""

    @abc.abstractmethod
    def get_bytes(self, chunk_id: str) -> tuple[bytes, str]:
        """Return ``(image_bytes, content_type)`` for a media chunk."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_metadata(self, chunk_id: str) -> MediaMetadata:
        """Return the :class:`MediaMetadata` sidecar for a media chunk."""
        raise NotImplementedError


class CitationResolver(abc.ABC):
    """Resolves a citation id to its full popover payload."""

    @abc.abstractmethod
    def resolve(self, citation_id: str) -> Citation:
        """Return the fully-populated :class:`Citation` (snippet, page image, bbox)."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# Section assembly helpers
# --------------------------------------------------------------------------
def provenance_from_media(media: Sequence[MediaRef]) -> list[Provenance]:
    """Ordered, de-duplicated provenance kinds present in ``media``."""
    seen: list[Provenance] = []
    for m in media:
        if m.provenance not in seen:
            seen.append(m.provenance)
    return seen


def build_section(
    *,
    name: SectionName,
    body_markdown: str = "",
    deep_dive_markdown: Optional[str] = None,
    citations: Sequence[Citation] = (),
    media: Sequence[MediaRef] = (),
    confidence: Optional[float] = None,
    provenance: Optional[Sequence[Provenance]] = None,
    badge: Any = _UNSET,
    degraded: bool = False,
) -> Section:
    """Assemble one :class:`Section` per the fixed report design.

    ``default_open`` always follows :data:`DEFAULT_OPEN_SECTIONS` and
    ``question`` always follows
    :data:`~deepvision.models.report.SECTION_QUESTIONS` — neither is a caller
    decision. ``badge`` falls back to :data:`SECTION_BADGES` unless explicitly
    passed (pass ``badge=None`` to force "no badge"). ``provenance`` defaults to
    the provenance kinds present in ``media``, falling back to ``[TEXT]`` when
    there is a body but no other signal (citations don't carry a provenance
    field of their own — see :class:`~deepvision.models.report.Citation`).
    """
    resolved_badge = SECTION_BADGES.get(name) if badge is _UNSET else badge
    if provenance is not None:
        resolved_provenance = list(dict.fromkeys(provenance))
    else:
        resolved_provenance = provenance_from_media(media)
        if not resolved_provenance and body_markdown.strip():
            resolved_provenance = [Provenance.TEXT]
    return Section(
        id=section_id(),
        name=name,
        question=SECTION_QUESTIONS.get(name),
        body_markdown=body_markdown,
        deep_dive_markdown=deep_dive_markdown,
        badge=resolved_badge,
        provenance=resolved_provenance,
        citations=list(citations),
        media=list(media),
        confidence=confidence,
        degraded=degraded,
        default_open=name in DEFAULT_OPEN_SECTIONS,
    )


def sections_from_rows(rows: Sequence[Any]) -> list[Section]:
    """Validate persisted section dicts, **skipping names that no longer exist**.

    This is the read side of a section-set change, and it must never raise.
    ``SectionName`` is a closed enum, so ``Section.model_validate`` on a row
    persisted under an older set raises ``ValidationError`` — and that is not a
    theoretical concern: when ``Glossary`` and ``Conclusions`` were retired,
    every report already in the library started answering ``GET /report`` with a
    500 instead of rendering. ``normalize_sections`` is documented to drop an
    unknown section, but it never got the chance, because validation runs first.

    A retired section is dropped with a log line; :func:`normalize_sections`
    then backfills whatever the current :data:`SECTION_ORDER` is missing. The
    reader sees an empty placeholder for the new sections and no trace of the
    old ones, which is exactly the intended migration — re-run generation to
    fill them in.
    """
    known = {name.value for name in SectionName}
    out: list[Section] = []
    skipped: list[str] = []
    for row in rows or []:
        name = row.get("name") if isinstance(row, dict) else getattr(row, "name", None)
        if isinstance(name, str) and name not in known:
            skipped.append(name)
            continue
        try:
            out.append(Section.model_validate(row))
        except Exception:  # noqa: BLE001 - one bad row must not 500 the report
            skipped.append(str(name))
            log.warning("dropping unreadable persisted section", extra={"name": name})
    if skipped:
        log.info(
            "dropped %d persisted section(s) no longer in SECTION_ORDER: %s",
            len(skipped),
            ", ".join(skipped),
        )
    return out


def normalize_sections(sections: Sequence[Section]) -> list[Section]:
    """Reorder ``sections`` to :data:`SECTION_ORDER`, enforcing design defaults.

    - Any missing section (of the eleven fixed names) is filled with an empty
      placeholder via :func:`build_section`. This is also what backfills newly
      added sections onto a report persisted before they existed — the
      placeholders are empty, so a re-run is needed for real content. A report
      persisted under the old section set therefore comes back with empty
      ``At a Glance`` / ``Limitations & Open Questions`` / ``Why It Matters``
      cards, and its now-removed ``Glossary`` / ``Conclusions`` sections are
      dropped (a name not in :data:`SECTION_ORDER` has nowhere to render).
    - Every section is assigned an id if it doesn't have one.
    - ``default_open`` is (re)applied per :data:`DEFAULT_OPEN_SECTIONS`
      regardless of what produced the section, so the UI contract holds even
      if an upstream agent forgot to set it.
    - ``question`` is (re)applied per
      :data:`~deepvision.models.report.SECTION_QUESTIONS`, which is what
      backfills it onto reports persisted before the field existed.
    - A missing ``badge`` (``None``) is filled from :data:`SECTION_BADGES`;
      an explicitly-set badge (including one an agent intentionally cleared)
      is left alone.
    """
    by_name = {s.name: s for s in sections}
    ordered: list[Section] = []
    for name in SECTION_ORDER:
        sec = by_name.get(name)
        if sec is None:
            sec = build_section(name=name)
        else:
            if not sec.id:
                sec.id = section_id()
            sec.default_open = name in DEFAULT_OPEN_SECTIONS
            sec.question = SECTION_QUESTIONS.get(name)
            if sec.badge is None:
                sec.badge = SECTION_BADGES.get(name)
        ordered.append(sec)
    return ordered


# --------------------------------------------------------------------------
# Media / citation lookups
# --------------------------------------------------------------------------
def _media_from_row(row: ReportRow, chunk_id: str) -> Optional[tuple[str, dict[str, Any]]]:
    """Scan a single :class:`ReportRow` for a MediaRef/Citation with ``chunk_id``."""
    for sec in row.sections or []:
        for media in sec.get("media", []) or []:
            if media.get("chunk_id") == chunk_id:
                return row.paper_id, {
                    "kind": media.get("kind") or "figure",
                    "label": media.get("label"),
                    "caption": media.get("caption"),
                    "provenance": media.get("provenance"),
                    "page": media.get("page"),
                    "image_path": media.get("image_path"),
                    "thumbnail_path": media.get("thumbnail_path"),
                }
        for cit in sec.get("citations", []) or []:
            if cit.get("chunk_id") == chunk_id:
                return row.paper_id, {
                    "kind": "page",
                    "label": cit.get("page_label"),
                    "caption": None,
                    "provenance": Provenance.TEXT.value,
                    "page": cit.get("page"),
                    "image_path": cit.get("page_image_path"),
                    "thumbnail_path": None,
                }
    return None


def _find_media(chunk_id: str) -> tuple[str, dict[str, Any]]:
    """Look up the MediaRef/Citation carrying ``chunk_id`` in its owning report.

    ``chunk_id`` is always formatted ``<paper_id>::c<ordinal>`` (see
    :func:`deepvision.utils.ids.chunk_id`), so the owning report can be
    fetched directly by ``paper_id`` instead of scanning every persisted
    report in the library. Falls back to a full scan if the id doesn't carry
    a recognizable paper_id prefix or that report doesn't (yet) contain the
    reference, so this stays correct even for unexpected id shapes.

    Returns ``(paper_id, info)`` where ``info`` has ``kind``, ``label``,
    ``caption``, ``provenance``, ``page``, ``image_path``, ``thumbnail_path``.
    Raises :class:`LookupError` if no reference is found anywhere.
    """
    paper_id = chunk_id.split("::", 1)[0] if "::" in chunk_id else None

    with session_scope() as session:
        if paper_id:
            row = session.exec(
                select(ReportRow).where(ReportRow.paper_id == paper_id)
            ).first()
            if row is not None:
                found = _media_from_row(row, chunk_id)
                if found is not None:
                    return found

        # Fallback: unrecognized id shape, or the targeted report didn't
        # (yet) contain the reference — scan the full library rather than
        # fail outright.
        rows = session.exec(select(ReportRow)).all()
        for row in rows:
            found = _media_from_row(row, chunk_id)
            if found is not None:
                return found
    raise LookupError(f"media not found for chunk_id={chunk_id!r}")


def _safe_data_path(rel_path: str) -> Path:
    """Resolve ``rel_path`` under ``data_dir``, rejecting any escape attempt.

    ``rel_path`` is read back from persisted report JSON
    (``MediaRef.image_path`` / ``Citation.page_image_path``); it is normally
    relative (writers use ``to_relpath``), but the read side must not trust
    that. An absolute path or one containing ``..`` segments would otherwise
    let it join outside ``data_dir`` (or discard ``data_dir`` entirely, per
    ``pathlib`` join semantics for absolute paths) and serve arbitrary host
    files. Raises :class:`LookupError` if the resolved path would escape.
    """
    base = get_config().data_dir.resolve()
    abs_path = (base / rel_path).resolve()
    if not abs_path.is_relative_to(base):
        raise LookupError(f"media path escapes data_dir: {rel_path!r}")
    return abs_path


def _probe_image_size(abs_path: Path) -> tuple[Optional[int], Optional[int]]:
    """Best-effort ``(width, height)`` via Pillow; ``(None, None)`` if unavailable."""
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None, None
    try:
        with Image.open(abs_path) as im:
            return im.size
    except Exception:
        return None, None


class DefaultMediaService(MediaService):
    """Resolves chunk ids against persisted reports and serves file-store bytes."""

    def get_bytes(self, chunk_id: str) -> tuple[bytes, str]:
        _paper_id, info = _find_media(chunk_id)
        rel_path = info.get("image_path") or info.get("thumbnail_path")
        if not rel_path:
            raise LookupError(f"media chunk has no stored image: {chunk_id!r}")
        abs_path = _safe_data_path(rel_path)
        if not abs_path.is_file():
            raise FileNotFoundError(f"media file missing on disk: {abs_path}")
        content_type = mimetypes.guess_type(str(abs_path))[0] or "image/png"
        return abs_path.read_bytes(), content_type

    def get_metadata(self, chunk_id: str) -> MediaMetadata:
        paper_id, info = _find_media(chunk_id)
        rel_path = info.get("image_path") or info.get("thumbnail_path")
        width: Optional[int] = None
        height: Optional[int] = None
        if rel_path:
            try:
                width, height = _probe_image_size(_safe_data_path(rel_path))
            except LookupError:
                width, height = None, None
        content_type = mimetypes.guess_type(rel_path or "")[0] or "image/png"
        return MediaMetadata(
            chunk_id=chunk_id,
            paper_id=paper_id,
            kind=info.get("kind") or "figure",
            label=info.get("label"),
            caption=info.get("caption"),
            provenance=info.get("provenance"),
            page=info.get("page"),
            content_type=content_type,
            width=width,
            height=height,
            url=f"/api/media/{chunk_id}",
        )


class DefaultCitationResolver(CitationResolver):
    """Resolves a citation id by scanning persisted reports.

    ``citation_id`` (see :func:`deepvision.utils.ids.citation_id`) is an
    opaque random id that does not encode its owning ``paper_id`` — unlike
    ``chunk_id`` there is no prefix to split on, and the
    :class:`~deepvision.report.interactive_sections.CitationResolver`
    single-row lookup analogous to :func:`_find_media` isn't possible without
    a schema change to ``ReportRow`` (frozen). As a practical mitigation,
    the ``citation_id`` search is pushed down to SQLite as a substring
    filter on the serialized ``sections`` column so only candidate rows
    are pulled across and JSON-decoded, instead of every persisted report's
    full section tree; the exact id match is still re-verified in Python.
    """

    def resolve(self, citation_id: str) -> Citation:
        with session_scope() as session:
            candidates = session.exec(
                select(ReportRow).where(
                    cast(ReportRow.sections, String).contains(citation_id)
                )
            ).all()
            for row in candidates:
                for sec in row.sections or []:
                    for cit in sec.get("citations", []) or []:
                        if cit.get("id") == citation_id:
                            return Citation.model_validate(cit)
        raise LookupError(f"citation not found: {citation_id!r}")
