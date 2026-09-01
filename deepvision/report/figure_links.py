"""Figure/table auto-linking — make an in-prose "Figure 6" render as the image.

Deterministic, no LLM. This is the second half of fixing the headline bug: a
generated section's prose legitimately says "as shown in Figure 6", but only
the *Figures* section ever carried :class:`~deepvision.models.report.MediaRef`
cards, so every other section rendered that reference as bare text. This
module scans a section's markdown for figure/table mentions and attaches the
matching :class:`MediaRef` (image, thumbnail, caption) from the paper's full
media pool, so the reader sees the picture inline instead of just the words.

Nothing here calls a model and nothing here can raise out of
:func:`attach_referenced_figures` — a malformed label, an unparseable
reference, or an empty media pool degrades to "attach nothing", never to a
crash, per the project rule that every report stage must ship *something*.
"""

from __future__ import annotations

import re
from typing import Sequence

from deepvision.models.report import MediaRef, Section, SectionName
from deepvision.utils import get_logger

__all__ = ["attach_referenced_figures"]

log = get_logger(__name__)

#: Matches "Figure 6", "Fig. 6", "Fig 6", "Table 4", "Figures 6 and 7",
#: "Fig. 6, 7", "Table 4-5", etc. — case-insensitive. Captures the kind word
#: (group 1) and the run of digits/joiners that follows the first number
#: (group 2), which is post-processed by ``_extract_numbers`` below to pull
#: out every individual figure/table number mentioned in that run (so
#: "Figures 6 and 7" yields both 6 and 7, "Table 4-5" yields both 4 and 5).
#: Deliberately does NOT match a bare "Figure" / "Fig." with no digits at all
#: (e.g. "Figure (p3)" has no number right after the word) — those can never
#: resolve to a specific MediaRef and are skipped, per the docstring below.
_REFERENCE_RE = re.compile(
    r"\b(Fig(?:ure)?s?\.?|Tables?\.?)\s*\.?\s*"
    r"(\d+(?:\s*(?:,|and|&|-|–|—|to)\s*\d+)*)",
    re.IGNORECASE,
)

#: Splits the captured number-run ("6 and 7", "4-5", "6, 7, 8") into ints.
_NUMBER_RE = re.compile(r"\d+")


def _extract_numbers(run: str) -> list[int]:
    """Return every integer mentioned in a matched reference's number-run.

    ``"6"`` -> ``[6]``; ``"6 and 7"`` -> ``[6, 7]``; ``"4-5"`` -> ``[4, 5]``.
    Never raises: a run with no digits (shouldn't happen given the regex,
    but this function is kept defensive) returns ``[]``.
    """
    try:
        return [int(n) for n in _NUMBER_RE.findall(run)]
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return []


def _kind_of(word: str) -> str:
    """Map the matched kind word to the plain ``MediaRef.kind`` string."""
    return "table" if word.strip().lower().startswith("table") else "figure"


def _extract_referenced_numbers(text: str) -> list[tuple[str, int]]:
    """Return ``(kind, number)`` pairs mentioned in ``text``, in reading order.

    ``kind`` is ``"figure"`` or ``"table"``. Duplicates are preserved here
    (order-of-first-mention de-duplication happens in the caller, against the
    already-attached set) so "Figure 6 ... later, Figure 6 again" is handled
    correctly by the caller's de-dup rather than silently collapsed here.
    """
    out: list[tuple[str, int]] = []
    for m in _REFERENCE_RE.finditer(text or ""):
        kind = _kind_of(m.group(1))
        for num in _extract_numbers(m.group(2)):
            out.append((kind, num))
    return out


def _label_number(label: str) -> int | None:
    """Pull the trailing integer out of a free-form ``MediaRef.label``.

    Real labels look like ``"Fig 2"``, ``"Figure 6"``, ``"Table 4"``, or the
    numberless ``"Figure (p3)"`` (a page hint, not a figure number — the ``3``
    there belongs to "p3", not to the figure, so it must NOT be picked up).
    Only the *last* run of digits in the label is treated as the figure/table
    number, and only when it is not immediately preceded by a letter (which
    would make it part of a token like "p3"). Returns ``None`` if no such
    number is found, so a label like ``"Figure (p3)"`` correctly matches
    nothing.
    """
    if not label:
        return None
    matches = list(re.finditer(r"(?<![A-Za-z])(\d+)", label))
    if not matches:
        return None
    try:
        return int(matches[-1].group(1))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _build_index(media: Sequence[MediaRef]) -> dict[tuple[str, int], MediaRef]:
    """Index ``media`` by ``(kind, number)`` parsed from each ref's label.

    Refs whose label carries no parseable number are skipped (never
    indexed), which is what keeps a "Figure N" mention from ever matching a
    numberless ref like "Figure (p3)". First ref wins on a collision.
    """
    index: dict[tuple[str, int], MediaRef] = {}
    for ref in media:
        num = _label_number(ref.label)
        if num is None:
            continue
        key = (ref.kind or "figure", num)
        index.setdefault(key, ref)
    return index


def _referenced_media(
    section: Section, index: dict[tuple[str, int], MediaRef], max_per_section: int
) -> list[MediaRef]:
    """Return the ordered, de-duplicated, capped list of refs mentioned in ``section``.

    Scans ``body_markdown`` then ``deep_dive_markdown`` (in that reading
    order), preserves first-mention order, de-duplicates by ``MediaRef.id``,
    skips anything already attached to the section, and stops once
    ``max_per_section`` new refs have been collected.
    """
    already = {m.id for m in section.media}
    picked: list[MediaRef] = []
    seen_ids: set[str] = set(already)

    text = "\n".join(
        t for t in (section.body_markdown, section.deep_dive_markdown) if t
    )
    for kind, num in _extract_referenced_numbers(text):
        ref = index.get((kind, num))
        if ref is None or ref.id in seen_ids:
            continue
        seen_ids.add(ref.id)
        picked.append(ref)
        if len(picked) >= max_per_section:
            break
    return picked


def attach_referenced_figures(
    sections: Sequence[Section],
    media: Sequence[MediaRef],
    *,
    max_per_section: int = 3,
) -> list[Section]:
    """Attach figures/tables that a section's prose references by number.

    For every section other than :data:`~deepvision.models.report.SectionName.FIGURES`
    (which already carries every figure — re-adding would double them), scans
    ``body_markdown`` and ``deep_dive_markdown`` for references of the form
    ``Figure 6``, ``Fig. 6``, ``Fig 6``, ``Figures 6 and 7``, ``Table 4-5``
    (case-insensitive), resolves each to a :class:`MediaRef` in ``media`` by
    matching kind ("figure" mentions only match figure-kind refs, "table"
    only table-kind) and the trailing number parsed out of that ref's
    free-form ``label``, and appends the matches to ``section.media`` —
    preserving first-mention order, de-duplicated by ``MediaRef.id``, never
    re-adding a ref the section already carries, and capped at
    ``max_per_section`` *newly attached* refs per section.

    **Mutation:** sections are mutated in place (``section.media`` is
    extended via a new list, i.e. reassigned rather than appended-to, so no
    caller holding a reference to the old list is surprised) and the same
    objects are returned in a new list; callers should use the returned list
    but do not need to discard the input.

    Never raises. A section with no references, a media pool with no
    parseable labels, or a reference that matches nothing simply attaches
    nothing for that section — this is a best-effort enrichment, not a
    correctness-critical step.

    Awkward cases, worked through:

    - ``"Figure (p3)"`` — no digit immediately follows the word "Figure" in
      a way the reference regex accepts as a *reference* (it requires a
      digit run right after "Figure"/"Fig."), so this text is simply never
      matched as a reference. Separately, a ``MediaRef`` whose own *label*
      is ``"Figure (p3)"`` has no number attributable to the figure itself
      (the ``3`` belongs to "p3", i.e. is preceded by a letter) so
      :func:`_label_number` returns ``None`` and that ref is never indexed —
      it can never be matched by number from either direction.
    - ``"...as shown in Figure 6, and again in Figure 6 below."`` — matched
      twice, but the second mention is dropped by the ``seen_ids`` de-dup, so
      the section ends up with exactly one copy of Figure 6's ``MediaRef``.
    - ``"Figures 6 and 7 both show..."`` — one regex match, two numbers
      extracted, both looked up and attached (order 6 then 7) subject to the
      cap.
    - ``"Table 4"`` vs a figure labelled ``"Fig 4"`` — kind-scoped lookup
      keeps these separate; ``Table 4`` will never resolve to the figure
      just because the numbers match.
    - A reference to a number with no corresponding media (e.g. the prose
      says "Figure 12" but the paper only has 9 figures) resolves to
      ``index.get(...) is None`` and is silently skipped.
    """
    try:
        index = _build_index(media)
    except Exception:  # pragma: no cover - defensive, see module docstring
        log.warning("figure_links: failed to index media pool", exc_info=True)
        return list(sections)

    out: list[Section] = []
    for section in sections:
        if section.name is SectionName.FIGURES:
            out.append(section)
            continue
        try:
            new_refs = _referenced_media(section, index, max_per_section)
        except Exception:  # pragma: no cover - defensive, see module docstring
            log.warning(
                "figure_links: failed to scan section for references",
                extra={"section": getattr(section, "name", None)},
                exc_info=True,
            )
            new_refs = []
        if new_refs:
            section.media = list(section.media) + new_refs
        out.append(section)
    return out
