"""Deterministic bibliographic citation formatting — no LLM, ever.

Why this is code and not a prompt: asked "give me the citation for the pdf in
APA style", the chat used to hand the question to the language model along
with a handful of retrieved *body-text* chunks — never the paper's own
:class:`~deepvision.models.paper.PaperMeta` — and got back "The citation ..
is not directly provided within the given text. However, based on the [n]
markers, it appears that the citations are as follows: [1], [2], [3]...".
That is not a prompting bug to iterate on: the answer was structurally
unavailable to the model (it never saw the authors, year, or arXiv id), so it
fabricated something plausible-looking instead. Citation formatting is also a
*solved* problem with exact, mechanical rules per style — an LLM gets the
punctuation, the ampersand placement, or the 20-author truncation subtly
wrong in ways a casual reader will not notice until a reviewer does. Every
function below is a pure function over :class:`PaperMeta`: same input, same
string, every time, with no network or model call in the path.

The hard part is not the templates, it is turning a free-form author string
("S. Sumathi", "Sumathi S.", "Ms.S.Sumathi", "M. Uma Maheswari") into
(given, family) — see :func:`_split_name`.

Degrade honestly, never fabricate: a missing year becomes the style's
"no date" convention, missing authors means the entry leads with the title,
and a missing title is spelled out rather than rendered as an empty pair of
quotes. An uploaded PDF's ``arxiv_id`` is a placeholder
(``"upload:<paper_id>"``) and never printed or linked as if it were real.
``PaperMeta`` has no DOI or venue field, so none is ever invented here.
"""

from __future__ import annotations

import re
from enum import Enum

from deepvision.models.paper import PaperMeta

__all__ = [
    "CitationStyle",
    "CITATION_STYLE_LABELS",
    "format_citation",
    "format_all_citations",
    "detect_styles",
]


class CitationStyle(str, Enum):
    """Supported citation styles. Wire-facing string values — never rename."""

    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    BIBTEX = "bibtex"


CITATION_STYLE_LABELS: dict[CitationStyle, str] = {
    CitationStyle.APA: "APA (7th edition)",
    CitationStyle.MLA: "MLA (9th edition)",
    CitationStyle.CHICAGO: "Chicago (author-date)",
    CitationStyle.IEEE: "IEEE",
    CitationStyle.BIBTEX: "BibTeX",
}

#: Placeholder used when a title is missing. Never render an empty "" pair.
_TITLE_MISSING = "[title unavailable]"

#: Honorific prefixes to strip before name-splitting ("Ms.S.Sumathi" -> "S.Sumathi").
_HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "prof", "professor"}

#: A single letter, optionally followed by a period: "S", "S.", "K." — an initial.
_INITIAL_RE = re.compile(r"^[A-Za-z]\.?$")


# ---------------------------------------------------------------------------
# Name parsing — the actually-hard part
# ---------------------------------------------------------------------------


def _normalize_token_spacing(name: str) -> str:
    """Insert a space after any period glued to the next character.

    "Ms.S.Sumathi" -> "Ms. S. Sumathi" so plain whitespace-splitting works.
    """
    return re.sub(r"\.(?=\S)", ". ", name.strip())


def _normalize_given(tokens: list[str]) -> str:
    """Join given-name tokens, turning a bare initial letter into "X."."""
    out = []
    for t in tokens:
        if _INITIAL_RE.match(t) and not t.endswith("."):
            out.append(t + ".")
        else:
            out.append(t)
    return " ".join(out)


def _split_name(name: str) -> tuple[str, str]:
    """Split a free-form author name into ``(given, family)``.

    Handles the messy real-world shapes an arXiv author list actually
    contains: initials before or after the surname, honorifics glued on
    with no spaces, comma-separated "Surname, Given" pairs already in
    citation form, and bare mononyms. Never invents a full first name from
    an initial — if the source only gave "S.", the result only has "S.".

    Heuristic: strip any leading honorific, then classify each remaining
    token as an "initial" (a single letter, optionally with a period) or a
    "word". If exactly one token is a word, that token is the family name
    and every initial (wherever it sits) is given-name material — this
    covers both "S. Sumathi" and the reversed "Sumathi S." the same way.
    Otherwise (multiple word tokens, e.g. a spelled-out middle name) the
    last token is taken as the family name per the standard
    First [Middle] Last convention.

    >>> _split_name("S. Sumathi")
    ('S.', 'Sumathi')
    >>> _split_name("Sumathi S.")
    ('S.', 'Sumathi')
    >>> _split_name("Ms.S.Sumathi")
    ('S.', 'Sumathi')
    >>> _split_name("S. K. Srivatsa")
    ('S. K.', 'Srivatsa')
    >>> _split_name("M. Uma Maheswari")
    ('M. Uma', 'Maheswari')
    >>> _split_name("Sumathi, S.")
    ('S.', 'Sumathi')
    >>> _split_name("Plato")
    ('', 'Plato')
    >>> _split_name("")
    ('', '')
    """
    name = name.strip()
    if not name:
        return ("", "")

    if "," in name:
        family_part, _, given_part = name.partition(",")
        given_tokens = _normalize_token_spacing(given_part).split()
        return (_normalize_given(given_tokens), family_part.strip())

    tokens = _normalize_token_spacing(name).split()
    if tokens and tokens[0].rstrip(".").lower() in _HONORIFICS:
        tokens = tokens[1:]
    if not tokens:
        return ("", "")
    if len(tokens) == 1:
        return ("", tokens[0])

    word_idx = [i for i, t in enumerate(tokens) if not _INITIAL_RE.match(t)]
    if len(word_idx) == 1:
        fam_idx = word_idx[0]
        family = tokens[fam_idx]
        given_tokens = tokens[:fam_idx] + tokens[fam_idx + 1 :]
    else:
        family = tokens[-1]
        given_tokens = tokens[:-1]
    return (_normalize_given(given_tokens), family)


def _initials(given: str) -> str:
    """Reduce a given-name string to initials: "M. Uma" -> "M. U.".

    Used by styles (APA, IEEE) that always abbreviate given names,
    regardless of whether the source already gave a full word.
    """
    if not given:
        return ""
    return " ".join(p[0].upper() + "." for p in given.split() if p)


# ---------------------------------------------------------------------------
# Shared field helpers — honest degrade, never fabricate
# ---------------------------------------------------------------------------


def _is_upload(meta: PaperMeta) -> bool:
    """True for a user-uploaded PDF, whose arxiv_id/arxiv_label are placeholders."""
    return meta.arxiv_id.startswith("upload:") or meta.arxiv_label == "Uploaded PDF"


def _year(meta: PaperMeta) -> str:
    return str(meta.published.year) if meta.published else "n.d."


def _stop(value: str) -> str:
    """Append a sentence period unless ``value`` already ends in one.

    The no-date marker "n.d." carries its own trailing period, so every
    ``f"...{year}."`` produced "n.d.." on a paper with no publication date —
    which is every uploaded PDF.
    """
    value = (value or "").rstrip()
    return value if value.endswith(".") else f"{value}."


def _title_field(meta: PaperMeta, *, quote: bool, terminator: str = ".") -> str:
    """Title formatted for a citation, closed with ``terminator``.

    IEEE always closes a quoted title with a comma (more fields follow on
    the same sentence); every other style here closes it with a period.
    A missing title becomes an explicit placeholder rather than an empty
    quoted pair (``""``), per the "never fabricate, degrade honestly" rule.
    """
    if meta.title and meta.title.strip():
        t = meta.title.strip()
        return f'"{t}{terminator}"' if quote else f"{t}{terminator}"
    return f"{_TITLE_MISSING}{terminator}"


def _bibtex_title(meta: PaperMeta) -> str:
    return meta.title.strip() if meta.title and meta.title.strip() else _TITLE_MISSING


def _arxiv_url(meta: PaperMeta) -> str | None:
    """Real arXiv abstract URL, or None for an upload (never a fake arxiv.org link)."""
    if _is_upload(meta):
        return None
    if meta.abs_url:
        return meta.abs_url
    return f"https://arxiv.org/abs/{meta.arxiv_id}" if meta.arxiv_id else None


def _arxiv_number(meta: PaperMeta) -> str | None:
    return None if _is_upload(meta) else (meta.arxiv_id or None)


# ---------------------------------------------------------------------------
# APA 7
# ---------------------------------------------------------------------------


def _apa_authors(authors: list[str]) -> str:
    if not authors:
        return ""
    formatted = []
    for a in authors:
        given, family = _split_name(a)
        initials = _initials(given)
        formatted.append(f"{family}, {initials}" if initials else family)
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) <= 20:
        return ", ".join(formatted[:-1]) + ", & " + formatted[-1]
    # APA 7 rule for 21+ authors: first 19, an ellipsis, then the final author.
    head = ", ".join(formatted[:19])
    return f"{head}, ... {formatted[-1]}"


def _format_apa(meta: PaperMeta) -> str:
    """
    >>> from datetime import date
    >>> meta = PaperMeta(
    ...     id="p1", arxiv_id="1002.2191",
    ...     title="Vision Based Game Development Using Human Computer Interaction",
    ...     authors=["S. Sumathi", "S. K. Srivatsa", "M. Uma Maheswari"],
    ...     published=date(2010, 2, 10),
    ... )
    >>> format_citation(meta, CitationStyle.APA)
    'Sumathi, S., Srivatsa, S. K., & Maheswari, M. U. (2010). Vision Based Game Development Using Human Computer Interaction. arXiv. https://arxiv.org/abs/1002.2191'
    """
    authors = _apa_authors(meta.authors)
    year = _year(meta)
    title = _title_field(meta, quote=False)
    lead = f"{authors} ({year}). {title}" if authors else f"{title} ({year})."
    if _is_upload(meta):
        return lead
    url = _arxiv_url(meta)
    return f"{lead} arXiv. {url}" if url else lead


# ---------------------------------------------------------------------------
# MLA 9
# ---------------------------------------------------------------------------


def _mla_authors(authors: list[str]) -> str:
    if not authors:
        return ""
    g1, f1 = _split_name(authors[0])
    first = f"{f1}, {g1}" if g1 else f1
    if len(authors) == 1:
        return first
    if len(authors) == 2:
        g2, f2 = _split_name(authors[1])
        second = f"{g2} {f2}".strip()
        return f"{first}, and {second}"
    return f"{first}, et al."


def _format_mla(meta: PaperMeta) -> str:
    authors = _mla_authors(meta.authors).rstrip(".")
    year = _year(meta)
    title = _title_field(meta, quote=True)
    lead = f"{authors}. {title}" if authors else title
    if _is_upload(meta):
        return f"{lead} {year}." if year != "n.d." else lead
    url = _arxiv_url(meta)
    tail = "arXiv" if year == "n.d." else f"arXiv, {year}"
    if url:
        tail = f"{tail}, {url}"
    return f"{lead} {tail}."


# ---------------------------------------------------------------------------
# Chicago (author-date)
# ---------------------------------------------------------------------------


def _chicago_authors(authors: list[str]) -> str:
    if not authors:
        return ""
    parts = []
    for i, a in enumerate(authors):
        given, family = _split_name(a)
        if i == 0:
            parts.append(f"{family}, {given}" if given else family)
        else:
            parts.append(f"{given} {family}".strip())
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _format_chicago(meta: PaperMeta) -> str:
    authors = _chicago_authors(meta.authors).rstrip(".")
    year = _year(meta)
    title = _title_field(meta, quote=True)
    lead = (f"{authors}. {_stop(year)} {title}" if authors
            else f"{title} {_stop(year)}")
    if _is_upload(meta):
        return lead
    url = _arxiv_url(meta)
    return f"{lead} arXiv. {url}" if url else lead


# ---------------------------------------------------------------------------
# IEEE
# ---------------------------------------------------------------------------


def _ieee_authors(authors: list[str]) -> str:
    if not authors:
        return ""
    parts = []
    for a in authors:
        given, family = _split_name(a)
        initials = _initials(given)
        parts.append(f"{initials} {family}".strip())
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _format_ieee(meta: PaperMeta) -> str:
    """
    >>> from datetime import date
    >>> meta = PaperMeta(
    ...     id="p1", arxiv_id="1002.2191",
    ...     title="Vision Based Game Development Using Human Computer Interaction",
    ...     authors=["S. Sumathi", "S. K. Srivatsa", "M. Uma Maheswari"],
    ...     published=date(2010, 2, 10),
    ... )
    >>> format_citation(meta, CitationStyle.IEEE)
    'S. Sumathi, S. K. Srivatsa, and M. U. Maheswari, "Vision Based Game Development Using Human Computer Interaction," arXiv preprint arXiv:1002.2191, 2010.'
    """
    authors = _ieee_authors(meta.authors)
    title = _title_field(meta, quote=True, terminator=",")
    year = _year(meta)
    lead = f"{authors}, {title}" if authors else title
    if _is_upload(meta):
        return f"{lead} {_stop(year)}"
    num = _arxiv_number(meta)
    if num:
        return f"{lead} arXiv preprint arXiv:{num}, {_stop(year)}"
    return f"{lead} {_stop(year)}"


# ---------------------------------------------------------------------------
# BibTeX
# ---------------------------------------------------------------------------

_TITLE_STOPWORDS = {"a", "an", "the", "of", "on", "in", "for", "and", "to"}


def _bibtex_authors(authors: list[str]) -> str:
    parts = []
    for a in authors:
        given, family = _split_name(a)
        parts.append(f"{family}, {given}" if given else family)
    return " and ".join(parts)


def _cite_key(meta: PaperMeta) -> str:
    """Generate a BibTeX cite key like "sumathi2010vision".

    family-name + year + first non-stopword title word, all lowercased and
    stripped to letters — never fabricated, purely derived from what the
    metadata already has (falls back to "unknown"/"nd" when absent).
    """
    family = "unknown"
    if meta.authors:
        _, family = _split_name(meta.authors[0])
        family = re.sub(r"[^a-z]", "", family.lower()) or "unknown"
    year = str(meta.published.year) if meta.published else ""
    words = re.findall(r"[A-Za-z]+", meta.title or "")
    word = next((w.lower() for w in words if w.lower() not in _TITLE_STOPWORDS), "")
    return f"{family}{year}{word}" if year else f"{family}_{word or 'untitled'}"


def _format_bibtex(meta: PaperMeta) -> str:
    """
    >>> from datetime import date
    >>> meta = PaperMeta(
    ...     id="p1", arxiv_id="1002.2191",
    ...     title="Vision Based Game Development Using Human Computer Interaction",
    ...     authors=["S. Sumathi", "S. K. Srivatsa", "M. Uma Maheswari"],
    ...     published=date(2010, 2, 10),
    ... )
    >>> print(format_citation(meta, CitationStyle.BIBTEX))
    @misc{sumathi2010vision,
      author = {Sumathi, S. and Srivatsa, S. K. and Maheswari, M. Uma},
      title = {Vision Based Game Development Using Human Computer Interaction},
      year = {2010},
      eprint = {1002.2191},
      archivePrefix = {arXiv},
      url = {https://arxiv.org/abs/1002.2191}
    }
    """
    key = _cite_key(meta)
    authors = _bibtex_authors(meta.authors)
    title = _bibtex_title(meta)
    # BibTeX has no "n.d." convention -- `year = {n.d.}` is not a value any
    # bibliography tool can consume, so an unknown year omits the field and
    # lets the style decide how to render the absence.
    year = str(meta.published.year) if meta.published else ""

    fields = []
    if authors:
        fields.append(f"  author = {{{authors}}},")
    fields.append(f"  title = {{{title}}},")
    if year:
        fields.append(f"  year = {{{year}}},")
    if _is_upload(meta):
        fields.append("  note = {Uploaded PDF; no arXiv identifier},")
    else:
        num = _arxiv_number(meta)
        if num:
            fields.append(f"  eprint = {{{num}}},")
        fields.append("  archivePrefix = {arXiv},")
        url = _arxiv_url(meta)
        if url:
            fields.append(f"  url = {{{url}}},")
    fields[-1] = fields[-1].rstrip(",")
    body = "\n".join(fields)
    return f"@misc{{{key},\n{body}\n}}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_FORMATTERS = {
    CitationStyle.APA: _format_apa,
    CitationStyle.MLA: _format_mla,
    CitationStyle.CHICAGO: _format_chicago,
    CitationStyle.IEEE: _format_ieee,
    CitationStyle.BIBTEX: _format_bibtex,
}


def format_citation(meta: PaperMeta, style: CitationStyle) -> str:
    """Format ``meta`` as a citation string in ``style``. Pure, no I/O."""
    return _FORMATTERS[style](meta).strip()


def format_all_citations(meta: PaperMeta) -> dict[CitationStyle, str]:
    """Format ``meta`` in every supported style at once."""
    return {style: format_citation(meta, style) for style in CitationStyle}


_STYLE_PATTERNS: dict[CitationStyle, re.Pattern[str]] = {
    CitationStyle.APA: re.compile(r"\bapa\b", re.IGNORECASE),
    CitationStyle.MLA: re.compile(r"\bmla\b", re.IGNORECASE),
    CitationStyle.CHICAGO: re.compile(r"\bchicago\b", re.IGNORECASE),
    CitationStyle.IEEE: re.compile(r"\bieee\b", re.IGNORECASE),
    CitationStyle.BIBTEX: re.compile(r"\bbib\s*tex\b|\blatex\b", re.IGNORECASE),
}


def detect_styles(question: str) -> list[CitationStyle]:
    """Which citation styles a question names, in canonical order, or ``[]``.

    Plain keyword matching with a word-boundary requirement so a style name
    never fires as a substring of an unrelated word. A question can name more
    than one style ("give me APA and MLA") and both are returned.
    """
    if not question:
        return []
    return [s for s in CitationStyle if _STYLE_PATTERNS[s].search(question)]
