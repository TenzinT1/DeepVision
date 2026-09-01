"""Chunk hygiene — keep caption/boilerplate junk out of prose report sections.

Pure, deterministic, no-LLM filtering over retrieved chunks. This module exists
because retrieval over a PDF's raw chunks surfaces exactly the material a human
skimming the paper would skip, and the extractive draft (the code path that
ships when the model is unavailable — see ``agents/base.py::complete_with_fallback``)
turns whatever it is handed straight into body prose with no judgment applied.
Three real failures observed in a generated report motivate every function here:

1. **Duplicated figure captions as an entire section.** A real ``Background``
   section was, verbatim, its entire body::

       Figure 6:The horizontal profile of ROI [1]
       Figure 6:The horizontal profile of ROI [2]
       Figure 8:The final result of the face and eye detection process [3]

   — the same caption chunk retrieved twice, four times over, with nothing
   else. See :func:`is_caption` and :func:`dedupe_similar`.

2. **Running headers glued onto the front of chunk after chunk.** Retrieval
   over a scanned/OCR'd journal PDF pulls in the page header on nearly every
   page, e.g. ``"(IJCSIS) International Journal of Computer Science and
   Information Security, Vol. 7, No. 1, 2010"``, prefixed onto otherwise-good
   paper text. See :func:`strip_repeated_prefixes`.

3. **Author-biography back matter mined as glossary terms.** A real extracted
   "term" was ``"Computer Science" — AUTHORS PROFILE She has completed her
   Bachelor of engineering...`` — page furniture that happens to contain
   capitalized noun phrases, not paper vocabulary. See :func:`is_boilerplate`.

Every filter here is deliberately conservative: a false positive silently
deletes real paper content from a section, which is strictly worse than
letting one stray header line through. When in doubt, keep the chunk.
"""

from __future__ import annotations

import re
from typing import Sequence

from deepvision.models.chunks import AnyChunk
from deepvision.utils.logger import get_logger

__all__ = [
    "is_caption",
    "is_caption_sentence",
    "is_ordinal_number",
    "is_boilerplate",
    "strip_title_block",
    "prose_score",
    "demote_low_prose",
    "diversify",
    "strip_repeated_prefixes",
    "dedupe_similar",
    "narrative_chunks",
]

log = get_logger(__name__)

#: A chunk is "short" (caption-length) below this many characters. A long
#: chunk that merely *starts* with a caption-shaped line is body text that
#: follows the caption, not a caption itself — it must be kept.
_SHORT_CHUNK_CHARS = 300

#: Matches a leading figure/table/chart/plate/scheme caption marker, e.g.
#: "Figure 6:", "Fig. 2 -", "Table 4.", "Chart 1)". Anchored at the start of
#: the (stripped) text; case-insensitive.
_CAPTION_RE = re.compile(
    r"^\s*(fig(?:ure)?|table|chart|plate|scheme)\s*\.?\s*\d+\s*[:.\-—)]?",
    re.IGNORECASE,
)

#: Leading list bullets, markdown bold/italic and quote marks, stripped before
#: the caption match in :func:`is_caption_sentence`. A caption that has been
#: through the summarizer arrives wrapped ("**Figure 5: ...**"), and an
#: unstripped ``**`` defeats ``_CAPTION_RE``'s anchor.
_CAPTION_LEAD_TRIM_RE = re.compile(r'^[\s\-*+•>"“”\'’]*(?:\d+[.)]\s*)?')

# --- Boilerplate sub-patterns, each named after exactly what it catches -----

#: ISSN / DOI / arXiv identifier stamps, e.g. "ISSN 1947-5500", "DOI:
#: 10.1234/abcd", "arXiv:2101.00001".
_ID_STAMP_RE = re.compile(
    r"\b(issn|isbn|doi)\s*[:#]?\s*[\d\w.\-/]+|arxiv\s*[:#]?\s*\d{4}\.\d{4,5}",
    re.IGNORECASE,
)

#: Copyright / rights lines: "(c) 2020 ...", "© 2020 IEEE", "All rights
#: reserved".
#:
#: The year is REQUIRED. An earlier version allowed ``\d{0,4}``, which made a
#: bare ``(c)`` match — and ``(a) (b) (c)`` sub-figure labels are everywhere in
#: papers. That one optional quantifier classified a real multi-panel figure
#: description ("(a) Text prompt: a teddy bear on a bench …") as copyright
#: furniture and deleted it.
_COPYRIGHT_RE = re.compile(
    r"(©|\(c\))\s*\d{4}|all rights reserved", re.IGNORECASE
)

#: A running journal header of the shape "... International Journal of ..
#: Vol. 7, No. 1, 2010" (the exact IJCSIS pattern observed, generalized to any
#: journal name followed by a Vol./No./year triple).
_JOURNAL_HEADER_RE = re.compile(
    r"international journal of [a-z ,&]+,?\s*vol\.?\s*\d+,?\s*no\.?\s*\d+,?\s*\d{4}",
    re.IGNORECASE,
)

#: The literal heading that opens an author-biography block. Decisive on its
#: own at any length — it is a back-matter section header, and everything after
#: it is biography.
_AUTHOR_BIO_HEADER_RE = re.compile(r"authors?\s+profile", re.IGNORECASE)

#: A single biography *phrase*: "received his B.E. in...", "is currently
#: pursuing her Ph.D", "Bachelor of Engineering". One of these alone is not
#: decisive — a paper can legitimately say "models pursuing a Bachelor-level
#: curriculum" — so :func:`is_boilerplate` requires two or more, or a short
#: chunk. Counted with ``findall``, so keep every branch capture-group-free.
_AUTHOR_BIO_PHRASE_RE = re.compile(
    r"received\s+(?:his|her|their|the)\s+[a-z.]*\s*(?:b\.?e\.?|b\.?tech|m\.?e\.?|"
    r"m\.?tech|ph\.?d\.?|bachelor|master|doctorate)|"
    r"is\s+currently\s+pursuing|"
    r"(?:bachelor|master)\s+of\s+(?:engineering|science|technology|arts)",
    re.IGNORECASE,
)

#: Below this many characters of *remaining* text, a chunk that carries a
#: journal running header is the header and nothing else. Above it, the header
#: is merely glued to the front of real paper prose, and dropping the chunk
#: would delete that prose — which is exactly what happened to this sentence:
#: "(IJCSIS) International Journal of Computer Science and Information
#: Security, Vol. 7, No. 1, 2010 we propose the method combine both
#: feature-based and image-based approach to detect the point between the
#: eyes..." — the single most important sentence in that paper.
_MIN_RESIDUE_CHARS = 80

#: A licence / permissions notice. Real text, legally required, and never
#: something to study — but it carries no ©-plus-year and no "all rights
#: reserved", so the copyright rule misses it. Observed leading the source pool
#: of 1706-03762: "Provided proper attribution is provided, Google hereby grants
#: permission to reproduce the tables and figures in this paper..."
_LICENSE_RE = re.compile(
    r"grants?\s+permission|permission\s+to\s+(reproduce|copy|use)|"
    r"licen[sc]ed\s+under|creative\s+commons|"
    r"reproduced\s+with\s+permission|for\s+personal\s+or\s+classroom\s+use",
    re.IGNORECASE,
)

#: An author/affiliation block — the e-mail-and-institution slab under a paper's
#: title. Requires an actual e-mail address, so a sentence that merely mentions a
#: university is untouched. Observed leading the source pool of 2304-06790:
#: "{yutao666, fengruns, ustcfry}@mail.ustc.edu.cn ...".
_EMAIL_RE = re.compile(r"[\w.\-{}, ]+@[\w.\-]+\.[a-z]{2,}", re.IGNORECASE)
_INSTITUTION_RE = re.compile(
    r"\b(university|universität|institute|laborator(y|ies)|department|college|"
    r"academy|research\s+cent(er|re)|inc\.|ltd\.|gmbh)\b",
    re.IGNORECASE,
)

#: A chunk that is nothing but digits and punctuation/whitespace — a bare page
#: number (e.g. "12", "- 3 -", "Page 4").
_BARE_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?[\d\s\-.]+\s*$", re.IGNORECASE)

#: A chunk that is nothing but a URL.
_BARE_URL_RE = re.compile(r"^\s*(https?://|www\.)\S+\s*$", re.IGNORECASE)

#: Leading run of characters used to *detect* a repeated running header/footer.
#: ~90 chars is enough to group the affected chunks without being so long that
#: two merely-similar-topic sentences collide. How much is actually removed is
#: decided separately, by the group's longest common prefix.
_PREFIX_LEN = 90

#: Hard ceiling on how much a "running header" may be. Guards the longest-common-
#: prefix step: three chunks that genuinely open with the same long passage must
#: not have that passage deleted as furniture.
_MAX_HEADER_CHARS = 400

#: Collapse whitespace when normalizing a prefix or a dedupe key.
_WHITESPACE_RE = re.compile(r"\s+")

#: Strip everything but letters/digits for the dedupe key.
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

#: How many chars of the (normalized) chunk start the dedupe key is built from
#: — long enough to distinguish two different captions, short enough that
#: minor trailing variation (a stray "[1]" citation marker) doesn't matter.
_DEDUPE_KEY_CHARS = 120


def is_caption(text: str) -> bool:
    """True if ``text`` is *just* a figure/table caption, not body prose.

    A short chunk whose text opens with a caption marker ("Figure 6:", "Table
    4.", ...) is the caption itself. A long chunk that happens to start the
    same way is body text that follows the caption in the source PDF and must
    be kept as prose.

    >>> is_caption("Figure 6:The horizontal profile of ROI [1]")
    True
    >>> is_caption("Table 4. Ablation results on the validation set.")
    True
    >>> is_caption("We evaluate our approach on three datasets " * 20)
    False
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    if len(stripped) >= _SHORT_CHUNK_CHARS:
        return False
    return bool(_CAPTION_RE.match(stripped))


#: A label word immediately before a number makes that number a pointer
#: ("Figure 3", "Table 2", "Section 4", "Eq. 7"), not a quantity anyone should
#: be asked to recall.
_ORDINAL_LABEL_RE = re.compile(
    r"(?:fig(?:ure)?|table|chart|plate|scheme|section|sec|equation|eq|appendix|"
    r"chapter|algorithm|alg|step|phase|stage|layer|line|row|column|col|"
    r"footnote|ref|reference|page|p|pp|no|nr|item|part|version|v)"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)


def is_ordinal_number(text: str, start: int) -> bool:
    """True if the number at offset ``start`` in ``text`` is a pointer.

    Shared by the flashcard and quiz generators, which both used to blank
    "whichever number comes first" and both shipped the same nonsense as a
    result: "Fill in the blank: Table ____: Variations on the Transformer
    architecture" tests nothing but whether you remember a table's number.

    One home for the rule so the two generators cannot drift apart on it.
    """
    return bool(_ORDINAL_LABEL_RE.search((text or "")[:start]))


def is_caption_sentence(text: str) -> bool:
    """True if a single *sentence* opens as a figure/table caption.

    Deliberately **not** :func:`is_caption` with a different threshold. That
    function judges a whole retrieved chunk and must keep a long chunk that
    merely starts with a caption line, because the rest of it is the body prose
    that followed the caption in the PDF. Here the caller has already split
    text into sentences, so there is no "rest of it": a sentence beginning
    "Figure 5: ..." *is* the caption, however long it runs.

    That distinction is the whole reason this exists. Captions were reaching
    flashcard and quiz generation as study material — real shipped cards asked
    "How does the paper characterize **Figure 5: Many of the attention**?" —
    because the chunk-level filter correctly declined to drop a 300+ char
    caption, and nothing downstream re-checked at sentence granularity.

    Leading list bullets, markdown bold and quote marks are stripped first: a
    caption that has been through a summarizer often arrives as
    ``**Figure 5: Many of the attention** — Figure 5: ...``.

    >>> is_caption_sentence("Figure 1: The Transformer - model architecture.")
    True
    >>> is_caption_sentence("**Table 3: Variations on the Transformer**")
    True
    >>> is_caption_sentence("We evaluate our approach on three datasets.")
    False
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    stripped = _CAPTION_LEAD_TRIM_RE.sub("", stripped, count=1)
    return bool(_CAPTION_RE.match(stripped))


def is_boilerplate(text: str) -> bool:
    """True if ``text`` is page furniture / back matter, never paper prose.

    Catches ISSN/DOI/arXiv stamps, copyright lines, journal running headers,
    author-biography blocks, bare page numbers, and bare URLs. Conservative by
    design: a real sentence that merely mentions a DOI in passing is NOT
    boilerplate (the regexes below match the id-stamp *shape*, not the word),
    so this only fires on chunks that are substantially furniture.

    The load-bearing rule is **furniture must dominate**. A running header glued
    onto the front of a real paragraph makes the chunk *dirty*, not worthless —
    dropping it deletes the paragraph. Use :func:`strip_repeated_prefixes` to
    clean those; this function only rejects a chunk when little else is left.

    >>> is_boilerplate(
    ...     "(IJCSIS) International Journal of Computer Science and "
    ...     "Information Security, Vol. 7, No. 1, 2010"
    ... )
    True
    >>> is_boilerplate(
    ...     "(IJCSIS) International Journal of Computer Science and "
    ...     "Information Security, Vol. 7, No. 1, 2010 we propose the method "
    ...     "combine both feature-based and image-based approach to detect the "
    ...     "point between the eyes by using a Six-Segmented Rectangular filter."
    ... )
    False
    >>> is_boilerplate(
    ...     "AUTHORS PROFILE She has completed her Bachelor of engineering "
    ...     "from XYZ University."
    ... )
    True
    >>> is_boilerplate("ISSN 1947-5500")
    True
    >>> is_boilerplate("(a) Text prompt: a teddy bear on a bench")
    False
    >>> is_boilerplate("We propose a novel method for face detection.")
    False
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    if _BARE_PAGE_NUMBER_RE.match(stripped) and any(c.isdigit() for c in stripped):
        return True
    if _BARE_URL_RE.match(stripped):
        return True
    # "AUTHORS PROFILE" heads a back-matter block: everything after it is
    # biography, so it is decisive at any length.
    if _AUTHOR_BIO_HEADER_RE.search(stripped):
        return True
    # Two or more biography phrases (or one in a short chunk) is a bio; one in a
    # long chunk could be an ordinary sentence.
    _bio_hits = len(_AUTHOR_BIO_PHRASE_RE.findall(stripped))
    if _bio_hits >= 2 or (_bio_hits and len(stripped) < _SHORT_CHUNK_CHARS):
        return True
    # A licence notice is furniture at any length — it is boilerplate in the
    # literal sense, identical across every paper from that publisher.
    if _LICENSE_RE.search(stripped):
        return True
    # An affiliation slab: an e-mail address plus an institution, in something
    # short enough to be the header block rather than a paragraph that happens
    # to give a contact address.
    if (
        len(stripped) < _SHORT_CHUNK_CHARS * 2
        and _EMAIL_RE.search(stripped)
        and _INSTITUTION_RE.search(stripped)
    ):
        return True
    # A journal running header only condemns the chunk if barely anything is
    # left once it is removed.
    _header = _JOURNAL_HEADER_RE.search(stripped)
    if _header:
        residue = _JOURNAL_HEADER_RE.sub(" ", stripped).strip(" \t\n,.;:-—")
        if len(residue) < _MIN_RESIDUE_CHARS:
            return True
    # ID stamps and copyright lines are only decisive on short chunks — a long
    # chunk that happens to cite a DOI mid-paragraph is still real prose.
    if len(stripped) < _SHORT_CHUNK_CHARS:
        if _ID_STAMP_RE.search(stripped):
            return True
        if _COPYRIGHT_RE.search(stripped):
            return True
    return False


def _collapse(text: str) -> str:
    """Whitespace-collapsed, stripped copy of ``text``."""
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def _normalized_prefix(text: str) -> str:
    head = _collapse((text or "")[: _PREFIX_LEN + 20])
    return head[:_PREFIX_LEN].lower()


def _prefix_pattern(raw_head: str) -> re.Pattern[str]:
    """Build a regex that matches ``raw_head`` in the wild, whitespace-loose.

    The fingerprint is built from *whitespace-collapsed* text, so matching it
    back against a chunk's raw text (which may have different line-wrap
    whitespace) needs any run of whitespace in the sample to match any run of
    whitespace in the target, not the exact original spacing.
    """
    words = raw_head.split()
    pattern = r"\s+".join(re.escape(w) for w in words)
    return re.compile(pattern, re.IGNORECASE)


#: The "Abstract" heading that ends a paper's title block. Matched only near the
#: start of a chunk, and only when something follows it.
_ABSTRACT_MARKER_RE = re.compile(
    r"^(?P<front>.{0,600}?)\babstract\b\s*[—–\-:.]?\s+(?=\S)",
    re.IGNORECASE | re.DOTALL,
)

#: A title block is only worth cutting if it actually looks like one — i.e. it
#: carries at least this many author/affiliation signals. Without the check,
#: any sentence using the word "abstract" would lose its own opening.
_TITLE_BLOCK_SIGNAL_RE = re.compile(
    r"\b(dr|prof|mr|ms|mrs)\.|university|college|institute|department|"
    r"@[\w.\-]+\.\w+|\b\d{5,}\b",
    re.IGNORECASE,
)


def strip_title_block(text: str) -> str:
    """Drop a leading title/author/affiliation block that runs up to "Abstract".

    A paper's first-page chunk usually opens with the journal line, the author
    names and their affiliations before reaching ``Abstract—``. Retrieval likes
    that chunk (it contains the abstract), but the front matter is noise that
    then leads a section:

        "(IJCSIS) International Journal of Computer Science and Information
        Security, Ms.S.Sumathi Dr.S.K.Srivatsa Dr.M.Uma Maheswari Bharath
        University St.Joseph College of Engineering ... Chennai,India Abstract—
        A Human Computer Interface (HCI) System for playing games is designed..."

    Only fires when the material before "Abstract" carries an actual
    author/affiliation signal, so a sentence that merely uses the word is
    untouched. Returns ``text`` unchanged when it does not apply.

    >>> strip_title_block(
    ...     "J. Doe, Anna University, Chennai, India Abstract— We propose a "
    ...     "vision-based tracker."
    ... )
    'We propose a vision-based tracker.'
    >>> strip_title_block("The abstract syntax tree is built in one pass.")
    'The abstract syntax tree is built in one pass.'
    """
    match = _ABSTRACT_MARKER_RE.match(text or "")
    if match:
        front = match.group("front")
        if _TITLE_BLOCK_SIGNAL_RE.search(front):
            remainder = (text[match.end():]).strip()
            return remainder or text
    return _strip_leading_contact_block(text)


#: A block that is essentially just contact details: an e-mail address and
#: little else. Papers without an "Abstract" heading in the extracted text still
#: open with one of these.
def _is_contact_block(block: str) -> bool:
    stripped = block.strip()
    if not stripped or len(stripped) > 200:
        return False
    if not _EMAIL_RE.search(stripped):
        return False
    # An e-mail inside a real sentence is contact info *within* prose; a contact
    # block is mostly the address itself.
    without = _EMAIL_RE.sub(" ", stripped).strip(" ,;{}()\n\t")
    return len(without) < 60 or bool(_INSTITUTION_RE.search(without))


def _strip_leading_contact_block(text: str) -> str:
    """Drop leading e-mail/affiliation blocks from an otherwise-good chunk.

    The counterpart to :func:`strip_title_block` for papers whose extracted text
    has no "Abstract" heading to cut at. Observed on 2304-06790, whose best
    first-page chunk is 1,882 characters of genuinely useful text about the
    method — with three lines of author e-mail addresses welded to the front.
    Dropping the chunk would lose the content (the mistake the journal-header
    rule originally made); stripping the front keeps it.

    Only *leading* blocks are removed, and only while they look like contact
    details — the first block of real prose stops the scan.

    >>> _strip_leading_contact_block(
    ...     "{a, b}@mail.example.edu\\n\\nWe propose a new method for inpainting."
    ... )
    'We propose a new method for inpainting.'
    >>> _strip_leading_contact_block("Write to us at a@b.edu for the dataset.")
    'Write to us at a@b.edu for the dataset.'
    """
    blocks = (text or "").split("\n\n")
    index = 0
    while index < len(blocks) and _is_contact_block(blocks[index]):
        index += 1
    if index == 0:
        return text
    remainder = "\n\n".join(blocks[index:]).strip()
    return remainder or text


def _longest_common_prefix(samples: Sequence[str]) -> str:
    """Longest case-insensitive common prefix of ``samples``, cut at a word boundary.

    The fingerprint (:func:`_normalized_prefix`) is a *fixed*-length slice, so
    it finds the group but does not describe the header's real extent. Stripping
    only the fingerprint left a visible tail: the IJCSIS header is ~96 chars, so
    a 90-char fingerprint strip left every affected chunk starting with the
    literal text ``", 2010"``. The group's actual common prefix is the header's
    true length, whatever that happens to be.
    """
    if not samples:
        return ""
    first = samples[0]
    end = min(len(s) for s in samples)
    i = 0
    while i < end and all(s[i].lower() == first[i].lower() for s in samples[1:]):
        i += 1
    head = first[:min(i, _MAX_HEADER_CHARS)]
    # Never cut mid-word: back off to the last whitespace so a header ending in
    # "…No. 1, 2010" is not clipped to "…No. 1, 20".
    if i > len(head) or (i < len(first) and not first[i - 1:i].isspace()):
        cut = head.rfind(" ")
        if cut > 0:
            head = head[:cut]
    return head


def strip_repeated_prefixes(chunks: Sequence[AnyChunk]) -> list[AnyChunk]:
    """Strip a running header/footer that repeats across 3+ distinct pages.

    Normalises the leading ~90 characters of every chunk to *find* the groups.
    When the same normalised prefix recurs on three or more *distinct* pages, it
    is page furniture (a running header/footer), not content that happens to
    start the same way twice. The amount actually removed is then the group's
    **longest common prefix**, not the fingerprint — the fingerprint is a fixed
    slice and would leave a tail of the header behind (see
    :func:`_longest_common_prefix`).

    Chunks are immutable Pydantic models, so a match returns a ``model_copy``
    with the prefix sliced off ``.text``; chunks without a repeated prefix are
    returned unchanged (same object). If stripping would leave a chunk
    (near-)empty, that chunk is dropped entirely.
    """
    prefix_pages: dict[str, set[int]] = {}
    prefix_samples: dict[str, list[str]] = {}
    for chunk in chunks:
        text = chunk.text or ""
        if not text.strip():
            continue
        norm = _normalized_prefix(text)
        if len(norm) < 20:
            # Too short a prefix to be a reliable header fingerprint.
            continue
        prefix_pages.setdefault(norm, set()).add(chunk.page)
        prefix_samples.setdefault(norm, []).append(_collapse(text))

    repeated = {p for p, pages in prefix_pages.items() if len(pages) >= 3}
    if not repeated:
        return list(chunks)

    # One match pattern per group, built from that group's real common prefix.
    patterns: dict[str, re.Pattern[str]] = {}
    for norm in repeated:
        header = _longest_common_prefix(prefix_samples[norm])
        if len(header) < 20:
            continue
        patterns[norm] = _prefix_pattern(header)

    out: list[AnyChunk] = []
    dropped = 0
    stripped_count = 0
    for chunk in chunks:
        text = chunk.text or ""
        norm = _normalized_prefix(text)
        pattern = patterns.get(norm)
        if pattern is None:
            out.append(chunk)
            continue
        match = pattern.match(text.lstrip()) or pattern.search(text[: _MAX_HEADER_CHARS + 40])
        # Trim the punctuation the header's own trailing comma/period leaves
        # behind, so the remainder starts on a real word.
        remainder = (text[match.end():] if match else text).strip(" \t\r\n,.;:-—")
        if len(remainder) < 5:
            dropped += 1
            continue
        stripped_count += 1
        out.append(chunk.model_copy(update={"text": remainder}))

    log.debug(
        "strip_repeated_prefixes: %d distinct repeated prefix(es) found; "
        "stripped from %d chunk(s), dropped %d now-empty chunk(s)",
        len(patterns),
        stripped_count,
        dropped,
    )
    return out


#: Function words that any run of real English prose is dense with, and that
#: OCR noise / UI-screenshot text / axis labels are almost entirely free of.
#: Deliberately small and unambiguous.
_STOPWORDS: frozenset[str] = frozenset(
    """a an the of and or but if then than that this these those to in on at by
    from for with as is are was were be been being it its we our you your they
    their he she his her which who whom when where while can may might must
    should would could has have had do does did not no so such into over under
    between about after before during because""".split()
)

#: Below this share of stopwords a chunk is very unlikely to be prose. Used only
#: to *reorder*, never to delete — see :func:`demote_low_prose` for why.
_MIN_PROSE_SCORE = 0.15

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")


def prose_score(text: str) -> float:
    """Share of ``text``'s tokens that are English function words, in ``[0, 1]``.

    A cheap, language-model-free proxy for "is this readable prose". Real paper
    prose sits around 0.25-0.40 even when it is dense with notation, because
    every sentence still needs *the*, *of*, *is*, *than*. OCR noise off a rotated
    axis label ("Ist une Last tine Maximum Value Coordinate of the nase tp, x
    axis, sau auy Jo suns feans9A") and text scraped out of a UI screenshot
    ("File Help Scale Factor xan fiz.e=] yaxs Refresh Preview Show Eyes") sit
    near zero. Short fragments are unreliable, so anything under 10 tokens
    scores 1.0 — "no opinion" rather than "bad".

    >>> prose_score("We propose a method to detect the point between the eyes.") > 0.3
    True
    >>> prose_score("File Help Scale Factor xan fiz.e=] yaxs Refresh Preview "
    ...             "Show Eyes Show Blink Show Nose Show Eyebrows") < 0.15
    True
    """
    tokens = _TOKEN_RE.findall((text or "").lower())
    if len(tokens) < 10:
        return 1.0
    return sum(1 for t in tokens if t in _STOPWORDS) / len(tokens)


def demote_low_prose(chunks: Sequence[AnyChunk]) -> list[AnyChunk]:
    """Move chunks that don't look like prose to the back, preserving order.

    **Reorders, never deletes.** Sections truncate their chunk list to a
    per-detail-level cap, so pushing OCR noise behind real prose is enough to
    keep it out of the draft when there is enough real material — while a paper
    that has *nothing but* noisy chunks still gets them rather than an empty
    section. Deleting instead would be unsafe: an equation-dense passage
    ("The nose area (Sn) is brighter than the right and left eye area...") scores
    lower than plain prose but is genuine, load-bearing content.
    """
    good = [c for c in chunks if prose_score(c.text or "") >= _MIN_PROSE_SCORE]
    poor = [c for c in chunks if prose_score(c.text or "") < _MIN_PROSE_SCORE]
    if poor:
        log.debug("demote_low_prose: moved %d low-prose chunk(s) to the back", len(poor))
    return good + poor


def _token_set(text: str) -> frozenset[str]:
    """Content-word token set used for the overlap test in :func:`diversify`."""
    return frozenset(
        t for t in _TOKEN_RE.findall((text or "").lower())
        if len(t) > 2 and t not in _STOPWORDS
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


#: Two chunks sharing more than this fraction of their content words are
#: redundant *for the purpose of answering one question* — keeping both spends
#: the context budget saying the same thing twice.
_MAX_OVERLAP = 0.6


def diversify(chunks: Sequence[AnyChunk], limit: int) -> list[AnyChunk]:
    """Pick up to ``limit`` chunks that are relevant *and* say different things.

    Greedy selection down a relevance-ordered list: take the top chunk, then keep
    taking the next one that is not a near-paraphrase of anything already taken
    (Jaccard overlap of content words above :data:`_MAX_OVERLAP`). If that leaves
    fewer than ``limit``, the skipped chunks are appended back in their original
    order — this trims redundancy, it never returns less material than asked for
    when material exists.

    This is a poor-man's MMR: no second embedding pass, no model call, and it
    only needs the text already in hand. It exists because chat answers all read
    the same. Measured on a real paper, "give me the citation in APA style" and
    "what are the limitations of this work?" retrieved **four of the same six
    chunks** — a short paper has few chunks, a vague query separates them
    poorly, and near-identical context can only produce near-identical answers.
    Over-fetching and then diversifying gives each question a context that
    actually differs.
    """
    if limit <= 0:
        return []
    picked: list[AnyChunk] = []
    picked_tokens: list[frozenset[str]] = []
    skipped: list[AnyChunk] = []
    for chunk in chunks:
        if len(picked) >= limit:
            skipped.append(chunk)
            continue
        tokens = _token_set(chunk.text or "")
        if any(_jaccard(tokens, seen) > _MAX_OVERLAP for seen in picked_tokens):
            skipped.append(chunk)
            continue
        picked.append(chunk)
        picked_tokens.append(tokens)
    if len(picked) < limit:
        picked.extend(skipped[: limit - len(picked)])
    return picked


def _dedupe_key(text: str) -> str:
    head = (text or "")[:_DEDUPE_KEY_CHARS].lower()
    head = _WHITESPACE_RE.sub(" ", head)
    head = _NON_ALNUM_RE.sub("", head)
    return head


def dedupe_similar(chunks: Sequence[AnyChunk]) -> list[AnyChunk]:
    """Drop near-duplicate chunks, keeping the first occurrence, order preserved.

    Kills the "same caption retrieved twice in a row" symptom: keys on the
    first ~120 chars of each chunk's text, lowercased with whitespace
    collapsed and non-alphanumerics stripped, so trivial punctuation/citation-
    marker differences don't defeat the match. Also de-dupes by ``chunk.id``
    (stronger than the plain id-only dedupe elsewhere: this also catches two
    *different* chunk ids that happen to carry the same text).
    """
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    out: list[AnyChunk] = []
    dropped = 0
    for chunk in chunks:
        if chunk.id in seen_ids:
            dropped += 1
            continue
        key = _dedupe_key(chunk.text or "")
        if key and key in seen_keys:
            dropped += 1
            continue
        seen_ids.add(chunk.id)
        if key:
            seen_keys.add(key)
        out.append(chunk)
    if dropped:
        log.debug("dedupe_similar: dropped %d duplicate/near-duplicate chunk(s)", dropped)
    return out


def narrative_chunks(
    chunks: Sequence[AnyChunk], *, allow_fallback: bool = True
) -> list[AnyChunk]:
    """The composition callers use to clean chunks bound for a prose section.

    Pipeline: :func:`strip_repeated_prefixes` -> drop :func:`is_caption` /
    :func:`is_boilerplate` -> :func:`dedupe_similar` -> :func:`demote_low_prose`.

    ``allow_fallback`` decides what happens when filtering removes *everything*,
    and the two callers genuinely want opposite things:

    - **Retrieval** (``ResearchAgent.plan_outline``, ``allow_fallback=True``,
      the default) would rather hand the section something than nothing — the
      outline is also what citations and figure grouping are drawn from, so an
      empty group loses more than it saves.
    - **Drafting** (``SummarizerAgent``, ``allow_fallback=False``) must NOT get
      the junk back. Its draft is what ships when the model falls back, and
      handing it a pile of figure captions is precisely the observed bug: a
      ``Why It Matters`` section whose entire body was
      ``"Fig 9 — vision analysis unavailable; showing the extracted crop only."``
      An empty list there is honest and the agent writes "no usable prose was
      retrieved" instead.
    """
    original = list(chunks)
    if not original:
        return original

    stripped = strip_repeated_prefixes(original)
    stripped = [
        c if (t := strip_title_block(c.text or "")) == (c.text or "")
        else c.model_copy(update={"text": t})
        for c in stripped
    ]

    kept: list[AnyChunk] = []
    dropped_caption = 0
    dropped_boilerplate = 0
    for chunk in stripped:
        text = chunk.text or ""
        if is_caption(text):
            dropped_caption += 1
            continue
        if is_boilerplate(text):
            dropped_boilerplate += 1
            continue
        kept.append(chunk)

    deduped = demote_low_prose(dedupe_similar(kept))

    if not deduped:
        if not allow_fallback:
            log.info(
                "narrative_chunks: nothing usable survived filtering (dropped %d "
                "caption, %d boilerplate of %d) and fallback is disabled; "
                "returning empty so the caller can say so honestly",
                dropped_caption,
                dropped_boilerplate,
                len(original),
            )
            return []
        fallback = dedupe_similar(original)
        log.info(
            "narrative_chunks: filtering emptied the section (dropped %d caption, "
            "%d boilerplate of %d); falling back to %d de-duplicated original "
            "chunk(s)",
            dropped_caption,
            dropped_boilerplate,
            len(original),
            len(fallback),
        )
        return fallback

    if dropped_caption or dropped_boilerplate or len(deduped) < len(stripped):
        log.debug(
            "narrative_chunks: kept %d of %d (dropped %d caption, %d boilerplate, "
            "%d duplicate)",
            len(deduped),
            len(original),
            dropped_caption,
            dropped_boilerplate,
            len(stripped) - dropped_caption - dropped_boilerplate - len(deduped),
        )

    return deduped
