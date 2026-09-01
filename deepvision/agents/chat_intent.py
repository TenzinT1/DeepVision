"""Chat intent routing — decide what kind of question was actually asked.

The chat used to send *every* question through the same pipeline: embed it,
retrieve the top-k passages of the paper's body text, hand those to the LLM, ask
it to answer. That is the right machinery for exactly one class of question, and
using it for the others is what made the chat feel useless.

Two measured failures motivated this module:

1. **"give me the citation for the pdf in APA style"** was answered with *"The
   citation for the PDF in APA style is not directly provided within the given
   text. However, based on the [n] markers, it appears that the citations are as
   follows: [1], [2], [3]..."* The paper's body text does not contain its own
   bibliographic record, so no amount of retrieval could ever answer this. The
   answer lives in :class:`~deepvision.models.paper.PaperMeta`, which the agent
   was never given. Faced with an unanswerable question and a pile of irrelevant
   passages, the model invented something.
2. **"What are the limitations of this work?"** and the citation question above
   retrieved **four of the same six chunks**, because a short paper has few
   chunks and a vague query separates them poorly. Near-identical context
   produces near-identical answers — the "it gives the same answer to everything"
   complaint.

So: route first, retrieve second. A question about the paper's *identity* or the
*document* is answered from the database, deterministically and without a model
call, which is both correct and instant. Only a genuine question about the
paper's content earns a retrieval pass.

**The classifier is deliberately deterministic keyword matching, not a model
call.** Three reasons: an extra LLM round-trip before every answer would add
~100 s on the local model this project targets; a misrouted question is worse
than a slow one, and hand-written rules are auditable and testable; and the
categories are narrow enough that keywords genuinely suffice. When nothing
matches, it falls through to :attr:`Intent.CONTENT`, which is the old behaviour —
so the failure mode of this module is "no worse than before", never "wrong".
"""

from __future__ import annotations

import re
from enum import Enum

__all__ = ["Intent", "classify"]


class Intent(str, Enum):
    """What the user is actually asking for.

    Maps onto :class:`~deepvision.models.chat.AnswerKind`, but is an internal
    routing decision rather than a wire value — keep them separate so the
    routing can gain a category without changing the API.
    """

    #: A formatted bibliographic citation ("cite this in MLA", "bibtex please").
    CITATION = "citation"
    #: A fact about the paper's identity (authors, year, arXiv id, title).
    METADATA = "metadata"
    #: A fact about the document as an artifact (page count, figure count).
    STRUCTURE = "structure"
    #: A question about what the paper actually says. The default.
    CONTENT = "content"


#: An explicit "cite THIS paper" request, in any of its usual phrasings. The
#: word "cite" alone is deliberately not here — "which papers does this cite?"
#: is a content question about the reference list.
_CITE_THIS_RE = re.compile(
    r"\bcite\s+(this|it|the\s+(paper|pdf|article|work))\b"
    r"|\bcitation\s+(for|of)\s+(this|the|it)\b"
    r"|\b(how\s+(do|would|should)\s+i\s+cite)\b"
    r"|\breference\s+(list\s+)?entry\b"
    r"|\bbibliograph(y|ic)\s+(entry|record|citation)\b"
    # "give me every citation style" names no style and does not say "cite
    # this", so it fell through to retrieval and got a content answer about the
    # paper's method instead of the five citations it asked for.
    r"|\bcitation\s+styles?\b"
    r"|\b(all|every|each)\s+(the\s+)?(citation|reference)s?\b",
    re.IGNORECASE,
)

#: The name of a citation style.
_STYLE_NAME_RE = re.compile(
    r"\b(apa|mla|chicago|harvard|ieee|bibtex|bib\s?tex)\b", re.IGNORECASE
)

#: Words that make a style name mean "format a citation" rather than being an
#: acronym that happens to collide. A style name on its own is ambiguous:
#: "what is the MLA dataset used for" is a content question about a dataset
#: called MLA, and routing it to the citation formatter — which is what a bare
#: \b(mla)\b rule did — answers a question nobody asked. IEEE collides even
#: harder, being a publisher, a standards body and a style at once.
_CITE_CUE_RE = re.compile(
    r"\b(cite|citation|citations|cited|reference|references|bibliograph\w*|"
    r"format|style|quote)\b",
    re.IGNORECASE,
)

#: A bare style name is enough on its own only when it is essentially the whole
#: message ("bibtex", "apa please") — there is nothing else it could mean.
_BARE_STYLE_MAX_WORDS = 3

#: Explicitly a question about *other* works — must never reach the formatter.
_CITES_OTHERS_RE = re.compile(
    r"\b(which|what|how many|whose)\b[^?]{0,40}\b(cite|cites|cited|references?)\b"
    r"|\bworks?\s+cited\s+by\b"
    r"|\bwho\s+(do|does)\s+(they|it|the\s+authors?)\s+cite\b",
    re.IGNORECASE,
)

#: A fact about the paper's identity, answerable straight from PaperMeta.
_METADATA_RE = re.compile(
    r"\bwho\s+(wrote|are\s+the\s+authors?|is\s+the\s+author)\b"
    r"|\b(author|authors)\s+(of\s+(this|the)\b|names?\b)"
    r"|\bwhen\s+was\s+(it|this|the\s+paper)\s+(published|written|released)\b"
    r"|\b(publication|published)\s+(date|year)\b"
    r"|\bwhat\s+year\b"
    r"|\bwhat\s+is\s+(the\s+)?(title|arxiv\s*(id|number))\b"
    r"|\barxiv\s*(id|number|link|url)\b"
    r"|\bwhat\s+(is|are)\s+(the\s+)?(categor|subject)",
    re.IGNORECASE,
)

#: A fact about the document as an artifact, answerable from the database.
_STRUCTURE_RE = re.compile(
    r"\bhow\s+(many|long)\b[^?]{0,30}\b(page|pages|figure|figures|table|tables|"
    r"chapter|chapters|section|sections|word|words)\b"
    r"|\b(page|figure|table|chapter)\s+count\b"
    r"|\bhow\s+long\s+is\s+(this|it|the\s+(paper|pdf|document))\b"
    r"|\bwhat\s+(chapters|sections)\s+(are|does)\b",
    re.IGNORECASE,
)


def classify(question: str) -> Intent:
    """Return the :class:`Intent` of ``question``.

    Order matters, and the first rule is the one that stops a confident wrong
    answer: a question about *whom this paper cites* is a content question about
    its reference list, and must never be routed to the citation formatter,
    which would answer the entirely different question "how do I cite this
    paper". That check runs before everything else.

    >>> classify("give me the citation for the pdf in APA style")
    <Intent.CITATION: 'citation'>
    >>> classify("cite this in bibtex")
    <Intent.CITATION: 'citation'>
    >>> classify("which papers does this cite?")
    <Intent.CONTENT: 'content'>
    >>> classify("what is the MLA dataset used for")
    <Intent.CONTENT: 'content'>
    >>> classify("bibtex")
    <Intent.CITATION: 'citation'>
    >>> classify("who wrote this?")
    <Intent.METADATA: 'metadata'>
    >>> classify("how many figures are there?")
    <Intent.STRUCTURE: 'structure'>
    >>> classify("What are the limitations of this work?")
    <Intent.CONTENT: 'content'>
    >>> classify("explain the SSR filter")
    <Intent.CONTENT: 'content'>
    """
    text = (question or "").strip()
    if not text:
        return Intent.CONTENT

    # A question about the paper's own reference list is CONTENT, always.
    if _CITES_OTHERS_RE.search(text):
        return Intent.CONTENT
    if _CITE_THIS_RE.search(text):
        return Intent.CITATION
    if _STYLE_NAME_RE.search(text) and (
        _CITE_CUE_RE.search(text) or len(text.split()) <= _BARE_STYLE_MAX_WORDS
    ):
        return Intent.CITATION
    if _STRUCTURE_RE.search(text):
        return Intent.STRUCTURE
    if _METADATA_RE.search(text):
        return Intent.METADATA
    return Intent.CONTENT
