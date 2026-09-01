"""Agent base class and shared agent helpers.

Agents are LLM-driven units that consume retrieved chunks and produce a piece of
the report/chat/compare output. They share an :class:`LLMProvider` and operate
over a common :class:`AgentContext`.
This module also hosts small, dependency-light helpers shared by every concrete
agent in this package:

- :func:`complete_with_fallback` — the "draft, then optionally polish" LLM call
  pattern. Every agent first builds a deterministic, grounded draft straight from
  retrieved chunk text, then asks the LLM to rewrite it. Because the draft text
  is always passed as the *last user message*, this degrades perfectly with the
  ``EchoLLM`` default (which echoes the last user message back verbatim): the
  "polished" output is simply the deterministic draft, which is already a
  sensible, grounded, structured result. With a real LLM the same call lets it
  actually rewrite the draft into better prose while (per the system prompt)
  preserving citation markers. Any provider error/timeout falls back to the
  draft, so a broken model backend can never crash the pipeline.
- :func:`safe_json` — lenient JSON extraction with a default fallback.
- :func:`clean_snippet` — whitespace-normalize + sentence-aware truncation used
  for citation snippets and extractive drafts.
- :func:`provenance_list` — de-duplicated, order-preserving provenance list for
  a group of chunks (drives section/media provenance badges).
- :func:`paper_meta_from_row` / :func:`load_paper_meta` — turn a persisted
  ``PaperRow`` (JSON columns re-validated) into a :class:`PaperMeta`, used by the
  synthesis agent and orchestrator to populate ``Report.paper`` / compare
  columns without duplicating DB access logic in every agent.
"""

from __future__ import annotations

import abc
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Generic, Optional, Sequence, TypeVar

from deepvision.models.chunks import AnyChunk, Provenance
from deepvision.models.paper import PaperMeta, PaperStatus
from deepvision.models.settings import AppSettings
from deepvision.providers.base import LLMProvider, Message
from deepvision.utils.logger import get_logger

__all__ = [
    "AgentContext",
    "Agent",
    "complete_with_fallback",
    "llm_fallback_count",
    "reset_llm_fallbacks",
    "strip_preamble",
    "repair_citation_markers",
    "safe_json",
    "clean_snippet",
    "provenance_list",
    "paper_meta_from_row",
    "load_paper_meta",
]

TOut = TypeVar("TOut")

log = get_logger(__name__)

_WS_RE = re.compile(r"\s+")
_JSON_OBJ_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)

#: A leading meta-preamble line some chat models emit before the real content,
#: e.g. "Here is a rewritten version of the draft in markdown format:". Matched
#: only when the first line starts with a giveaway opener AND ends in a colon,
#: so genuine section prose ("The Transformer model consists of…") is untouched.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:sure[,.!:]?\s*)?"
    r"(?:here(?:'s| is| are)?|below is|the following(?: is)?)\b[^\n]*:\s*\n+",
    re.IGNORECASE,
)
#: A single surrounding ```markdown ... ``` code fence the model may wrap output in.
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", re.DOTALL)

#: Appended to every polish prompt so the model returns only the content and does
#: not invent facts absent from the grounded draft.
_OUTPUT_GUARD = (
    "\n\nIMPORTANT: Output only the rewritten text itself in markdown — no "
    "preamble, no meta-commentary, no code fences, and no opener like 'Here is' "
    "or 'Sure'. Do not introduce any fact, number, name, or claim that is not "
    "present in the draft."
)


#: A leaked literal ``[n]`` placeholder followed by the real number, e.g.
#: ``"[n] 3"``. Produced when a model treats the prompt's ``'[n]'`` as text to
#: copy rather than as a pattern to preserve — observed from ``llama3.1:8b`` on
#: every line of a real At a Glance card.
_LEAKED_MARKER_RE = re.compile(r"\[\s*n\s*\]\s*(\d{1,3})\b", re.IGNORECASE)

#: A citation number the model re-styled into bold parentheses, e.g.
#: ``"**(1)**"`` or ``"*(5)*"`` — observed in a real Background section.
_RESTYLED_MARKER_RE = re.compile(r"\*{1,2}\(\s*(\d{1,3})\s*\)\*{1,2}")

#: A bare leftover ``[n]`` with no number attached — nothing to resolve, so it
#: is removed rather than shown to the reader.
_BARE_PLACEHOLDER_RE = re.compile(r"\s*\[\s*n\s*\]\s*", re.IGNORECASE)


def repair_citation_markers(text: str, max_marker: int) -> str:
    """Normalise mangled inline citation markers back to ``[k]``.

    The reader renders **only** ``**bold**``, ``*italic*`` and ``[n]`` markers
    (``pages/report/inlineMarkdown.tsx``), and a marker that does not survive in
    that exact shape silently stops resolving to its citation — the popover just
    never appears. Small local models routinely re-style them: a real
    ``llama3.1:8b`` run produced ``"[n] 3"`` on every At a Glance line and
    ``"**(1)**"`` / ``"**(5) (6)**"`` in Background.

    Deliberately conservative. It repairs only unambiguous damage and never
    invents a marker:

    - ``[n] 3`` -> ``[3]`` (the placeholder leaked, the number is right there)
    - ``**(1)**`` -> ``[1]`` (re-styled, only when bold/italic-wrapped)
    - a bare leftover ``[n]`` -> removed

    A **bare** ``(5)`` is left completely alone: papers number their equations
    that way ("substituting into (5)"), and rewriting those would fabricate
    citations. Numbers above ``max_marker`` (the count of chunks the section was
    written from) are left alone for the same reason — there is nothing for them
    to resolve to.

    >>> repair_citation_markers("The nose is brighter. [n] 3", 5)
    'The nose is brighter. [3]'
    >>> repair_citation_markers("...lighting conditions **(1)**.", 5)
    '...lighting conditions [1].'
    >>> repair_citation_markers("substituting into (5) gives", 5)
    'substituting into (5) gives'
    >>> repair_citation_markers("out of range **(97)**", 5)
    'out of range **(97)**'
    """
    if not text:
        return text

    def _sub(match: "re.Match[str]") -> str:
        number = int(match.group(1))
        if 1 <= number <= max_marker:
            return f"[{number}]"
        return match.group(0)

    out = _LEAKED_MARKER_RE.sub(_sub, text)
    out = _RESTYLED_MARKER_RE.sub(_sub, out)
    out = _BARE_PLACEHOLDER_RE.sub(" ", out)
    return out.strip()


def strip_preamble(text: str) -> str:
    """Remove a leading meta-preamble line and any wrapping code fence."""
    stripped = (text or "").strip()
    fence = _FENCE_RE.match(stripped)
    if fence:
        stripped = fence.group(1).strip()
    without = _PREAMBLE_RE.sub("", stripped, count=1).strip()
    return without or stripped


@dataclass
class AgentContext:
    """Shared inputs available to every agent for one paper."""

    paper_id: str
    settings: AppSettings
    chunks: Sequence[AnyChunk] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class Agent(abc.ABC, Generic[TOut]):
    """Base class for all agents.

    Subclasses declare their output type via the ``TOut`` type parameter and
    implement :meth:`run`.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    @abc.abstractmethod
    def run(self, context: AgentContext) -> TOut:
        """Produce this agent's output from ``context``."""
        raise NotImplementedError

    def _complete(
        self,
        system: str,
        draft: str,
        *,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Convenience wrapper over :func:`complete_with_fallback` using ``self.llm``."""
        return complete_with_fallback(
            self.llm, system, draft, temperature=temperature, max_tokens=max_tokens
        )


#: Per-thread tally of LLM calls that silently fell back to their deterministic
#: draft. Thread-local, not global, because ingestion runs on its own worker
#: thread while API requests run on others — a shared counter would mix them.
_fallbacks = threading.local()


def reset_llm_fallbacks() -> None:
    """Zero this thread's :func:`llm_fallback_count` tally."""
    _fallbacks.n = 0


def llm_fallback_count() -> int:
    """How many LLM calls on this thread degraded to their deterministic draft.

    This is the *normal* failure mode of a local model, not an exotic one: a
    call that exceeds the ``LocalLLM`` timeout raises inside
    :func:`complete_with_fallback`, which swallows it and returns the extractive
    draft. Nothing above ever sees an exception —
    ``AgentOrchestrator._stage`` has nothing to catch, ``build_report`` returns
    a perfectly well-formed ``Report``, and the ingest job's "Generate report"
    stage records ``done``. Without this counter, "half the report is extractive
    because six model calls timed out" is indistinguishable from a clean run,
    which is the same invisibility the tracked report stage exists to end.
    """
    return int(getattr(_fallbacks, "n", 0))


def _note_llm_fallback() -> None:
    _fallbacks.n = int(getattr(_fallbacks, "n", 0)) + 1


def complete_with_fallback(
    llm: LLMProvider,
    system: str,
    draft: str,
    *,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> str:
    """Ask ``llm`` to rewrite/polish ``draft`` (given as the last user message).

    ``draft`` must already be a complete, sensible, grounded answer on its own —
    that is what is returned verbatim when the provider is unavailable, errors,
    or (in the case of ``EchoLLM``) simply echoes the last user message back.
    Never raises: any provider failure is logged and swallowed — but it IS
    counted (see :func:`llm_fallback_count`) so a caller that reports status can
    tell a clean run from a silently degraded one.

    **A fallback is any path that returns the draft instead of model text, not
    just a raised exception.** That distinction was a real bug: the adapters
    themselves swallow transport errors and return an *empty string* rather than
    raising — ``LocalLLM`` logs "Ollama unreachable; returning empty completion"
    and returns ``""``. Because only the ``except`` branch used to increment the
    tally, a run with Ollama simply not started reported **zero** fallbacks while
    every one of its calls had fallen back, so ``StageProgress.degraded`` and
    ``Section.degraded`` both said "model-written" about text extracted verbatim
    from the PDF. That is precisely the failure mode those flags exist to
    surface, so the empty-completion path is counted here too. A timeout also
    counted.
    """
    if not draft or not draft.strip():
        # Nothing to polish, so nothing fell back — do not count this.
        return draft
    try:
        messages = [
            Message(role="system", content=system + _OUTPUT_GUARD),
            Message(role="user", content=draft),
        ]
        out = llm.complete(messages, temperature=temperature, max_tokens=max_tokens)
        out = strip_preamble(out or "")
        if out:
            return out
        _note_llm_fallback()
        log.warning(
            "llm returned an empty completion; using deterministic draft",
            extra={"provider": type(llm).__name__},
        )
        return draft
    except Exception:  # noqa: BLE001 - provider failures must never break agents
        _note_llm_fallback()
        log.warning("llm completion failed; using deterministic draft", exc_info=True)
        return draft


def safe_json(text: str, default: Any) -> Any:
    """Leniently parse JSON out of ``text``, returning ``default`` on any failure."""
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        pass
    match = _JSON_OBJ_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return default


def clean_snippet(text: Optional[str], max_chars: int = 280) -> str:
    """Whitespace-normalize ``text`` and truncate near a sentence/word boundary."""
    cleaned = _WS_RE.sub(" ", (text or "").strip())
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars]
    for punct in (". ", "! ", "? "):
        idx = cut.rfind(punct)
        if idx > max_chars * 0.4:
            return cut[: idx + 1].strip()
    idx = cut.rfind(" ")
    if idx > 0:
        cut = cut[:idx]
    return cut.rstrip(",;:- ") + "…"


def provenance_list(chunks: Sequence[AnyChunk]) -> list[Provenance]:
    """Unique provenance values across ``chunks``, in first-seen order."""
    seen: list[Provenance] = []
    for chunk in chunks:
        prov = chunk.provenance
        if prov not in seen:
            seen.append(prov)
    return seen


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except Exception:
        return None


def _report_exists(paper_id: Optional[str]) -> bool:
    """True iff a whole-paper ``reports`` row exists. Never raises."""
    if not paper_id:
        return False
    try:
        from sqlmodel import select

        from deepvision.db import session_scope
        from deepvision.db.schema import ReportRow

        with session_scope() as session:
            return (
                session.exec(
                    select(ReportRow.paper_id).where(ReportRow.paper_id == paper_id)
                ).first()
                is not None
            )
    except Exception:  # noqa: BLE001 - DB must never crash agent/report building
        return False


def paper_meta_from_row(row: Any, *, has_report: Optional[bool] = None) -> PaperMeta:
    """Re-validate a persisted ``PaperRow`` into a :class:`PaperMeta`.

    ``row`` is duck-typed (any object with the matching attributes) so this
    helper has no hard import-time dependency on the SQLModel row class.

    ``PaperMeta.has_report`` is derived from the ``reports`` table, not from
    ``row``: pass it when the caller already has the answer (it is inside a
    session, or it just persisted the report), otherwise it is looked up here so
    this conversion can never silently emit ``has_report=False`` for a paper
    that does have a report.
    """
    if has_report is None:
        has_report = _report_exists(getattr(row, "id", None))
    try:
        status = PaperStatus(row.status) if row.status else PaperStatus.QUEUED
    except ValueError:
        status = PaperStatus.QUEUED
    return PaperMeta(
        id=row.id,
        arxiv_id=row.arxiv_id,
        arxiv_label=row.arxiv_label or "",
        version=row.version,
        title=row.title or "",
        authors=list(row.authors or []),
        abstract=row.abstract or "",
        categories=list(row.categories or []),
        published=_parse_iso_date(row.published),
        updated=_parse_iso_date(row.updated),
        pdf_url=row.pdf_url,
        abs_url=row.abs_url,
        status=status,
        ingested=bool(row.ingested),
        has_report=bool(has_report),
        thumbnail_path=row.thumbnail_path,
        page_count=row.page_count,
        figure_count=row.figure_count,
        error_message=row.error_message,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def load_paper_meta(paper_id: str) -> Optional[PaperMeta]:
    """Load and convert ``PaperRow(paper_id)`` to :class:`PaperMeta`, or ``None``.

    Never raises: DB errors (e.g. schema not yet initialized) are logged and
    treated as "no metadata available" so report/compare generation can still
    proceed in a degraded form.
    """
    try:
        from sqlmodel import select

        from deepvision.db import session_scope
        from deepvision.db.schema import PaperRow, ReportRow

        with session_scope() as session:
            row = session.get(PaperRow, paper_id)
            if row is None:
                return None
            # Resolved in this same session rather than letting
            # paper_meta_from_row open a nested one.
            has_report = (
                session.exec(
                    select(ReportRow.paper_id).where(ReportRow.paper_id == paper_id)
                ).first()
                is not None
            )
            return paper_meta_from_row(row, has_report=has_report)
    except Exception:  # noqa: BLE001 - DB must never crash agent/report building
        log.warning(
            "failed to load paper metadata", extra={"paper_id": paper_id}, exc_info=True
        )
        return None
