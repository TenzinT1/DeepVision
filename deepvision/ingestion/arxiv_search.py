"""arXiv search + metadata fetch.

Backs ``GET /search`` and the metadata half of ``POST /ingest``.

"""

from __future__ import annotations

import abc
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from typing import Optional

from deepvision.api.schemas import SearchDateRange, SearchResultItem, SearchSort
from deepvision.models.paper import PaperMeta, PaperStatus
from deepvision.utils import get_logger
from deepvision.utils.ids import paper_id_from_arxiv

__all__ = ["ArxivSearcher", "ArxivSearcherService"]

log = get_logger(__name__)


class ArxivSearcher(abc.ABC):
    """Queries the arXiv API and resolves single-paper metadata."""

    @abc.abstractmethod
    def search(
        self,
        query: str,
        *,
        category: Optional[str] = None,
        date_range: SearchDateRange = SearchDateRange.ANY,
        sort: SearchSort = SearchSort.RELEVANCE,
        limit: int = 25,
    ) -> list[SearchResultItem]:
        """Return search hits (with library/ingested flags filled by the caller)."""
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_meta(self, arxiv_id: str) -> PaperMeta:
        """Fetch full :class:`PaperMeta` for a single arXiv id."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# Concrete implementation
# --------------------------------------------------------------------------

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_EXPORT_API = "http://export.arxiv.org/api/query"
_USER_AGENT = "DeepVision/0.1 (mailto:research-assistant@example.com)"


def _date_range_bounds(date_range: SearchDateRange) -> Optional[tuple[date, date]]:
    """Return an inclusive ``(start, end)`` date window, or ``None`` for ANY."""
    today = date.today()
    if date_range == SearchDateRange.PAST_MONTH:
        return today - timedelta(days=30), today
    if date_range == SearchDateRange.PAST_6_MONTHS:
        return today - timedelta(days=182), today
    if date_range == SearchDateRange.PAST_YEAR:
        return today - timedelta(days=365), today
    if date_range == SearchDateRange.RANGE_2024_2025:
        return date(2024, 1, 1), date(2025, 12, 31)
    return None


def _build_search_query(
    query: str, *, category: Optional[str], date_range: SearchDateRange
) -> str:
    clauses = [f"all:{query}"] if query else []
    if category:
        clauses.append(f"cat:{category}")
    bounds = _date_range_bounds(date_range)
    if bounds:
        start, end = bounds
        clauses.append(
            f"submittedDate:[{start.strftime('%Y%m%d')}0000 TO "
            f"{end.strftime('%Y%m%d')}2359]"
        )
    return " AND ".join(f"({c})" for c in clauses) if clauses else "all:*"


def _sort_params(sort: SearchSort) -> tuple[str, str]:
    """Map :class:`SearchSort` to arXiv ``sortBy``/``sortOrder`` params.

    ``most_cited`` is dropped per product decision (arXiv exposes no citation
    counts) and treated as ``newest``.
    """
    if sort == SearchSort.RELEVANCE:
        return "relevance", "descending"
    return "submittedDate", "descending"


def _parse_atom_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _split_id_version(raw_id: str) -> tuple[str, Optional[str]]:
    """Split ``2411.10842v2`` (or a full ``.../abs/...`` URL) into id + version."""
    tail = raw_id.rstrip("/").rsplit("/", 1)[-1]
    match = re.match(r"^(?P<id>.+?)(?P<version>v\d+)?$", tail)
    if not match:
        return tail, None
    return match.group("id"), match.group("version")


def _entry_to_meta(entry: ET.Element) -> PaperMeta:
    raw_id = (entry.findtext(f"{_ATOM_NS}id") or "").strip()
    bare_id, version = _split_id_version(raw_id)
    title = (entry.findtext(f"{_ATOM_NS}title") or "").strip()
    title = re.sub(r"\s+", " ", title)
    summary = (entry.findtext(f"{_ATOM_NS}summary") or "").strip()
    summary = re.sub(r"\s+", " ", summary)
    authors = [
        (a.findtext(f"{_ATOM_NS}name") or "").strip()
        for a in entry.findall(f"{_ATOM_NS}author")
    ]
    authors = [a for a in authors if a]
    categories = [
        c.attrib.get("term", "")
        for c in entry.findall(f"{_ATOM_NS}category")
        if c.attrib.get("term")
    ]
    published = _parse_atom_date(entry.findtext(f"{_ATOM_NS}published"))
    updated = _parse_atom_date(entry.findtext(f"{_ATOM_NS}updated"))

    pdf_url: Optional[str] = None
    abs_url: Optional[str] = None
    for link in entry.findall(f"{_ATOM_NS}link"):
        rel = link.attrib.get("rel")
        href = link.attrib.get("href")
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == (
            "application/pdf"
        ):
            pdf_url = href
        elif rel == "alternate":
            abs_url = href
    if abs_url is None and raw_id:
        abs_url = raw_id
    if pdf_url is None and bare_id:
        pdf_url = f"https://arxiv.org/pdf/{bare_id}"

    paper_id = paper_id_from_arxiv(bare_id)
    return PaperMeta(
        id=paper_id,
        arxiv_id=bare_id,
        arxiv_label=f"arXiv:{bare_id}",
        version=version,
        title=title,
        authors=authors,
        abstract=summary,
        categories=categories,
        published=published,
        updated=updated,
        pdf_url=pdf_url,
        abs_url=abs_url,
        status=PaperStatus.QUEUED,
        ingested=False,
    )


def _fetch_atom(params: dict[str, str]) -> ET.Element:
    url = f"{_EXPORT_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 (arXiv is trusted)
        payload = resp.read()
    return ET.fromstring(payload)


class ArxivSearcherService(ArxivSearcher):
    """arXiv Atom-API client.

    Uses the ``arxiv`` PyPI package when available; otherwise falls back to a
    dependency-free client against the public Atom export API
    (``export.arxiv.org/api/query``) using only the standard library, so the
    module still functions when the optional package isn't installed.
    """

    def search(
        self,
        query: str,
        *,
        category: Optional[str] = None,
        date_range: SearchDateRange = SearchDateRange.ANY,
        sort: SearchSort = SearchSort.RELEVANCE,
        limit: int = 25,
    ) -> list[SearchResultItem]:
        try:
            return self._search_via_package(
                query, category=category, date_range=date_range, sort=sort, limit=limit
            )
        except ImportError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("arxiv package search failed, falling back", extra={"error": str(exc)})

        sort_by, sort_order = _sort_params(sort)
        params = {
            "search_query": _build_search_query(
                query, category=category, date_range=date_range
            ),
            "start": "0",
            "max_results": str(limit),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        try:
            root = _fetch_atom(params)
        except (urllib.error.URLError, ET.ParseError, TimeoutError) as exc:
            log.error("arxiv search request failed", extra={"error": str(exc)})
            return []

        results: list[SearchResultItem] = []
        for entry in root.findall(f"{_ATOM_NS}entry"):
            try:
                meta = _entry_to_meta(entry)
            except Exception as exc:  # pragma: no cover - malformed entry
                log.warning("skipping malformed arxiv entry", extra={"error": str(exc)})
                continue
            results.append(SearchResultItem(paper=meta, ingested=False, in_library=False))
        return results

    def fetch_meta(self, arxiv_id: str) -> PaperMeta:
        try:
            return self._fetch_meta_via_package(arxiv_id)
        except ImportError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "arxiv package fetch_meta failed, falling back", extra={"error": str(exc)}
            )

        params = {"id_list": arxiv_id, "max_results": "1"}
        root = _fetch_atom(params)
        entry = root.find(f"{_ATOM_NS}entry")
        if entry is None:
            raise ValueError(f"arXiv id not found: {arxiv_id}")
        return _entry_to_meta(entry)

    # -- optional accelerated path via the `arxiv` package -----------------

    def _search_via_package(
        self,
        query: str,
        *,
        category: Optional[str],
        date_range: SearchDateRange,
        sort: SearchSort,
        limit: int,
    ) -> list[SearchResultItem]:
        import arxiv  # type: ignore

        sort_by, _ = _sort_params(sort)
        criterion = (
            arxiv.SortCriterion.Relevance
            if sort_by == "relevance"
            else arxiv.SortCriterion.SubmittedDate
        )
        search = arxiv.Search(
            query=_build_search_query(query, category=category, date_range=date_range),
            max_results=limit,
            sort_by=criterion,
        )
        client = arxiv.Client()
        results: list[SearchResultItem] = []
        for result in client.results(search):
            bare_id, version = _split_id_version(result.get_short_id())
            paper_id = paper_id_from_arxiv(bare_id)
            meta = PaperMeta(
                id=paper_id,
                arxiv_id=bare_id,
                arxiv_label=f"arXiv:{bare_id}",
                version=version,
                title=(result.title or "").strip(),
                authors=[a.name for a in result.authors],
                abstract=(result.summary or "").strip(),
                categories=list(result.categories or []),
                published=result.published.date() if result.published else None,
                updated=result.updated.date() if result.updated else None,
                pdf_url=result.pdf_url,
                abs_url=result.entry_id,
                status=PaperStatus.QUEUED,
                ingested=False,
            )
            results.append(SearchResultItem(paper=meta, ingested=False, in_library=False))
        return results

    def _fetch_meta_via_package(self, arxiv_id: str) -> PaperMeta:
        import arxiv  # type: ignore

        search = arxiv.Search(id_list=[arxiv_id])
        client = arxiv.Client()
        result = next(iter(client.results(search)), None)
        if result is None:
            raise ValueError(f"arXiv id not found: {arxiv_id}")
        bare_id, version = _split_id_version(result.get_short_id())
        paper_id = paper_id_from_arxiv(bare_id)
        return PaperMeta(
            id=paper_id,
            arxiv_id=bare_id,
            arxiv_label=f"arXiv:{bare_id}",
            version=version,
            title=(result.title or "").strip(),
            authors=[a.name for a in result.authors],
            abstract=(result.summary or "").strip(),
            categories=list(result.categories or []),
            published=result.published.date() if result.published else None,
            updated=result.updated.date() if result.updated else None,
            pdf_url=result.pdf_url,
            abs_url=result.entry_id,
            status=PaperStatus.QUEUED,
            ingested=False,
        )
