"""ID helpers shared across DeepVision.

Every persistent entity (paper, job, chunk, report, chat message, section,
citation, media ref) gets a stable string id from one of these helpers. IDs are
URL-safe and human-inspectable so they can appear directly in API routes such as
``GET /media/{chunk_id}``.

This module is fully implemented (it is a shared utility, not builder-owned).
"""

from __future__ import annotations

import hashlib
import re
import uuid

__all__ = [
    "new_id",
    "job_id",
    "chunk_id",
    "report_id",
    "section_id",
    "citation_id",
    "media_id",
    "chat_id",
    "message_id",
    "normalize_arxiv_id",
    "paper_id_from_arxiv",
    "upload_paper_id",
    "upload_paper_id_from_digest",
    "slugify",
]

#: Number of hex chars of the sha256 digest kept in an upload paper id -
#: plenty of collision resistance for a single-user local library while
#: keeping the id (and its data/<id>/ directory name) short.
_UPLOAD_ID_DIGEST_LEN = 24

# A permissive arXiv id matcher covering both the modern ("2411.10842" with an
# optional version suffix) and legacy ("cs/0112017") identifier schemes, with an
# optional "arXiv:" prefix.
_ARXIV_RE = re.compile(
    r"^(?:arxiv:)?\s*(?P<id>(?:\d{4}\.\d{4,5}(?:v\d+)?)|(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?))$",
    re.IGNORECASE,
)


def new_id(prefix: str = "") -> str:
    """Return a fresh random id, optionally namespaced with ``<prefix>_``.

    Uses a uuid4 hex body so ids are globally unique without coordination
    between the parallel builder services.
    """
    body = uuid.uuid4().hex
    return f"{prefix}_{body}" if prefix else body


def job_id() -> str:
    """Return a new ingest-job id (``job_...``)."""
    return new_id("job")


def report_id() -> str:
    """Return a new report id (``rpt_...``)."""
    return new_id("rpt")


def section_id() -> str:
    """Return a new report-section id (``sec_...``)."""
    return new_id("sec")


def citation_id() -> str:
    """Return a new citation id (``cit_...``)."""
    return new_id("cit")


def media_id() -> str:
    """Return a new media-ref id (``med_...``)."""
    return new_id("med")


def chat_id() -> str:
    """Return a new chat-session id (``chat_...``)."""
    return new_id("chat")


def message_id() -> str:
    """Return a new chat-message id (``msg_...``)."""
    return new_id("msg")


def normalize_arxiv_id(raw: str) -> str:
    """Normalize a user-supplied arXiv identifier.

    Strips an optional ``arXiv:`` prefix and surrounding whitespace, lower-cases
    the scheme portion, and returns the bare id (version suffix preserved).

    Raises:
        ValueError: if ``raw`` is not a recognizable arXiv identifier.
    """
    match = _ARXIV_RE.match(raw.strip())
    if not match:
        raise ValueError(f"not a valid arXiv id: {raw!r}")
    return match.group("id")


def chunk_id(paper_id: str, ordinal: int) -> str:
    """Return a deterministic chunk id for ``paper_id`` at ``ordinal``.

    Deterministic ids let re-runs overwrite prior chunks in Chroma and SQLite
    instead of duplicating them. Format: ``<paper_id>::c<ordinal>``.
    """
    return f"{paper_id}::c{ordinal:05d}"


def paper_id_from_arxiv(arxiv_id: str) -> str:
    """Derive the canonical internal paper id from an arXiv id.

    The version suffix is dropped so ``2411.10842`` and ``2411.10842v2`` map to
    the same library entry, and the id is slugified to be filesystem-safe (it is
    used as a directory name under ``data/``).
    """
    base = normalize_arxiv_id(arxiv_id)
    base = re.sub(r"v\d+$", "", base)
    return slugify(base)


def upload_paper_id(data: bytes) -> str:
    """Derive a stable, filesystem-safe paper id for an uploaded PDF.

    Hashes the raw file bytes (sha256, truncated) so re-uploading the exact
    same PDF is idempotent — it maps to the same internal paper id / data
    directory — rather than creating a duplicate library entry each time.
    Distinct from :func:`paper_id_from_arxiv`, which only accepts real arXiv
    identifiers and raises on anything else.

    For large uploads, prefer streaming the bytes through a ``hashlib.sha256``
    incrementally and calling :func:`upload_paper_id_from_digest` with the
    resulting ``hexdigest()`` instead of buffering the whole file to pass here.
    """
    return upload_paper_id_from_digest(hashlib.sha256(data).hexdigest())


def upload_paper_id_from_digest(hexdigest: str) -> str:
    """Format an already-computed sha256 hexdigest as an upload paper id.

    Shares the ``upload-<hex prefix>`` convention with :func:`upload_paper_id`
    so a streamed, chunk-by-chunk hash (computed without ever holding the full
    file in memory) yields the exact same id as hashing the bytes directly.
    """
    return f"upload-{hexdigest[:_UPLOAD_ID_DIGEST_LEN]}"


def slugify(text: str) -> str:
    """Lower-case, collapse non-alphanumerics to ``-``, and trim.

    Used for filesystem-safe ids and directory names.
    """
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")
