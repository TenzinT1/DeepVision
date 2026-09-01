"""Upload router — POST /ingest/upload.

Owned by: INGESTION domain. Lets a user ingest their own PDF (any length, no
page limit) instead of an arXiv id: the file is streamed to
``data/<paper_id>/main.pdf`` in chunks, a stable id is minted from a hash of
the bytes, a minimal :class:`PaperMeta` is built (title sniffed from the PDF
itself), and the exact same ingestion pipeline / job-polling flow as
``POST /ingest`` takes over from there.

Response shape is the existing :class:`IngestResponse` — see its docstring in
``api/schemas.py`` and/§4.1 — so the frontend reuses its
``pollJob`` / ``IngestionModal`` flow unchanged.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from deepvision.api.deps import get_settings
from deepvision.api.schemas import IngestResponse
from deepvision.config import get_config
from deepvision.ingestion import repo
from deepvision.ingestion.orchestrator import DefaultIngestionOrchestrator
from deepvision.ingestion.paths import pdf_path
from deepvision.models.paper import PaperMeta
from deepvision.utils import get_logger
from deepvision.utils.ids import upload_paper_id_from_digest

router = APIRouter(tags=["ingest"])

log = get_logger(__name__)

_orchestrator = DefaultIngestionOrchestrator()

#: Chunk size for the disk-to-disk copy from the incoming upload to
#: ``data/<paper_id>/main.pdf`` — kept small and constant regardless of file
#: size so a very long PDF never gets fully buffered in memory.
_CHUNK_SIZE = 1024 * 1024  # 1 MiB

#: Hard ceiling on an uploaded PDF. There is no page limit — a 900-page book is
#: fine — but an unbounded stream would let a single request fill the disk under
#: ``data/``, and nothing downstream would ever clean it up. Enforced *during*
#: the streaming copy (not from Content-Length, which a client controls), so the
#: partial file is discarded the moment the limit is crossed.
_MAX_UPLOAD_BYTES = 512 * 1024 * 1024  # 512 MiB

#: arxiv_id placeholder convention for uploads, per the contracts agent: since
#: PaperMeta.arxiv_id is required, uploads use "upload:<paper_id>" rather than
#: making the field Optional (agents/base.py, agent_orchestrator.py,
#: synthesis_agent.py, report_generator.py, exporters.py all just read it as
#: an opaque display string, so the placeholder is a safe, non-invasive fit).
_ARXIV_ID_PREFIX = "upload:"
_ARXIV_LABEL = "Uploaded PDF"


def _looks_like_pdf(file: UploadFile) -> bool:
    content_type = (file.content_type or "").lower()
    if content_type == "application/pdf":
        return True
    filename = (file.filename or "").lower()
    return filename.endswith(".pdf")


def _open_pdf(path: Path):
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyMuPDF ('fitz') is required for PDF upload processing but is not installed"
        ) from exc
    return fitz.open(str(path))


def _derive_title(doc, filename: Optional[str]) -> str:
    """Best-effort title: PDF metadata -> first non-blank line of page 1 -> filename."""
    meta = doc.metadata or {}
    meta_title = str(meta.get("title") or "").strip()
    if meta_title:
        return meta_title
    try:
        page0_text = doc.load_page(0).get_text("text") or ""
        for line in page0_text.splitlines():
            line = line.strip()
            if line:
                return line[:300]
    except Exception:  # pragma: no cover - defensive, page unreadable
        pass
    stem = Path(filename).stem.strip() if filename else ""
    return stem or "Untitled upload"


@router.post("/ingest/upload", response_model=IngestResponse, status_code=202)
async def upload_pdf(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
) -> IngestResponse:
    """Queue an async ingestion job for an uploaded PDF (no arXiv id needed).

    Multipart form fields: ``file`` (the PDF bytes, required) and ``title``
    (optional override; falls back to the PDF's own metadata title, then the
    first line of page 1, then the filename).
    """
    if not _looks_like_pdf(file):
        raise HTTPException(
            status_code=415,
            detail="only application/pdf uploads are supported (filename must end in .pdf)",
        )

    data_dir = get_config().data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(data_dir), suffix=".upload.part")
    tmp_path = Path(tmp_name)
    hasher = hashlib.sha256()
    total_bytes = 0
    too_large = False
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            while True:
                chunk = await file.read(_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > _MAX_UPLOAD_BYTES:
                    too_large = True
                    break
                hasher.update(chunk)
                out.write(chunk)
    except Exception:
        # A client disconnect or a write failure mid-copy must not leave a
        # ``*.upload.part`` orphan under data/ — nothing else ever sweeps it.
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if too_large:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=(
                "uploaded file is too large "
                f"(limit {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"
            ),
        )

    if total_bytes == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="uploaded file is empty")

    paper_id = upload_paper_id_from_digest(hasher.hexdigest())
    dest = pdf_path(paper_id)  # also ensures data/<paper_id>/ exists
    reused_cached_pdf = dest.exists() and dest.stat().st_size > 0
    if reused_cached_pdf:
        # Same bytes already ingested once (hash match) - reuse the cached
        # PDF (this is what makes re-uploading idempotent) and drop the temp
        # copy instead of overwriting it.
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.replace(dest)

    try:
        doc = _open_pdf(dest)
        try:
            page_count = doc.page_count
            resolved_title = (title or "").strip() or _derive_title(doc, file.filename)
        finally:
            doc.close()
    except Exception as exc:
        # The bytes are not a readable PDF. Don't leave the rejected upload
        # sitting in data/<paper_id>/ - nothing else will ever clean it up,
        # and no PaperRow exists yet to make it reachable.
        if not reused_cached_pdf:
            dest.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                dest.parent.rmdir()  # only succeeds if we left it empty
        log.error(
            "failed to read uploaded PDF", extra={"paper_id": paper_id, "error": str(exc)}
        )
        # Keep the underlying exception (which embeds absolute server paths)
        # in the log only; the client gets a generic, actionable message.
        raise HTTPException(
            status_code=422,
            detail="could not read uploaded PDF - the file appears to be corrupt "
            "or is not a valid PDF",
        ) from exc

    meta = PaperMeta(
        id=paper_id,
        arxiv_id=f"{_ARXIV_ID_PREFIX}{paper_id}",
        arxiv_label=_ARXIV_LABEL,
        version=None,
        title=resolved_title,
        authors=[],
        abstract="",
        categories=[],
        published=None,
        updated=None,
        pdf_url=None,
        abs_url=None,
        page_count=page_count,
    )
    repo.upsert_paper_from_meta(meta, reset_status=True)

    settings = get_settings()
    try:
        job = _orchestrator.start_async(paper_id, settings)
    except Exception as exc:
        log.error(
            "failed to start ingestion for upload",
            extra={"paper_id": paper_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=500, detail=f"failed to start ingestion: {exc}"
        ) from exc

    return IngestResponse(job=job, paper_id=paper_id)
