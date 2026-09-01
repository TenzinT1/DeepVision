"""PDF downloader — stage 1 (Download).

Fetches the arXiv PDF into the local ``data/`` store.
"""

from __future__ import annotations

import abc
import urllib.error
import urllib.request

from deepvision.ingestion.paths import pdf_path
from deepvision.utils import get_logger

__all__ = ["PDFDownloader", "PDFDownloaderService"]

log = get_logger(__name__)


class PDFDownloader(abc.ABC):
    """Downloads a paper's PDF to local disk."""

    @abc.abstractmethod
    def download(self, arxiv_id: str, paper_id: str) -> str:
        """Download the PDF for ``arxiv_id`` and return its local path.

        The path is under ``config.data_dir/<paper_id>/main.pdf``.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# Concrete implementation
# --------------------------------------------------------------------------

_USER_AGENT = "DeepVision/0.1 (mailto:research-assistant@example.com)"
_TIMEOUT_S = 60


class PDFDownloaderService(PDFDownloader):
    """Downloads arXiv PDFs over HTTP using only the standard library.

    Caches: if ``main.pdf`` already exists locally (a prior successful
    download or a rerun), the download is skipped and the cached path is
    returned immediately.
    """

    def download(self, arxiv_id: str, paper_id: str) -> str:
        dest = pdf_path(paper_id)
        if dest.exists() and dest.stat().st_size > 0:
            log.info(
                "pdf already cached, skipping download",
                extra={"paper_id": paper_id, "path": str(dest)},
            )
            return str(dest)

        # Keep an explicit version if the caller passed one (e.g. "2411.10842v2").
        candidate_id = arxiv_id
        urls = [
            f"https://arxiv.org/pdf/{candidate_id}",
            f"https://arxiv.org/pdf/{candidate_id}.pdf",
            f"https://export.arxiv.org/pdf/{candidate_id}",
        ]
        last_error: Exception | None = None
        tmp_path = dest.with_suffix(".pdf.part")
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
                with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
                    data = resp.read()
                if not data:
                    raise ValueError("empty response body")
                tmp_path.write_bytes(data)
                tmp_path.replace(dest)
                log.info(
                    "downloaded pdf",
                    extra={"paper_id": paper_id, "url": url, "bytes": len(data)},
                )
                return str(dest)
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                last_error = exc
                log.warning(
                    "pdf download attempt failed, trying next mirror",
                    extra={"paper_id": paper_id, "url": url, "error": str(exc)},
                )
                continue
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

        raise RuntimeError(
            f"failed to download PDF for arXiv:{arxiv_id} from all mirrors: {last_error}"
        )
