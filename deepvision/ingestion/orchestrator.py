"""Ingestion orchestrator interface.

Coordinates the seven-stage async pipeline (Download -> Extract text -> Extract
images -> OCR -> Vision analysis -> Embedding -> Generate report), emitting
:class:`StageProgress` updates and log lines as it goes. The API's ingest/rerun
routes and the background worker drive this.

Report generation is the seventh *tracked* stage, not a fire-and-forget tail
call: the job only reaches ``done`` once the report exists (or its generation
has definitively failed, which is recorded on the stage). The abstract signature is the contract; the
body is not implemented here.
"""

from __future__ import annotations

import abc
import inspect
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from deepvision.config import get_config
from deepvision.models.chunks import AnyChunk
from deepvision.models.job import (
    STAGE_ORDER,
    IngestJob,
    JobStage,
    JobState,
    LogEntry,
    StageProgress,
    StageStatus,
)
from deepvision.models.paper import PaperStatus
from deepvision.models.settings import AppSettings
from deepvision.utils import get_logger
from deepvision.utils.ids import job_id

__all__ = [
    "IngestionOrchestrator",
    "DefaultIngestionOrchestrator",
    "build_chunker",
    "build_vector_store",
    "build_report_generator",
    "reconcile_job_stages",
]

log = get_logger(__name__)


class IngestionOrchestrator(abc.ABC):
    """Runs the full ingestion pipeline for one paper."""

    @abc.abstractmethod
    def run(self, paper_id: str, settings: AppSettings) -> Iterator[StageProgress]:
        """Execute the pipeline, yielding progress after each stage transition.

        Args:
            paper_id: the paper to ingest (its arXiv id is looked up from the DB).
            settings: effective settings for this run (persisted or overridden).

        Yields:
            :class:`StageProgress` snapshots as stages start/finish. Implementations
            should also persist the evolving :class:`IngestJob` so ``GET /jobs/{id}``
            reflects live state.

        Raises:
            NotImplementedError: contract stub.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def start_async(self, paper_id: str, settings: AppSettings) -> IngestJob:
        """Create + persist a queued job and schedule :meth:`run` in the background.

        Returns the created :class:`IngestJob` immediately (state = queued/running).
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# Concrete implementation
# --------------------------------------------------------------------------


def _now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _percent_for(stage: JobStage, status: StageStatus) -> int:
    idx = STAGE_ORDER.index(stage)
    total = len(STAGE_ORDER)
    fraction = idx / total if status == StageStatus.RUNNING else (idx + 1) / total
    return max(0, min(100, round(fraction * 100)))


def reconcile_job_stages(job: IngestJob) -> None:
    """Align ``job.stages`` with :data:`STAGE_ORDER`, in place.

    ``JobRow.stages`` is a persisted JSON column, so a job row written before a
    stage was added to ``STAGE_ORDER`` (e.g. any row predating the tracked
    "Generate report" stage) carries only the stages that existed then — and a
    rerun resumes that very row. Existing :class:`StageProgress` entries are
    kept, missing ones are seeded as ``pending``, and the list is returned to
    canonical order, so the orchestrator can't ``StopIteration`` looking up a
    stage that isn't there and the modal never renders a stale stage set.
    """
    existing = {sp.stage: sp for sp in job.stages}
    job.stages = [
        existing[s] if s in existing else StageProgress(stage=s) for s in STAGE_ORDER
    ]


def _resolve_concrete(module_path: str, base_cls: type, *, factory_names: tuple[str, ...]):
    """Instantiate a concrete implementation of ``base_cls`` from another
    domain's module without hard-coding that domain's class name.

    leaves the concrete class name up to the owning builder — the ingestion layer only
    depends on the rag layer / the report layer through their ABCs. This
    resolver first tries a conventional factory function (e.g.
    ``build_chunker`` / ``get_chunker``), then falls back to reflecting over
    the module for the first concrete (non-abstract) subclass of
    ``base_cls`` defined there.
    """
    import importlib

    module = importlib.import_module(module_path)

    for name in factory_names:
        factory = getattr(module, name, None)
        if callable(factory):
            try:
                return factory()
            except TypeError:
                continue

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, base_cls)
            and obj is not base_cls
            and not inspect.isabstract(obj)
        ):
            try:
                return obj()
            except TypeError:
                continue

    raise RuntimeError(
        f"no concrete implementation of {base_cls.__name__} found in {module_path}; "
        f"expected a factory function ({', '.join(factory_names)}) or a concrete "
        f"subclass with a no-arg constructor"
    )


def build_chunker():
    """Resolve and instantiate the rag layer's concrete ``Chunker`` implementation."""
    from deepvision.rag.chunker import Chunker

    return _resolve_concrete(
        "deepvision.rag.chunker",
        Chunker,
        factory_names=("build_chunker", "get_chunker", "default_chunker"),
    )


def build_vector_store():
    """Open the rag layer's persistent ``VectorStore`` at the default embedding dim.

    the rag layer keys its Chroma collection by embedding dimension and exposes
    ``open_vector_store(dim)`` (not a no-arg factory / no-arg concrete class),
    so the generic ``_resolve_concrete`` reflection can't build it. This helper
    is used by the delete/rerun routes (which don't know a paper's original
    embedding dim); it targets the default local embedding dim from config,
    matching how the default (local-first) provider ingests.
    """
    from deepvision.rag.vector_store import open_vector_store

    return open_vector_store(get_config().embedding_dim)


def build_report_generator():
    """Resolve and instantiate the report layer's concrete ``ReportGenerator`` implementation."""
    from deepvision.report.report_generator import ReportGenerator

    return _resolve_concrete(
        "deepvision.report.report_generator",
        ReportGenerator,
        factory_names=("build_report_generator", "get_report_generator", "default_report_generator"),
    )


class DefaultIngestionOrchestrator(IngestionOrchestrator):
    """Drives the pipeline stages in :data:`STAGE_ORDER`, persisting live
    progress to SQLite. Report generation is the last of those stages, so the
    job reaches ``done`` only after the report has actually been produced.

    ``run()`` always resumes/advances the *most recently created* job row for
    ``paper_id`` (creating one if none exists) since :class:`IngestJob`
    carries no explicit job id in its abstract signature — this mirrors the
    ``PaperMeta 1:N IngestJob`` relationship in, where the
    latest job is always the one actively running.
    """

    def __init__(self) -> None:
        # Imported lazily at call time inside run() as well to avoid heavy
        # PDF/vision deps loading merely by importing the orchestrator module;
        # instantiated once here since these are cheap, dependency-guarded
        # wrapper classes themselves.
        from deepvision.ingestion.image_extractor import ImageExtractorService
        from deepvision.ingestion.ocr_processor import OCRProcessorService
        from deepvision.ingestion.page_renderer import PageRendererService
        from deepvision.ingestion.pdf_downloader import PDFDownloaderService
        from deepvision.ingestion.pdf_parser import PDFParserService
        from deepvision.ingestion.vision_processor import VisionProcessorService

        self._downloader = PDFDownloaderService()
        self._parser = PDFParserService()
        self._img_extractor = ImageExtractorService()
        self._renderer = PageRendererService()
        self._ocr = OCRProcessorService()
        self._vision_proc = VisionProcessorService()

    # -- IngestionOrchestrator -------------------------------------------

    def start_async(self, paper_id: str, settings: AppSettings) -> IngestJob:
        from deepvision.ingestion import repo

        paper = repo.get_paper_meta(paper_id)
        if paper is None:
            raise ValueError(f"unknown paper_id: {paper_id!r}")

        job = IngestJob.new(id=job_id(), paper_id=paper_id, arxiv_id=paper.arxiv_id)
        repo.persist_job(job)

        thread = threading.Thread(
            target=self._run_worker,
            args=(paper_id, settings),
            name=f"ingest-{paper_id}",
            daemon=True,
        )
        thread.start()
        return job

    def run(self, paper_id: str, settings: AppSettings) -> Iterator[StageProgress]:
        from deepvision.ingestion import repo

        plog = get_logger(__name__, paper_id=paper_id)

        paper = repo.get_paper_meta(paper_id)
        if paper is None:
            raise ValueError(f"unknown paper_id: {paper_id!r}")

        # Chapter-report and study-generation jobs live in the *same* ``jobs``
        # table under the same paper_id and are deliberately stage-less
        # (``stages: []``). Resuming one of those would overwrite its row with
        # ingestion state and, since reconcile seeds the full STAGE_ORDER,
        # retro-fit it with seven stages it never ran — so ask for the latest
        # *ingestion* job specifically rather than looking at the newest row of
        # any kind and bailing when it happens to be stage-less (which would
        # also abandon a real ingestion row sitting just behind it).
        job = repo.latest_ingestion_job_for_paper(paper_id)
        if job is not None and not job.stages:  # defensive: can't happen
            job = None
        if job is None:
            job = IngestJob.new(id=job_id(), paper_id=paper_id, arxiv_id=paper.arxiv_id)

        # Old job rows (persisted before "Generate report" joined STAGE_ORDER)
        # are resumed by rerun; seed whatever they are missing. See
        # reconcile_job_stages().
        reconcile_job_stages(job)

        job.state = JobState.RUNNING
        job.error_message = None
        repo.persist_job(job)
        repo.set_paper_status(paper_id, PaperStatus.PROCESSING)

        arxiv_id = job.arxiv_id

        def log_line(stage: Optional[JobStage], message: str) -> None:
            job.log.append(LogEntry(t=_now_hms(), m=message, stage=stage))

        def stage_progress(stage: JobStage) -> StageProgress:
            return next(sp for sp in job.stages if sp.stage == stage)

        def start_stage(stage: JobStage) -> None:
            sp = stage_progress(stage)
            sp.status = StageStatus.RUNNING
            sp.started_at = datetime.utcnow()
            job.current_stage = stage
            job.percent = _percent_for(stage, StageStatus.RUNNING)
            repo.persist_job(job)

        def finish_stage(
            stage: JobStage, detail: Optional[str] = None, *, degraded: bool = False
        ) -> None:
            sp = stage_progress(stage)
            sp.status = StageStatus.DONE
            sp.finished_at = datetime.utcnow()
            sp.detail = detail
            sp.degraded = degraded
            job.percent = _percent_for(stage, StageStatus.DONE)
            repo.persist_job(job)

        def error_stage_nonfatal(stage: JobStage, detail: str) -> None:
            """Mark a stage as errored but keep the overall job running — used
            for enhancement stages (images/OCR/vision) whose failure shouldn't
            abort a pipeline that can still produce a usable report, and for the
            final report stage, whose failure leaves a paper that is still fully
            usable for chat/compare."""
            sp = stage_progress(stage)
            sp.status = StageStatus.ERROR
            sp.finished_at = datetime.utcnow()
            sp.detail = detail[:200]
            job.percent = _percent_for(stage, StageStatus.DONE)
            repo.persist_job(job)

        def abort(stage: JobStage, message: str) -> None:
            sp = stage_progress(stage)
            sp.status = StageStatus.ERROR
            sp.finished_at = datetime.utcnow()
            sp.detail = message[:200]
            job.state = JobState.ERROR
            job.error_message = message
            log_line(stage, f"ERROR: {message}")
            repo.persist_job(job)
            repo.set_paper_status(paper_id, PaperStatus.ERROR, error_message=message)

        all_text_chunks: list[AnyChunk] = []
        all_image_chunks: list[AnyChunk] = []
        all_ocr_chunks: list[AnyChunk] = []
        all_vision_chunks: list[AnyChunk] = []
        pdf_path_abs: Optional[str] = None

        # ---- Stage 1: Download ------------------------------------------
        stage = JobStage.DOWNLOAD
        start_stage(stage)
        try:
            log_line(stage, f"fetching arXiv:{arxiv_id} ...")
            pdf_path_abs = self._downloader.download(arxiv_id, paper_id)
            size_mb = Path(pdf_path_abs).stat().st_size / (1024 * 1024)
            log_line(stage, f"downloaded main.pdf ({size_mb:.1f} MB)")
            finish_stage(stage, detail=f"{size_mb:.1f} MB")
        except Exception as exc:
            plog.error("download failed", extra={"error": str(exc)})
            abort(stage, str(exc))
            yield stage_progress(stage)
            return
        yield stage_progress(stage)

        # ---- Stage 2: Extract text (+ page renders) ---------------------
        stage = JobStage.EXTRACT_TEXT
        start_stage(stage)
        try:
            page_count = self._parser.page_count(pdf_path_abs)
            all_text_chunks = list(self._parser.parse(pdf_path_abs, paper_id))
            self._renderer.render_all(pdf_path_abs, paper_id)
            thumb_rel = self._renderer.thumbnail(pdf_path_abs, paper_id)
            log_line(stage, f"extracted text from {page_count} pages")
            finish_stage(stage, detail=f"{page_count} pages")
            repo.update_paper_after_text(
                paper_id, page_count=page_count, thumbnail_path=thumb_rel
            )
        except Exception as exc:
            plog.error("extract text failed", extra={"error": str(exc)})
            abort(stage, str(exc))
            yield stage_progress(stage)
            return
        yield stage_progress(stage)

        # ---- Stage 3: Extract images -------------------------------------
        stage = JobStage.EXTRACT_IMAGES
        start_stage(stage)
        try:
            all_image_chunks = list(self._img_extractor.extract(pdf_path_abs, paper_id))
            log_line(stage, f"extracted {len(all_image_chunks)} figures/tables")
            finish_stage(stage, detail=f"{len(all_image_chunks)} figures")
            repo.update_paper_figure_count(paper_id, len(all_image_chunks))
        except Exception as exc:
            plog.error("extract images failed, continuing without figures", extra={"error": str(exc)})
            error_stage_nonfatal(stage, str(exc))
            log_line(stage, f"image extraction unavailable: {exc}")
        yield stage_progress(stage)

        # ---- Stage 4: OCR --------------------------------------------------
        stage = JobStage.OCR
        start_stage(stage)
        try:
            all_ocr_chunks = list(
                self._ocr.process(all_image_chunks, paper_id, language=settings.ocr_language)
            )
            log_line(
                stage,
                f"OCR ({settings.ocr_language.value}) recovered text from "
                f"{len(all_ocr_chunks)} region(s)",
            )
            finish_stage(stage, detail=f"{len(all_ocr_chunks)} regions")
        except Exception as exc:
            plog.warning("OCR stage failed, continuing", extra={"error": str(exc)})
            error_stage_nonfatal(stage, str(exc))
            log_line(stage, f"OCR unavailable: {exc}")
        yield stage_progress(stage)

        # ---- Stage 5: Vision analysis ---------------------------------------
        stage = JobStage.VISION
        start_stage(stage)
        try:
            from deepvision.providers.factory import build_vision

            vision_provider = build_vision(
                settings, strict=get_config().strict_providers
            )
            all_vision_chunks = list(
                self._vision_proc.process(all_image_chunks, paper_id, vision_provider)
            )
            log_line(stage, f"vision analysis on {len(all_vision_chunks)} figure(s)")
            finish_stage(stage, detail=f"{len(all_vision_chunks)} figures")
        except Exception as exc:
            plog.warning("vision stage failed, continuing", extra={"error": str(exc)})
            error_stage_nonfatal(stage, str(exc))
            log_line(stage, f"vision analysis unavailable: {exc}")
        yield stage_progress(stage)

        # ---- Stage 6: Embedding ----------------------------------------------
        stage = JobStage.EMBEDDING
        start_stage(stage)
        try:
            from deepvision.providers.factory import build_embeddings
            from deepvision.rag.embedding_pipeline import embed_chunks
            from deepvision.rag.vector_store import open_vector_store

            chunker = build_chunker()
            raw_chunks: list[AnyChunk] = [
                *all_text_chunks,
                *all_image_chunks,
                *all_ocr_chunks,
                *all_vision_chunks,
            ]
            final_chunks = list(chunker.chunk(raw_chunks, settings))
            embed_provider = build_embeddings(
                settings, strict=get_config().strict_providers
            )
            # Use the rag layer's embed_chunks glue (text/caption selection + optional
            # image embedding, dropping empty chunks) rather than embedding raw
            # ``.text`` — this keeps figure captions retrievable and avoids
            # upserting meaningless zero vectors for image-only chunks. The
            # vector store is opened at the embedding provider's actual dim so
            # ingestion and retrieval share the same dim-keyed collection.
            up_chunks, up_vectors = embed_chunks(final_chunks, settings, embed_provider)
            vector_store = open_vector_store(embed_provider.dim)
            if up_chunks:
                vector_store.upsert(up_chunks, up_vectors)
            log_line(stage, f"embedded {len(up_chunks)} chunk(s) into the vector store")
            finish_stage(stage, detail=f"{len(up_chunks)} chunks")
        except Exception as exc:
            plog.error("embedding failed", extra={"error": str(exc)})
            abort(stage, str(exc))
            yield stage_progress(stage)
            return

        # ---- The paper is USABLE now -------------------------------------
        # Chunks, vectors, page images and figure crops are all persisted, so
        # chat and compare would work perfectly from this instant. Flip the
        # paper-level signal here rather than after the report stage: on a local
        # model that stage runs for ~20 minutes, and gating chat/compare on it
        # made a freshly ingested paper un-chattable and invisible to Compare
        # for that whole window.
        #
        # This does NOT re-open the "Open report before a report exists" bug —
        # that button is gated on the separate ``PaperMeta.has_report`` signal
        # (derived from the existence of a ``reports`` row), not on this one.
        # The JOB still only reaches ``done`` after the report stage below.
        # Set BEFORE the yield, so the snapshot a caller stepping run() observes
        # for the Embedding stage already reflects the unlocked paper.
        repo.set_paper_status(paper_id, PaperStatus.READY, ingested=True)
        log_line(None, "paper ready for chat and compare (report still generating)")
        repo.persist_job(job)
        yield stage_progress(stage)

        # ---- Stage 7: Generate report -----------------------------------
        # Report generation (the report layer) is a TRACKED stage, not a tail call after
        # the job is marked done: it is the slowest step in the pipeline (agent
        # ensemble, minutes on a local model), and the modal's "Open report ->"
        # affordance is gated on this stage succeeding. Marking the job done
        # first would offer a report that does not exist yet.
        stage = JobStage.REPORT
        start_stage(stage)
        report_ok = False
        try:
            from deepvision.agents.base import llm_fallback_count, reset_llm_fallbacks

            log_line(stage, "generating report (agent ensemble)...")
            # Counted on THIS thread, around the whole ensemble. A hard failure
            # (build_report raising) lands in the except below; the far more
            # common failure on a local model is quieter — individual LLM calls
            # time out, complete_with_fallback swallows the exception and returns
            # the extractive draft, and every layer above sees a clean success.
            # Without this the modal green-checks a report of which half was
            # never written by the model, which is the same silent degradation
            # the tracked stage exists to expose.
            reset_llm_fallbacks()
            report_generator = build_report_generator()
            report_generator.generate(paper_id, settings)
            degraded = llm_fallback_count()
            report_ok = True
            if degraded:
                detail = f"ready — degraded ({degraded} model call(s) fell back)"
                log_line(
                    stage,
                    f"report ready, but {degraded} model call(s) timed out or failed and "
                    "fell back to extractive drafts — those parts are not "
                    "model-written. Re-run this paper from the Library (or switch "
                    "to a faster model in Settings) for a full-quality report.",
                )
            else:
                detail = "ready"
                log_line(stage, "report ready")
            finish_stage(stage, detail=detail, degraded=bool(degraded))
            plog.info("report generated", extra={"llm_fallbacks": degraded})
        except Exception as exc:
            # Non-fatal, deliberately: chunks, vectors and images are already
            # persisted, so the paper is genuinely usable for chat and compare.
            # The paper is NOT thrown away — but unlike the previous silent
            # `except`, the failure is recorded on the stage (and in the log) so
            # the UI can show it instead of presenting an empty report as fine.
            plog.error("report generation failed", extra={"error": str(exc)})
            error_stage_nonfatal(stage, str(exc))
            log_line(stage, f"ERROR: report generation failed: {exc}")
        yield stage_progress(stage)

        # ---- Done -------------------------------------------------------
        job.state = JobState.DONE
        job.current_stage = None
        job.percent = 100
        log_line(
            None,
            "ingestion complete"
            if report_ok
            else "ingestion complete — report generation failed (chat and compare "
            "still work; re-run this paper from the Library to retry the report)",
        )
        repo.persist_job(job)
        # NOTE: the paper was already marked READY/ingested right after the
        # embedding stage (see above) — deliberately, so chat and compare unlock
        # as soon as the content exists rather than ~20 minutes later. Nothing
        # between there and here can move it off READY except abort(), which
        # returns. Do not re-add a second set_paper_status here, and do not move
        # the first one back down: "Open report" is gated on the separate
        # ``PaperMeta.has_report`` signal, which is derived from the ``reports``
        # row this stage writes.

    # -- internal ----------------------------------------------------------

    def _run_worker(self, paper_id: str, settings: AppSettings) -> None:
        try:
            for _ in self.run(paper_id, settings):
                pass
        except Exception as exc:  # pragma: no cover - defensive top-level guard
            log.exception("ingestion worker crashed", extra={"paper_id": paper_id})
            self._mark_worker_failed(paper_id, exc)

    @staticmethod
    def _mark_worker_failed(paper_id: str, exc: Exception) -> None:
        """Best-effort terminal-state repair for a worker crash that happened
        outside run()'s per-stage try/except (e.g. a transient DB error
        raised out of start_stage()/persist_job()).

        Without this, the job row is left at whatever state the last
        successful persist_job() wrote (often ``running``) and the paper stays
        ``processing`` forever -- GET /jobs/{id} would report a job that will
        never finish. Guarded end-to-end so a secondary DB failure here is
        logged, not raised, out of a background thread.
        """
        try:
            from deepvision.ingestion import repo

            message = str(exc) or exc.__class__.__name__
            job = repo.latest_job_for_paper(paper_id)
            if job is not None and job.state not in (JobState.DONE, JobState.ERROR):
                job.state = JobState.ERROR
                job.error_message = message
                job.log.append(
                    LogEntry(t=_now_hms(), m=f"ERROR: worker crashed: {message}", stage=job.current_stage)
                )
                repo.persist_job(job)
            repo.set_paper_status(paper_id, PaperStatus.ERROR, error_message=message)
        except Exception:
            log.exception(
                "failed to persist terminal error state after worker crash",
                extra={"paper_id": paper_id},
            )
