#!/usr/bin/env python
"""End-to-end developer smoke check for DeepVision (NOT a pytest suite).

Runs the whole local-first pipeline against a synthetic fixture PDF using the
default (offline) providers — EchoLLM / NullVision / HashEmbeddings — so it needs
no model weights, no Ollama, and no network:

  1. Isolate a fresh data/db/chroma dir and ``init_db()``.
  2. Generate a tiny 3-page fixture PDF (heading + paragraph + figure + table)
     with PyMuPDF and register a PaperRow for it.
  3. Run the ingestion orchestrator end-to-end and assert every ``STAGE_ORDER``
     stage completes (including the final report stage, which must be the last
     progress the job yields), chunks land in the vector store, and a Report
     with every section in ``SECTION_ORDER`` + a stats bar is produced and
     persisted *before* the job reaches ``done``.
  4. Assert the *section contract* holds: every ``SectionName``-keyed table
     covers ``SECTION_ORDER`` exactly, the TypeScript mirror agrees with the
     Python enum, every section is stamped with the question it answers, no
     draft scaffolding leaked into a body, no prose section is nothing but
     figure captions or repeated lines, and ``At a Glance`` keeps its five
     fixed fields. Under EchoLLM the extractive draft *is* the output, so this
     sees exactly what a reader with no model installed would see.
  5. Unit-check the two deterministic helpers (``rag.chunk_quality`` and
     ``report.figure_links``) against the exact strings that caused the
     original bug.
  6. Exercise the FastAPI app in-process (TestClient) for GET /health,
     GET /report/{id}, POST /chat, GET /settings and assert 200 + shape.

Prints a clear PASS/FAIL summary and exits non-zero on any failure.

Usage:  .venv/bin/python scripts/smoke_check.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# --------------------------------------------------------------------------
# 0. Isolate all persistent state into a throwaway dir BEFORE importing
#    deepvision, so config (lru_cached, env-driven) points at the sandbox.
# --------------------------------------------------------------------------
_SANDBOX = Path(tempfile.mkdtemp(prefix="deepvision-smoke-"))
os.environ["DEEPVISION_DATA_DIR"] = str(_SANDBOX / "data")
os.environ["DEEPVISION_SQLITE_PATH"] = str(_SANDBOX / "data" / "deepvision.db")
os.environ["DEEPVISION_CHROMA_DIR"] = str(_SANDBOX / "data" / "chroma")
# Force the offline Echo/Null/Hash provider stubs so the smoke check runs with
# NO models installed (real providers now default on via strict_providers=True).
os.environ["DEEPVISION_STRICT_PROVIDERS"] = "0"
# Keep logs quiet-ish but visible.
os.environ.setdefault("DEEPVISION_LOG_LEVEL", "WARNING")

# Make the repo importable when run as `python scripts/smoke_check.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class Checks:
    """Tiny assertion recorder that keeps going and prints a summary."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.passed.append(name)
            print(f"  \033[32mPASS\033[0m  {name}" + (f"  ({detail})" if detail else ""))
        else:
            self.failed.append(name)
            print(f"  \033[31mFAIL\033[0m  {name}" + (f"  ({detail})" if detail else ""))
        return bool(condition)

    def summary(self) -> bool:
        total = len(self.passed) + len(self.failed)
        print("\n" + "=" * 64)
        if self.failed:
            print(f"RESULT: FAIL  ({len(self.passed)}/{total} checks passed)")
            print("Failed checks:")
            for name in self.failed:
                print(f"  - {name}")
        else:
            print(f"RESULT: PASS  (all {total} checks passed)")
        print("=" * 64)
        return not self.failed


def make_fixture_pdf(pdf_path: Path) -> None:
    """Create a small 3-page PDF: heading, paragraph, a figure, and a table."""
    import fitz  # PyMuPDF

    doc = fitz.open()

    # ---- Page 1: title + abstract paragraph ----
    p1 = doc.new_page(width=595, height=842)  # A4 in points
    p1.insert_text((72, 90), "Emergent Multimodal Abilities at Scale",
                    fontsize=20, fontname="helv")
    p1.insert_text((72, 140), "Abstract", fontsize=14, fontname="helv")
    abstract = (
        "We investigate how multimodal pretraining at scale gives rise to "
        "emergent abilities that smaller models lack. Across fourteen downstream "
        "tasks we observe sharp, non-linear transitions in capability as compute "
        "increases. Our method combines an interleaved decoder with contrastive "
        "alignment over two billion image-text pairs, and we evaluate on document "
        "understanding, table reasoning, and figure question answering benchmarks."
    )
    _insert_wrapped(p1, (72, 170), abstract, fontsize=11, width_chars=80)

    # ---- Page 2: Introduction heading + paragraph + a figure ----
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((72, 90), "1  Introduction", fontsize=15, fontname="helv")
    intro = (
        "An ability is called emergent when it is essentially absent below a "
        "scale threshold and present above it. Prior work reports such thresholds "
        "in language-only models; here we extend the analysis to the multimodal "
        "regime and show the same qualitative behavior on visual reasoning tasks."
    )
    _insert_wrapped(p2, (72, 120), intro, fontsize=11, width_chars=80)

    # Draw a simple raster "figure" (a colored gradient block) and embed it.
    fig_pix = _make_figure_pixmap(fitz)
    fig_rect = fitz.Rect(120, 220, 420, 400)
    p2.insert_image(fig_rect, pixmap=fig_pix)
    p2.insert_text((120, 420), "Figure 1: Emergent break points across tasks vs. compute.",
                   fontsize=10, fontname="helv")

    # ---- Page 3: Results heading + a ruled table + caption ----
    p3 = doc.new_page(width=595, height=842)
    p3.insert_text((72, 90), "2  Results", fontsize=15, fontname="helv")
    results = (
        "Our model reaches state of the art on eleven of fourteen benchmarks, "
        "with the largest gains on multi-hop table question answering."
    )
    _insert_wrapped(p3, (72, 120), results, fontsize=11, width_chars=80)
    _draw_table(p3, fitz, origin=(90, 180))
    p3.insert_text((90, 320), "Table 1: Accuracy by benchmark and modality ablation.",
                   fontsize=10, fontname="helv")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(pdf_path))
    doc.close()


def _insert_wrapped(page, origin, text, *, fontsize, width_chars):
    import textwrap

    x, y = origin
    for i, line in enumerate(textwrap.wrap(text, width=width_chars)):
        page.insert_text((x, y + i * (fontsize + 4)), line, fontsize=fontsize, fontname="helv")


def _make_figure_pixmap(fitz):
    """A small non-trivial RGB pixmap so get_image_info() detects a figure."""
    width, height = 240, 140
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), False)
    for y in range(height):
        for x in range(width):
            pix.set_pixel(x, y, ((x * 255) // width, (y * 255) // height, 128))
    return pix


def _draw_table(page, fitz, *, origin):
    """Draw a simple ruled 3x3 table so PyMuPDF find_tables() can detect it."""
    x0, y0 = origin
    cols = [0, 120, 240, 360]
    rows = [0, 30, 60, 90]
    shape = page.new_shape()
    for c in cols:
        shape.draw_line((x0 + c, y0), (x0 + c, y0 + rows[-1]))
    for r in rows:
        shape.draw_line((x0, y0 + r), (x0 + cols[-1], y0 + r))
    shape.finish(width=0.8, color=(0, 0, 0))
    shape.commit()
    headers = [["Model", "DocVQA", "TableQA"],
               ["Ours", "84.2", "71.5"],
               ["Base", "67.8", "55.0"]]
    for ri, row in enumerate(headers):
        for ci, val in enumerate(row):
            page.insert_text((x0 + cols[ci] + 8, y0 + rows[ri] + 20), val,
                             fontsize=9, fontname="helv")


def main() -> int:
    checks = Checks()
    print(f"DeepVision smoke check\nSandbox: {_SANDBOX}\n")

    # ---- Stage A: DB init -------------------------------------------------
    print("[1] Database + config")
    from deepvision.config import get_config
    from deepvision.db import init_db

    cfg = get_config()
    cfg.ensure_dirs()
    init_db(cfg)
    checks.check("config points at sandbox", str(cfg.data_dir).startswith(str(_SANDBOX)),
                 str(cfg.data_dir))
    checks.check("sqlite db created", cfg.sqlite_path.exists(), str(cfg.sqlite_path))

    # ---- Stage B: fixture PDF + paper row --------------------------------
    print("\n[2] Fixture PDF + paper registration")
    from datetime import date

    from deepvision.ingestion import repo
    from deepvision.ingestion.paths import pdf_path
    from deepvision.models import PaperMeta, PaperStatus

    arxiv_id = "2401.00001"
    paper_id = "2401-00001"
    meta = PaperMeta(
        id=paper_id,
        arxiv_id=arxiv_id,
        arxiv_label=f"arXiv:{arxiv_id}",
        version="v1",
        title="Emergent Multimodal Abilities at Scale",
        authors=["A. Researcher", "B. Scientist"],
        abstract="We investigate emergent multimodal abilities at scale.",
        categories=["cs.CL", "cs.LG"],
        published=date(2024, 1, 2),
        updated=date(2024, 1, 3),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        status=PaperStatus.QUEUED,
        ingested=False,
    )
    repo.upsert_paper_from_meta(meta, reset_status=True)

    target_pdf = pdf_path(paper_id)  # data/<pid>/main.pdf
    make_fixture_pdf(target_pdf)
    checks.check("fixture PDF written", target_pdf.exists() and target_pdf.stat().st_size > 0,
                 f"{target_pdf.stat().st_size} bytes")

    import fitz

    _d = fitz.open(str(target_pdf))
    n_pages = _d.page_count
    _d.close()
    checks.check("fixture PDF has 3 pages", n_pages == 3, f"{n_pages} pages")

    # ---- Stage C: run the ingestion orchestrator -------------------------
    from deepvision.models import STAGE_ORDER, JobStage, StageStatus

    print(f"\n[3] Ingestion pipeline ({len(STAGE_ORDER)} stages)")
    from deepvision.api.settings_store import default_settings
    from deepvision.ingestion.orchestrator import DefaultIngestionOrchestrator

    settings = default_settings(cfg)
    orch = DefaultIngestionOrchestrator()

    progresses = list(orch.run(paper_id, settings))
    # Reload the persisted job to inspect final per-stage status.
    job = repo.latest_job_for_paper(paper_id)
    checks.check("ingest job persisted", job is not None, job.id if job else "None")

    if job is not None:
        seen_stages = {sp.stage for sp in job.stages}
        checks.check(f"all {len(STAGE_ORDER)} stages present",
                     seen_stages == set(STAGE_ORDER),
                     f"{len(seen_stages)}/{len(STAGE_ORDER)}")
        checks.check("stages persisted in STAGE_ORDER order",
                     [sp.stage for sp in job.stages] == list(STAGE_ORDER),
                     ", ".join(sp.stage.value for sp in job.stages))
        # Every stage must have completed (enhancement stages may be 'done'
        # even with zero output; none should be left pending/running/error).
        terminal = {StageStatus.DONE, StageStatus.ERROR}
        all_terminal = all(sp.status in terminal for sp in job.stages)
        non_pending = all(sp.status != StageStatus.PENDING for sp in job.stages)
        checks.check("no stage left pending/running", non_pending and all_terminal,
                     ", ".join(f"{sp.stage.value}={sp.status.value}" for sp in job.stages))
        core_stages = (JobStage.DOWNLOAD, JobStage.EXTRACT_TEXT, JobStage.EMBEDDING,
                       JobStage.REPORT)
        core_done = all(
            sp.status == StageStatus.DONE for sp in job.stages if sp.stage in core_stages
        )
        checks.check("core stages (download/text/embedding/report) done", core_done)
        checks.check("job state == done", job.state.value == "done", job.state.value)

    checks.check("orchestrator yielded progress for each stage",
                 len(progresses) >= len(STAGE_ORDER),
                 f"{len(progresses)} yields")
    # Report generation is a tracked stage, so it is the LAST thing the job does
    # -- the job must not reach `done` (and the modal must not offer "Open
    # report") while generation is still pending.
    checks.check("report generation is the final tracked stage",
                 bool(progresses) and progresses[-1].stage == JobStage.REPORT,
                 progresses[-1].stage.value if progresses else "no yields")

    # Backward compatibility: `jobs.stages` is a persisted JSON column, so rows
    # written before a stage joined STAGE_ORDER carry fewer stages -- and a
    # rerun resumes that same row. Reconciling must seed the missing stage(s)
    # as pending without dropping the recorded ones or reordering them.
    from deepvision.ingestion.orchestrator import reconcile_job_stages
    from deepvision.models import IngestJob, StageProgress

    legacy = IngestJob.new(id="job_legacy", paper_id=paper_id, arxiv_id=arxiv_id)
    legacy.stages = [
        StageProgress(stage=s, status=StageStatus.DONE, detail="from an old row")
        for s in STAGE_ORDER
        if s is not JobStage.REPORT
    ]
    legacy_count = len(legacy.stages)
    reconcile_job_stages(legacy)
    checks.check("legacy 6-stage job row reconciles to STAGE_ORDER",
                 [sp.stage for sp in legacy.stages] == list(STAGE_ORDER),
                 f"{legacy_count} -> {len(legacy.stages)} stages")
    checks.check("reconcile keeps recorded stage results and seeds only the new one",
                 all(sp.status == StageStatus.DONE for sp in legacy.stages
                     if sp.stage is not JobStage.REPORT)
                 and legacy.stages[-1].stage is JobStage.REPORT
                 and legacy.stages[-1].status == StageStatus.PENDING)

    # ---- Stage D: chunks landed in the vector store ----------------------
    print("\n[4] Vector store (Chroma)")
    from deepvision.providers.factory import build_embeddings
    from deepvision.rag.vector_store import _HAS_CHROMADB, open_vector_store

    emb = build_embeddings(settings)
    store = open_vector_store(emb.dim)
    backend = "chromadb" if (_HAS_CHROMADB and store._collection is not None) else "json-fallback"
    qvec = emb.embed_query("emergent multimodal abilities and results")
    hits = store.query(qvec, paper_id=paper_id, k=6)
    checks.check("vector store returns hits for the paper", len(hits) > 0,
                 f"{len(hits)} hits via {backend}")
    if hits:
        checks.check("hits carry this paper_id in metadata",
                     all(h.metadata.get("paper_id") == paper_id for h in hits))

    # ---- Stage E: eager Report was produced + persisted ------------------
    from deepvision.models import SECTION_ORDER
    print(f"\n[5] Report ({len(SECTION_ORDER)} sections + stats)")
    from deepvision.report.report_generator import DefaultReportGenerator

    gen = DefaultReportGenerator()
    # Must already exist: the report stage runs *inside* the job, before done.
    report = gen.load(paper_id)
    checks.check("report persisted by the ingestion job's report stage",
                 report is not None, report.id)
    names = [s.name for s in report.sections]
    checks.check(f"report has all {len(SECTION_ORDER)} sections in order",
                 names == SECTION_ORDER,
                 ", ".join(s.name.value for s in report.sections))
    checks.check("report has a stats bar",
                 report.stats is not None and report.stats.pages >= 1,
                 f"pages={report.stats.pages}, figures={report.stats.figures}, "
                 f"citations={report.stats.citations_extracted}, "
                 f"read={report.stats.reading_time_min}min")
    checks.check("report references the paper", report.paper is not None
                 and report.paper.id == paper_id)
    total_citations = sum(len(s.citations) for s in report.sections)
    total_media = sum(len(s.media) for s in report.sections)
    checks.check("report sections carry grounded citations", total_citations > 0,
                 f"{total_citations} citations")
    checks.check("figures section carries media", total_media > 0, f"{total_media} media refs")

    # ---- Stage E2: the section contract itself ----------------------------
    # Everything below guards a regression that actually reached a reader: a
    # section set whose members duplicated each other, figure captions printed
    # as body prose, and draft scaffolding shipped as visible text. Under
    # EchoLLM the extractive draft *is* the output, so these checks see exactly
    # what a user with no model installed would see.
    print("\n[5b] Report section contract")
    import re as _re

    from deepvision.agents.research_agent import _SECTION_QUERIES
    from deepvision.models.report import (
        AT_A_GLANCE_FIELDS,
        SECTION_QUESTIONS,
        SectionName,
    )
    from deepvision.report.interactive_sections import SECTION_BADGES

    # Every SectionName-keyed table that gets *iterated over SECTION_ORDER* must
    # cover it exactly. A missing key never raises — it silently ships an empty
    # section, which is precisely how a rename of the section set would rot.
    for _label, _table in (
        ("SECTION_QUESTIONS", SECTION_QUESTIONS),
        ("SECTION_BADGES", SECTION_BADGES),
        ("research_agent._SECTION_QUERIES", _SECTION_QUERIES),
    ):
        _missing = [n.value for n in SECTION_ORDER if n not in _table]
        _extra = [getattr(k, "value", k) for k in _table if k not in SECTION_ORDER]
        checks.check(f"{_label} covers exactly SECTION_ORDER",
                     not _missing and not _extra,
                     f"missing={_missing} extra={_extra}")

    # The TS mirror is a second source of truth by necessity; drift shows up as
    # a section the UI silently cannot render.
    _ts_src = (_REPO_ROOT / "frontend" / "src" / "api" / "types.ts").read_text(encoding="utf-8")
    _ts_match = _re.search(
        r"export const SECTION_ORDER: SectionName\[\] = \[(.*?)\];", _ts_src, _re.S
    )
    _ts_names = _re.findall(r'"([^"]+)"', _ts_match.group(1)) if _ts_match else []
    checks.check("frontend SECTION_ORDER mirrors the Python one",
                 _ts_names == [n.value for n in SECTION_ORDER],
                 f"ts={_ts_names}")

    checks.check("every section is stamped with the question it answers",
                 all(s.question == SECTION_QUESTIONS[s.name] for s in report.sections),
                 str([s.name.value for s in report.sections if not s.question]))

    # Draft scaffolding must never reach a reader. The first fragment here
    # literally shipped at the top of a real Overview section a user read.
    _SCAFFOLD = (
        "Paper abstract (context", "not to be cited",
        "HARD LENGTH LIMIT", "HARD COUNT LIMIT", "FORMATTING:",
        "ALREADY COVERED", "ANSWERED ELSEWHERE",
        "This section answers exactly one question",
    )
    _leaks = [
        (s.name.value, frag)
        for s in report.sections
        for frag in _SCAFFOLD
        if frag.lower() in
        f"{s.body_markdown or ''} {s.deep_dive_markdown or ''}".lower()
    ]
    checks.check("no draft scaffolding leaks into a section body", not _leaks, str(_leaks))

    # The observed failure: a whole Background section that was nothing but the
    # same two figure captions, each printed twice.
    _CAPTION_LINE = _re.compile(r"^\s*(fig(ure)?|table)\s*\.?\s*\d+\s*[:.\-—)]", _re.I)
    _caption_only: list[str] = []
    _dupe_lines: list[str] = []
    for _s in report.sections:
        _lines = [ln.strip() for ln in (_s.body_markdown or "").split("\n") if ln.strip()]
        if not _lines:
            continue
        if _s.name is not SectionName.FIGURES and all(
            _CAPTION_LINE.match(ln) for ln in _lines
        ):
            _caption_only.append(_s.name.value)
        if len(_lines) != len({ln.lower() for ln in _lines}):
            _dupe_lines.append(_s.name.value)
    checks.check("no prose section is nothing but figure captions",
                 not _caption_only, str(_caption_only))
    checks.check("no section repeats the same line twice",
                 not _dupe_lines, str(_dupe_lines))

    # A report persisted under an OLDER section set must still load. SectionName
    # is a closed enum, so validating a stored "Conclusions"/"Glossary" row
    # raises -- which turned GET /report into a 500 for every report already in
    # the library the moment those two names were retired. The retired rows are
    # dropped and normalize_sections backfills the new names.
    from deepvision.report.interactive_sections import (
        normalize_sections,
        sections_from_rows,
    )

    _legacy_rows = [
        {"id": "s-old-1", "name": "Overview", "body_markdown": "Still here.",
         "provenance": [], "citations": [], "media": [], "default_open": True},
        {"id": "s-old-2", "name": "Conclusions", "body_markdown": "Retired name.",
         "provenance": [], "citations": [], "media": [], "default_open": False},
        {"id": "s-old-3", "name": "Glossary", "body_markdown": "Also retired.",
         "provenance": [], "citations": [], "media": [], "default_open": False},
    ]
    _migrated = normalize_sections(sections_from_rows(_legacy_rows))
    checks.check("a report persisted under the old section set still loads",
                 [s.name for s in _migrated] == SECTION_ORDER,
                 f"{len(_migrated)} sections")
    checks.check("its surviving section keeps its body",
                 (_migrated[SECTION_ORDER.index(SectionName.OVERVIEW)].body_markdown
                  == "Still here."))
    checks.check("retired section names are dropped, not rendered",
                 all(s.name.value not in ("Conclusions", "Glossary") for s in _migrated))

    _glance = report.section(SectionName.AT_A_GLANCE)
    _glance_body = (_glance.body_markdown if _glance else "") or ""
    _missing_fields = [lbl for lbl, _ in AT_A_GLANCE_FIELDS if f"**{lbl}**" not in _glance_body]
    checks.check("At a Glance keeps all five fixed fields even fully degraded",
                 not _missing_fields, f"missing={_missing_fields}")

    # ---- Stage E3: the two new deterministic helpers ----------------------
    # Both are pure functions, so they are checked against the exact strings
    # that caused the original bug rather than against the fixture PDF.
    print("\n[5c] Chunk hygiene + figure linking")
    from deepvision.models import Citation, MediaRef, Provenance, Section
    from deepvision.rag.chunk_quality import is_boilerplate, is_caption
    from deepvision.report.figure_links import attach_referenced_figures

    checks.check("is_caption catches a bare figure caption",
                 is_caption("Figure 6:The horizontal profile of ROI"))
    checks.check(
        "is_caption spares prose that merely mentions a figure",
        not is_caption(
            "As Figure 6 shows, the horizontal profile of the region of interest "
            "peaks near the nose bridge, and the tracker uses that peak to seed "
            "the search window for the next frame, which is what keeps the "
            "estimate stable across a moving head."
        ),
    )
    checks.check(
        "is_boilerplate catches a journal running header",
        is_boilerplate(
            "(IJCSIS) International Journal of Computer Science and Information "
            "Security, Vol. 7, No. 1, 2010"
        ),
    )
    checks.check(
        "is_boilerplate spares real paper prose",
        not is_boilerplate(
            "We propose a method combining feature-based and image-based "
            "approaches to detect the point between the eyes using a "
            "six-segmented rectangular filter."
        ),
    )

    # Sentence-level caption rejection + blank-target eligibility. These guard
    # study-card quality, and every string below is one that actually shipped
    # in a real deck or quiz built from this library:
    #   - "How does the paper characterize **Figure 5: Many of the attention**?"
    #   - "Fill in the blank: Table ____: Variations on the Transformer"
    #   - "...factorization tricks [____] and conditional computation"  (a
    #     bibliography index)
    #   - "...development set, newstest____"  (a year inside a dataset name)
    # Captions reach the generators from *persisted* reports written before the
    # report-side filters landed, so filtering at generation time is the only
    # thing that protects a library that already exists.
    from deepvision.agents.flashcard_agent import (
        _blank_number,
        _is_usable,
        _strip_markers,
        _mentions_value,
    )
    from deepvision.agents.quiz_agent import _as_predicate
    from deepvision.rag.chunk_quality import is_caption_sentence

    checks.check(
        "a long figure caption is still a caption at sentence level",
        is_caption_sentence(
            "Figure 5: Many of the attention heads exhibit behaviour that seems "
            "related to the structure of the sentence. We give two such examples "
            "above, from two different heads from the encoder self-attention."
        ),
    )
    checks.check(
        "a caption wrapped in markdown bold is still caught",
        is_caption_sentence("**Table 3: Variations on the Transformer**"),
    )
    checks.check(
        "prose that merely mentions a figure is not a caption sentence",
        not is_caption_sentence("As Figure 6 shows, the profile peaks near the nose."),
    )
    checks.check(
        "a caption can never become a flashcard side",
        not _is_usable("Figure 1: The Transformer - model architecture."),
    )
    checks.check(
        "real prose is still usable as a flashcard side",
        _is_usable("The encoder is composed of a stack of N = 6 identical layers."),
    )
    for _bad, _why in [
        ("Recent work improved efficiency through factorization tricks [21].",
         "a citation index is never the blank"),
        ("Efforts continued on encoder-decoder architectures [38, 24, 15].",
         "a grouped citation is never the blank"),
        ("All metrics are on the development set, newstest2013.",
         "a number glued inside a word is never the blank"),
        ("The architecture follows a specific design, as shown in Figure 1.",
         "a figure ordinal is never the blank"),
        ("The installation diagram of POLAR on TG-2 is shown here.",
         "a hyphen-joined name keeps its number"),
    ]:
        checks.check(_why, _blank_number(_strip_markers(_bad)) is None, _bad)
    _good = _blank_number("Our model achieves 28.4 BLEU on the translation task.")
    checks.check("a real reported quantity is still blanked",
                 _good is not None and _good[1] == "28.4", str(_good))
    checks.check(
        "polish may not drop a cloze answer",
        not _mentions_value("Recurrent models build hidden states in order.", "1"),
    )
    checks.check(
        "polish that keeps the number is accepted",
        _mentions_value("It reaches a BLEU score of 28.4 overall.", "28.4"),
    )
    from deepvision.agents.flashcard_agent import _is_real_subject

    checks.check(
        "a 'term' trailing off into a bare caption label is not a subject",
        not _is_real_subject("Scaled Dot-Product Attention Multi-Head Attention Figure"),
    )
    checks.check(
        "a genuine term is still a subject",
        _is_real_subject("Multi-Head Attention"),
    )
    checks.check(
        "a footnote URL can never become a card",
        not _is_usable("1http://www.collectspace.com/images/news-091516d-lg.jpg 2"),
    )
    checks.check(
        "a true/false prompt never reads 'refers to this is a'",
        _as_predicate("This is a technique that weighs inputs.")
        == "a technique that weighs inputs.",
    )

    # A quiz must not ask about the same passage twice while unused material
    # remains. A real 9-question quiz asked three definitions twice each (short
    # answer, then multiple choice on identical text) and carried about five
    # distinct subjects; the anchor cap allowed it because it counts terms, not
    # sources. Reuse is now deferred to the end and only spent to avoid coming
    # up short -- so the count is preserved and the redundancy is not.
    from deepvision.agents.quiz_agent import (
        _best_matching_citation,
        _select_drafts,
        _DraftPool,
        _draft,
        SourceItem,
    )
    from deepvision.models.study import Difficulty, QuestionKind

    _srcs = [
        SourceItem(id=f"s{i}", section=SectionName.KEY_CONCEPTS, page=i, chunk_id=None,
                   snippet=f"snippet {i}", term=f"Term {i}", body=f"definition body {i}")
        for i in range(1, 5)
    ]
    _pool = _DraftPool()
    for _kind in (QuestionKind.MULTIPLE_CHOICE, QuestionKind.TRUE_FALSE):
        for _src in _srcs:
            _pool.add(_kind, Difficulty.RECALL, _draft(
                family="definition", kind=_kind, difficulty=Difficulty.RECALL,
                prompt=f"{_kind.value} about {_src.term}?", source=_src,
                explanation="because.", correct_answer_text=_src.term,
            ))
    _sel = _select_drafts(_pool, count=4, kinds=[QuestionKind.MULTIPLE_CHOICE,
                          QuestionKind.TRUE_FALSE], difficulties=[Difficulty.RECALL])
    _used = [d["source_id"] for d in _sel]
    checks.check("a quiz spends each source once before reusing any",
                 len(_sel) == 4 and len(set(_used)) == 4, str(_used))
    # ...but never at the cost of a short quiz: 6 wanted, only 4 sources.
    _pool2 = _DraftPool()
    for _kind in (QuestionKind.MULTIPLE_CHOICE, QuestionKind.TRUE_FALSE):
        for _src in _srcs:
            _pool2.add(_kind, Difficulty.RECALL, _draft(
                family="definition", kind=_kind, difficulty=Difficulty.RECALL,
                prompt=f"{_kind.value} about {_src.term}?", source=_src,
                explanation="because.", correct_answer_text=_src.term,
            ))
    _sel2 = _select_drafts(_pool2, count=6, kinds=[QuestionKind.MULTIPLE_CHOICE,
                           QuestionKind.TRUE_FALSE], difficulties=[Difficulty.RECALL])
    checks.check("source dedupe never shortens a quiz", len(_sel2) == 6, str(len(_sel2)))

    # No single question *shape* may dominate. A real 10-question quiz came back
    # with five "which section is this from?" questions in four different
    # wordings -- varied phrasing on a monotonous quiz is still monotonous.
    _pool3 = _DraftPool()
    _many = [
        SourceItem(id=f"n{i}", section=SectionName.METHODS, page=i, chunk_id=f"ch{i}",
                   snippet=f"snip {i}", term=None, body=f"claim body number {i}")
        for i in range(1, 13)
    ]
    # Disjoint sources per family, so this isolates the family cap rather than
    # racing it against the one-question-per-source rule.
    for _i, _src in enumerate(_many[:6]):
        _pool3.add(QuestionKind.MULTIPLE_CHOICE, Difficulty.RECALL, _draft(
            family="locator", kind=QuestionKind.MULTIPLE_CHOICE,
            difficulty=Difficulty.RECALL,
            prompt=f"Which section is passage {_i} from?", source=_src,
            explanation="because it is.", correct_answer_text=f"sec{_i}"))
    for _i, _src in enumerate(_many[6:]):
        _pool3.add(QuestionKind.TRUE_FALSE, Difficulty.RECALL, _draft(
            family="numeric", kind=QuestionKind.TRUE_FALSE, difficulty=Difficulty.RECALL,
            prompt=f"True or false: value {_i} is reported.", source=_src,
            explanation="because it is.", correct_answer_text="true",
            # Distinct anchors: identical ones would hit the per-term cap of 2
            # and starve this family, masking whether the family cap works.
            anchor=f"value-{_i}"))
    _sel3 = _select_drafts(_pool3, count=9,
                           kinds=[QuestionKind.MULTIPLE_CHOICE, QuestionKind.TRUE_FALSE],
                           difficulties=[Difficulty.RECALL])
    import collections as _c
    _fams = _c.Counter(d.get("_family") for d in _sel3)
    # Two families present, nine questions -> fair share is 5 each.
    checks.check(
        "no one question shape may dominate a quiz",
        len(_sel3) == 9 and max(_fams.values()) <= 5,
        f"{dict(_fams)} of {len(_sel3)}",
    )

    # ---- Stem variety, and the SRS invariant that constrains it -----------
    # Measured on five real quizzes and four real decks: the quiz used 9 of its
    # 12 stems, its top three covered 61% of all questions, and "True or false:
    # in this paper, **X** refers to ..." alone was 30% of *every* quiz. Widening
    # the pools is easy; doing it without re-keying existing cards is the part
    # that needs guarding, because `content_key` hashes the front's wording.
    from deepvision.agents.flashcard_agent import (
        _STEM_FAMILIES,
        _render_stem,
        _widen_new_card_phrasing,
        _Candidate,
        content_key_for,
    )
    from deepvision.agents.quiz_agent import (
        _stem_rotate,
        _TF_DEFINITION_STEMS,
        _MC_DEFINITION_STEMS,
    )
    from deepvision.models.study import FlashcardOrigin

    for _fam, (_legacy, _extra) in _STEM_FAMILIES.items():
        checks.check(f"stem family '{_fam}' actually widened",
                     len(_extra) > 0 and len(_legacy) + len(_extra) >= 7,
                     f"{len(_legacy)} -> {len(_legacy) + len(_extra)}")
        checks.check(f"stem family '{_fam}' has no duplicate phrasings",
                     len(set(_legacy + _extra)) == len(_legacy) + len(_extra))

    # THE invariant: a card the deck already has keeps its exact stored wording,
    # so its content_key -- and its SM-2 schedule -- cannot move.
    def _mk() -> _Candidate:
        return _Candidate(front=_render_stem("term", "Attention", {"term": "Attention"},
                                             widened=False),
                          back="A mechanism that weighs input positions.",
                          origin=FlashcardOrigin.KEY_CONCEPT,
                          stem=("term", "Attention", {"term": "Attention"}))
    _known = _mk()
    _legacy_front = _known.front
    _key = content_key_for("p1", _known.origin, _legacy_front)
    _widen_new_card_phrasing([_known], "p1", frozenset({_key}))
    checks.check(
        "a card already in the deck is never re-worded (SRS schedule safe)",
        _known.front == _legacy_front
        and content_key_for("p1", _known.origin, _known.front) == _key,
        _known.front[:60],
    )
    # "BLEU" is picked deliberately: it hashes into the *extra* stems, so this
    # fails if widening silently stops happening. A seed that happens to land
    # on a legacy stem would pass either way and prove nothing.
    _fresh = _Candidate(
        front=_render_stem("term", "BLEU", {"term": "BLEU"}, widened=False),
        back="The machine-translation score the paper reports.",
        origin=FlashcardOrigin.KEY_CONCEPT,
        stem=("term", "BLEU", {"term": "BLEU"}),
    )
    _fresh_legacy = _fresh.front
    _widen_new_card_phrasing([_fresh], "p1", frozenset())
    checks.check(
        "a card the deck has never seen reaches a genuinely new stem",
        _fresh.front != _fresh_legacy
        and _fresh.front in {s.format(term="BLEU") for s in _STEM_FAMILIES["term"][1]},
        _fresh.front[:60],
    )
    _a, _b = _mk(), _mk()
    _widen_new_card_phrasing([_a], "p1", frozenset())
    _widen_new_card_phrasing([_b], "p1", frozenset())
    checks.check("stem choice is deterministic, never random", _a.front == _b.front)

    # The quiz rotates instead of hashing, so a run of same-kind questions
    # exhausts the pool before any stem comes back. Independent hashing shipped
    # the same MC stem twice inside one ten-question quiz.
    _rotated = [_stem_rotate(i, _TF_DEFINITION_STEMS) for i in range(len(_TF_DEFINITION_STEMS))]
    checks.check("quiz stems rotate before repeating",
                 len(set(_rotated)) == len(_TF_DEFINITION_STEMS))
    checks.check("quiz rotation is deterministic",
                 _stem_rotate(3, _MC_DEFINITION_STEMS) == _stem_rotate(3, _MC_DEFINITION_STEMS))
    checks.check("the one-stem true/false monotony is gone",
                 len(_TF_DEFINITION_STEMS) >= 4, str(len(_TF_DEFINITION_STEMS)))

    # A definition that opens with a scene-setting clause used to defeat the
    # ^-anchored copular pattern and ship "**QKT** refers to in the context of
    # attention mechanisms, ___ refers to a mathematical operation ...".
    from deepvision.agents.quiz_agent import _as_predicate as _pred
    checks.check(
        "a prepositional preamble no longer hides the definition's subject",
        _pred("In the context of attention mechanisms, ___ refers to a matrix product.")
        == "a matrix product.",
        _pred("In the context of attention mechanisms, ___ refers to a matrix product."),
    )
    checks.check(
        "a preamble that is real subject matter is left alone",
        _pred("Inference latency is the time taken to produce one token.").startswith(
            "Inference latency"
        ),
    )

    # The whole-paper pool must be sampled, not prefixed. Two of ten chunks were
    # filling the entire 30-claim budget, so 80% of the pool never reached a
    # quiz -- silently undoing most of what `source_pool` exists to provide.
    from deepvision.agents.quiz_agent import (
        collect_sources as _collect,
        _MAX_CLAIMS_PER_POOL_CHUNK as _PER_CHUNK,
        _ORDINAL_MARKER_RE as _ORD,
    )

    class _FakeChunk:
        def __init__(self, cid, text): self.id, self.text, self.page = cid, text, 1

    # Distinct text per chunk: `emit` de-duplicates by body, so two identical
    # chunks would collapse and the spread could not be observed.
    def _long(tag: str) -> str:
        return " ".join(
            f"The {tag} subsystem records measurement number {i} during the run."
            for i in range(12)
        )

    _srcs = _collect(
        [], extra_chunks=[_FakeChunk("c1", _long("optical")), _FakeChunk("c2", _long("thermal"))]
    )
    _per = {}
    for _s in _srcs:
        _per[_s.chunk_id] = _per.get(_s.chunk_id, 0) + 1
    checks.check(
        "no single pool chunk can monopolise the claim budget",
        _per and max(_per.values()) <= _PER_CHUNK,
        str(_per),
    )
    checks.check("every pool chunk gets represented", len(_per) == 2, str(sorted(_per)))

    # Sequence-marked sentences are what `_ordering_questions` needs, and a
    # blind prefix surfaced almost none of them -- that shape had never fired.
    _mixed = (
        "The apparatus was assembled in a clean room over several weeks of work. "
        "The calibration constants were taken from the manufacturer's datasheet. "
        "The enclosure is machined from a single aluminium billet for rigidity. "
        "First, we reject any acquired data that lacks a valid phase reference. "
        "Then, we fold the remaining events on the pulsar's rotational period."
    )
    _ord_srcs = _collect([], extra_chunks=[_FakeChunk("c9", _mixed)])
    _kept_ord = [s for s in _ord_srcs if _ORD.match(s.body)]
    checks.check(
        "sequence-marked sentences survive the per-chunk cap",
        len(_kept_ord) >= 2,
        f"{len(_kept_ord)} of {len(_ord_srcs)} kept",
    )
    checks.check(
        "and they keep document order (an ordering question's ground truth)",
        len(_kept_ord) < 2
        or _kept_ord[0].body.lower().startswith("first"),
        _kept_ord[0].body[:40] if _kept_ord else "",
    )

    # ---- Session scheduler ------------------------------------------------
    # SM-2 had ZERO assertions here, which is partly how it stayed unverified
    # for so long (HANDOFF §3). Its replacement gets real ones. Every rule
    # below is quoted from describe_scheduler_contract().
    print("\n[5e] Session scheduling (no dates, nothing persisted)")
    from datetime import datetime, timedelta, timezone as _tz
    from deepvision.models.study import (
        SESSION_REQUEUE_GAPS,
        Rating as _Rating,
        describe_scheduler_contract,
        rating_recalled,
    )
    from deepvision.study.session_scheduler import (
        insertion_index,
        local_day_bounds_utc,
        queue_sort_key,
        requeue_gap,
        strength_from_ratings,
    )

    checks.check("exactly three ratings, and they are the wire contract",
                 [r.value for r in _Rating] == ["again", "almost", "got_it"],
                 str([r.value for r in _Rating]))
    checks.check("'got it' has no gap — it retires from the session",
                 requeue_gap(_Rating.GOT_IT) is None)
    checks.check("'again' comes back sooner than 'almost'",
                 0 < requeue_gap(_Rating.AGAIN) < requeue_gap(_Rating.ALMOST),
                 f"{requeue_gap(_Rating.AGAIN)} < {requeue_gap(_Rating.ALMOST)}")
    checks.check("the gaps have exactly one home",
                 set(SESSION_REQUEUE_GAPS) == {_Rating.AGAIN, _Rating.ALMOST})

    # Placement, including the edge case that decides whether a failed card is
    # ever seen again: fewer cards left than the gap.
    checks.check("a rated card is reinserted after the gap",
                 insertion_index(0, 20, 6) == 7, str(insertion_index(0, 20, 6)))
    checks.check("a short tail clamps to the END, never drops the card",
                 insertion_index(0, 2, 6) == 3, str(insertion_index(0, 2, 6)))
    checks.check("retirement returns no index", insertion_index(0, 20, None) is None)

    # Strength is DERIVED. These are the only thing carrying a rating's
    # meaning into a later session, so they are worth being strict about.
    checks.check("strength counts consecutive recalls from the newest",
                 strength_from_ratings(["got_it", "almost", "again"]) == 2)
    checks.check("a fresh 'again' resets strength to zero",
                 strength_from_ratings(["again", "got_it", "got_it"]) == 0)
    checks.check("a never-reviewed card has strength zero",
                 strength_from_ratings([]) == 0)
    checks.check("legacy SM-2 ratings still count as recall (append-only log)",
                 strength_from_ratings(["good", "easy", "hard"]) == 3)
    checks.check("legacy 'again' still counts as a lapse",
                 not rating_recalled("again") and rating_recalled("good"))

    # Queue order: weakest first, then longest unseen. New material must lead
    # rather than trail behind cards the reader has already failed once.
    _mk = lambda st, ts, cid: queue_sort_key(st, ts, cid)
    _older = datetime(2026, 8, 1, 12, 0, 0)
    _newer = datetime(2026, 8, 15, 12, 0, 0)
    checks.check("weaker cards sort before stronger ones",
                 _mk(0, _newer, "b") < _mk(3, _older, "a"))
    checks.check("within a strength band, longest-unseen comes first",
                 _mk(1, _older, "b") < _mk(1, _newer, "a"))
    checks.check("a never-reviewed card leads its strength band",
                 _mk(0, None, "z") < _mk(0, _older, "a"))
    checks.check("ordering is deterministic on ties (stable card-id tail)",
                 _mk(1, _older, "a") < _mk(1, _older, "b"))

    # The local-day boundary. Getting this wrong resets the day's count in the
    # middle of an afternoon session for anyone west of Greenwich.
    _start, _end = local_day_bounds_utc(datetime(2026, 8, 16, 21, 30, 0))
    checks.check("the local day is exactly 24h wide",
                 (_end - _start) == timedelta(days=1), f"{_start} .. {_end}")
    checks.check("day bounds are naive UTC, matching reviewed_at's convention",
                 _start.tzinfo is None and _end.tzinfo is None)
    _local_midnight = (
        _start.replace(tzinfo=_tz.utc).astimezone().replace(tzinfo=None)
    )
    checks.check("the boundary is LOCAL midnight, not UTC midnight",
                 (_local_midnight.hour, _local_midnight.minute) == (0, 0),
                 f"local start = {_local_midnight}")

    # The contract is the single home; the engine must actually be described by it.
    _contract = describe_scheduler_contract()
    for _needle in ("again", "almost", "got_it", "LOCAL", "append-only"):
        checks.check(f"the scheduler contract documents {_needle!r}",
                     _needle in _contract)
    # Scoped to the LIVE rules (everything before section D). Section D names
    # the dead columns on purpose, and the contract says "NO dates and NO
    # intervals" — so a whole-document substring test would fail on its own
    # documentation. What must hold is that no SM-2 mechanic appears in the
    # part that tells you how to schedule a card.
    _live_rules = _contract.split("D. Logging")[0].lower()
    _sm2_terms = ["ease_factor", "ease factor", "learning step", "graduating",
                  "repetitions", "due_at", "lapse", "interval_days"]
    _leaked = [t for t in _sm2_terms if t in _live_rules]
    checks.check("no SM-2 mechanic survives in the live rules", not _leaked, str(_leaked))
    checks.check("the contract still names the dead columns it left behind",
                 "ease_factor" in _contract and "vestigial" not in _live_rules)
    checks.check("the contract states that nothing is persisted",
                 "nothing is persisted" in _contract.lower()
                 or "no table records scheduling state" in _contract.lower())

    # A citation is evidence, not decoration. Falling back to the section's
    # first citation made every question on one paper cite the same Figure 4
    # caption on p.14, whatever it was actually about.
    def _cit(marker: int, page: int, snippet: str) -> Citation:
        return Citation(id=f"cit{marker}", marker=marker, source=Provenance.TEXT,
                        page=page, page_label=str(page), snippet=snippet,
                        chunk_id=f"c{marker}")

    _cits = [
        _cit(1, 14, "Figure 4: Two attention heads involved in anaphora resolution."),
        _cit(2, 3, "The encoder is composed of a stack of six identical layers."),
    ]
    _matched = _best_matching_citation(
        "The encoder is composed of a stack of six identical layers.", _cits
    )
    checks.check(
        "a line is cited to the passage it actually overlaps",
        _matched is not None and _matched.page == 3,
        str(_matched.page if _matched else None),
    )
    checks.check(
        "an unrelated line gets no citation rather than a wrong one",
        _best_matching_citation(
            "Training used the Adam optimizer with a warmup schedule.", _cits
        ) is None,
    )

    _media_pool = [
        MediaRef(id="m1", kind="figure", label="Figure 6", caption="",
                 provenance=Provenance.VISION),
        MediaRef(id="m2", kind="table", label="Table 4", caption="",
                 provenance=Provenance.OCR),
        MediaRef(id="m3", kind="figure", label="Figure (p3)", caption="",
                 provenance=Provenance.VISION),
    ]
    _probe = Section(
        id="probe-1", name=SectionName.METHODS,
        body_markdown="The pipeline is shown in Fig. 6 and measured in Table 4.",
    )
    _linked = attach_referenced_figures([_probe], _media_pool)
    _linked_ids = [m.id for m in _linked[0].media]
    checks.check("prose that says 'Fig. 6' gets the real Figure 6 attached",
                 _linked_ids == ["m1", "m2"], str(_linked_ids))

    _figures_probe = Section(
        id="probe-2", name=SectionName.FIGURES,
        body_markdown="Figure 6 and Table 4.", media=list(_media_pool),
    )
    _relinked = attach_referenced_figures([_figures_probe], _media_pool)
    checks.check("the Figures section is never double-stuffed",
                 len(_relinked[0].media) == len(_media_pool),
                 f"{len(_relinked[0].media)} refs")

    # ---- Stage E4: the fallback tally must count EVERY fallback ------------
    # This is the mechanism the whole "done != model-written" contract rests on
    # an empty string instead of raising -- LocalLLM logs "Ollama unreachable;
    # returning empty completion" -- so counting only raised exceptions made a
    # run with Ollama stopped report ZERO fallbacks while every call had fallen
    # back, and every Section.degraded flag said "model-written" about verbatim
    # PDF extract. Both paths must count.
    print("\n[5d] Degradation is actually counted")
    from deepvision.agents.base import (
        complete_with_fallback,
        llm_fallback_count,
        repair_citation_markers,
        reset_llm_fallbacks,
    )

    # Marker repair: the reader resolves ONLY `[n]`, and llama3.1:8b really does
    # emit "[n] 3" and "**(1)**" -- a marker in any other shape silently stops
    # opening its citation popover. Repair must not invent markers either.
    checks.check("a leaked '[n] 3' marker is repaired to [3]",
                 repair_citation_markers("The nose is brighter. [n] 3", 5)
                 == "The nose is brighter. [3]")
    checks.check("a re-styled '**(1)**' marker is repaired to [1]",
                 repair_citation_markers("...lighting conditions **(1)**.", 5)
                 == "...lighting conditions [1].")
    checks.check("a bare equation reference '(5)' is left alone",
                 repair_citation_markers("substituting into (5) gives", 5)
                 == "substituting into (5) gives")
    checks.check("an out-of-range number is never turned into a citation",
                 repair_citation_markers("out of range **(97)**", 5)
                 == "out of range **(97)**")

    class _EmptyLLM:
        """An adapter that swallows its own transport error, as the real ones do."""

        def complete(self, messages, temperature=0.0, max_tokens=None):
            return ""

    class _RaisingLLM:
        def complete(self, messages, temperature=0.0, max_tokens=None):
            raise RuntimeError("provider exploded")

    _DRAFT = "A grounded extractive draft that must ship unchanged. [1]"
    for _label, _llm in (("empty completion", _EmptyLLM()), ("raised error", _RaisingLLM())):
        reset_llm_fallbacks()
        _out = complete_with_fallback(_llm, "system", _DRAFT)
        checks.check(f"a {_label} returns the draft verbatim", _out == _DRAFT)
        checks.check(f"a {_label} is counted as a fallback",
                     llm_fallback_count() == 1, f"count={llm_fallback_count()}")
    reset_llm_fallbacks()

    # ---- Stage E5: chat intent routing + citation formatting --------------
    # The chat used to send every question through retrieval, including ones
    # the paper's body text cannot possibly answer. Asked for an APA citation
    # it replied "based on the [n] markers, it appears that the citations are
    # as follows: [1], [2], [3]". Routing and deterministic formatting are what
    # fixed that, and both are pure functions, so both are checked directly.
    print("\n[5e] Chat routing + citation styles")
    from datetime import date as _date

    from deepvision.agents.chat_intent import Intent, classify
    from deepvision.models.paper import PaperMeta as _PaperMeta
    from deepvision.report.citation_styles import (
        CITATION_STYLE_LABELS,
        CitationStyle,
        detect_styles,
        format_citation,
    )

    _ROUTING_CASES = [
        ("give me the citation for the pdf in APA style", Intent.CITATION),
        ("cite this in bibtex", Intent.CITATION),
        ("how do I cite this", Intent.CITATION),
        # A style name is an acronym too -- these must NOT reach the formatter.
        ("what is the MLA dataset used for", Intent.CONTENT),
        ("what IEEE standard does it follow", Intent.CONTENT),
        # Asking whom the paper cites is a content question about its
        # reference list, NOT a request to cite the paper.
        ("which papers does this cite?", Intent.CONTENT),
        ("who wrote this?", Intent.METADATA),
        ("when was this published", Intent.METADATA),
        ("how many figures are in this paper", Intent.STRUCTURE),
        ("What are the limitations of this work?", Intent.CONTENT),
        ("what does the SSR filter do", Intent.CONTENT),
    ]
    _misrouted = [
        (q, classify(q).value, want.value)
        for q, want in _ROUTING_CASES
        if classify(q) is not want
    ]
    checks.check("every chat question routes to the right intent",
                 not _misrouted, str(_misrouted))

    checks.check("every citation style has a display label",
                 all(s in CITATION_STYLE_LABELS for s in CitationStyle))
    checks.check("detect_styles finds every style a question names",
                 detect_styles("give me apa and bibtex")
                 == [CitationStyle.APA, CitationStyle.BIBTEX],
                 str([s.value for s in detect_styles("give me apa and bibtex")]))
    checks.check("detect_styles stays silent on an unrelated question",
                 detect_styles("what are the limitations of this work") == [])

    _cite_meta = _PaperMeta(
        id="1002-2191", arxiv_id="1002.2191", arxiv_label="arXiv:1002.2191",
        title="Vision Based Game Development", authors=["S. Sumathi", "S. K. Srivatsa"],
        published=_date(2010, 2, 10),
    )
    _apa = format_citation(_cite_meta, CitationStyle.APA)
    checks.check("APA puts surname first, initials after, with the year",
                 _apa.startswith("Sumathi, S.") and "(2010)" in _apa, _apa)

    # An uploaded PDF's arXiv fields are placeholders, not identifiers. Printing
    # "arXiv:upload:..." or linking to arxiv.org for one would be a fabricated
    # citation -- the single worst thing this module could do.
    _upload_meta = _PaperMeta(
        id="upload-abc", arxiv_id="upload:upload-abc", arxiv_label="Uploaded PDF",
        title="A Local Thesis", authors=["Jane Q. Doe"],
    )
    _upload_all = [format_citation(_upload_meta, s) for s in CitationStyle]
    checks.check("an uploaded PDF never gets a fabricated arXiv id or link",
                 not any("arxiv.org" in c.lower() or "upload:" in c.lower()
                         for c in _upload_all))
    checks.check("a missing date never produces a doubled 'n.d..'",
                 not any("n.d.." in c for c in _upload_all))
    checks.check("BibTeX omits the year rather than emitting year = {n.d.}",
                 "n.d." not in format_citation(_upload_meta, CitationStyle.BIBTEX))

    # ---- Stage F: FastAPI in-process --------------------------------------
    print("\n[6] FastAPI app (in-process TestClient)")
    from fastapi.testclient import TestClient

    from deepvision.api.main import app

    client = TestClient(app)

    r = client.get("/api/health")
    checks.check("GET /health -> 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        body = r.json()
        checks.check("health payload shape",
                     {"status", "version", "db_ok"} <= set(body), str(body))

    r = client.get(f"/api/report/{paper_id}")
    checks.check("GET /report/{id} -> 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        body = r.json()
        payload_names = [s.get("name") for s in body.get("sections", [])]
        expected_names = [n.value for n in SECTION_ORDER]
        checks.check(f"report payload has all {len(SECTION_ORDER)} sections in order",
                     payload_names == expected_names,
                     ", ".join(str(n) for n in payload_names))
        checks.check("report payload has stats + paper",
                     "stats" in body and body.get("paper") is not None)

    r = client.post("/api/chat", json={"paper_id": paper_id,
                                       "message": "What abilities emerge at scale?",
                                       "top_k": 6})
    checks.check("POST /chat -> 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        body = r.json()
        msg = body.get("message", {})
        checks.check("chat response shape",
                     "session_id" in body and msg.get("role") == "assistant"
                     and isinstance(msg.get("text"), str) and msg.get("text") != "",
                     f"text_len={len(msg.get('text', ''))}, "
                     f"retrieved={len(body.get('retrieved_chunk_ids', []))}")

    r = client.get("/api/settings")
    checks.check("GET /settings -> 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        body = r.json()
        s = body.get("settings", {})
        checks.check("settings payload shape + keys redacted",
                     {"needs_keys", "keys_present", "vision_available"} <= set(body)
                     and s.get("keys", {}).get("llm_api_key") is None,
                     f"needs_keys={body.get('needs_keys')}")

    # ---- Bonus: papers list + media metadata ------------------------------
    r = client.get("/api/papers")
    checks.check("GET /papers -> 200 and lists the paper",
                 r.status_code == 200 and any(p["id"] == paper_id
                                              for p in r.json().get("papers", [])),
                 f"status={r.status_code}")

    # The two "due" figures must mean the same thing. They briefly did not:
    # the queue's `total_due` kept its SM-2 meaning ("due_at <= now") after the
    # filter it named was deleted, so the header said 138 while the panel under
    # it said 142 — both labelled "due".
    _ov = client.get("/api/study/overview")
    checks.check("GET /study/overview -> 200", _ov.status_code == 200, str(_ov.status_code))
    _ovj = _ov.json()
    checks.check("overview exposes a separate cards-due and quizzes-due",
                 "due" in _ovj and "quizzes_due" in _ovj,
                 str(sorted(k for k in _ovj if "due" in k)))
    checks.check("the retired date-based tiles are gone",
                 "due_today" not in _ovj and "due_next_7_days" not in _ovj)
    _dq = client.get("/api/study/due?limit=5").json()
    checks.check("the queue's total_due agrees with the header's due",
                 _dq["total_due"] == _ovj["due"],
                 f'panel={_dq["total_due"]} header={_ovj["due"]}')

    # Quizzes-due is a BACKLOG, not a daily figure — it must not use the
    # local-day reset that cards use, or it would tell you to retake
    # everything each morning.
    from sqlmodel import func, select as _select
    from deepvision.db import session_scope
    from deepvision.db.schema import QuizAttemptRow, QuizRow

    with session_scope() as _s:
        _qtotal = int(_s.exec(_select(func.count(QuizRow.id))).one() or 0)
        _qdone = int(
            _s.exec(_select(func.count(func.distinct(QuizAttemptRow.quiz_id)))).one() or 0
        )
    checks.check("quizzes due = quizzes never attempted",
                 _ovj["quizzes_due"] == max(0, _qtotal - _qdone),
                 f'{_ovj["quizzes_due"]} == {_qtotal} - {_qdone}')
    checks.check("quizzes due never exceeds the quiz count",
                 _ovj["quizzes_due"] <= _ovj["quiz_count"])

    return 0 if checks.summary() else 1


if __name__ == "__main__":
    keep = "--keep" in sys.argv
    code = 1
    try:
        code = main()
    except Exception:  # pragma: no cover - surface any unexpected crash clearly
        print("\n\033[31mUNEXPECTED ERROR\033[0m during smoke check:\n")
        traceback.print_exc()
        code = 2
    finally:
        if keep:
            print(f"\n(sandbox kept at {_SANDBOX})")
        else:
            shutil.rmtree(_SANDBOX, ignore_errors=True)
    sys.exit(code)
