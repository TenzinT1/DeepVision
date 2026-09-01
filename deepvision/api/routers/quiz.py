"""Quiz router — generation, hidden-answer delivery, server-side grading.

Owned by: **QUIZ builder**. Do not edit ``study.py`` or ``flashcards.py``, and
never touch ``routers/__init__.py`` — this module is already registered.

Routes (all mounted under ``/api/quiz``):

- ``POST /api/quiz/generate`` → :class:`QuizGenerateResponse` (**202**, always).
  Runs the LLM ensemble, so it is job-based (``IngestJob`` + ``GET /jobs/{id}``
  + frontend ``pollJob``), following ``api/chapter_job_runner.py``. There is no
  cache short-circuit: a new quiz is a new row, so past attempt history stays
  meaningful.
- ``GET /api/quiz/paper/{paper_id}`` → :class:`QuizListResponse` (newest first).
- ``GET /api/quiz/{quiz_id}`` → :class:`~deepvision.models.study.QuizPublic` —
  the quiz **with answers stripped**.
- ``POST /api/quiz/{quiz_id}/check`` → :class:`QuizCheckResponse` — grade ONE
  answer and persist **nothing**. This is what powers immediate per-question
  feedback while the user is mid-quiz; see the note below.
- ``POST /api/quiz/{quiz_id}/attempt`` → :class:`~deepvision.models.study.QuizAttempt`
  — graded server-side, synchronously, and **recorded**.
- ``POST /api/quiz/{quiz_id}/retry`` → :class:`~deepvision.models.study.QuizPublic`
  scoped to the misses of ``source_attempt_id`` ("retry only what I missed").
- ``GET /api/quiz/{quiz_id}/attempts`` → :class:`QuizAttemptHistoryResponse`.
- ``GET /api/quiz/attempts/{attempt_id}`` → the full
  :class:`~deepvision.models.study.QuizAttempt`, with explanations.

**The one rule you cannot get wrong: answers are hidden until answered.**

``QuizQuestion`` (and ``QuizQuestionRow``) hold ``correct_option_id``,
``correct_answer_text``, ``accepted_answers`` and ``explanation``. None of them
may appear in any payload sent before the user commits an answer — otherwise the
whole quiz is readable in devtools in one click. :func:`_public_question` builds
:class:`~deepvision.models.study.QuizQuestionPublic` **field by field**; never
``QuizQuestion.model_dump(exclude={...})``, which silently starts leaking on the
day someone adds a field. ``QuizQuestionPublic`` has no field to put an answer
in — that is the safety property, keep it.

The answer travels to the client exactly once: inside
:class:`~deepvision.models.study.GradedAnswer`, after grading — from ``/check``
for the question just answered, or from ``/attempt`` for the whole run.

**Check vs attempt (integration note).** ``/check`` grades and returns;
``/attempt`` grades and *records*. They share one grader
(:func:`~deepvision.study.grading.grade_question`), so live feedback can never
disagree with the final score. The split exists because the runner needs a
server round trip per question — ``QuizQuestionPublic`` has no answer field, by
design — and doing those round trips against ``/attempt`` would write one
``quiz_attempts`` row per question: a ten-question run would land as ten
attempts, nine of them junk partial scores, corrupting attempt history and every
launch card's ``attempt_count``. So the client calls ``/check`` once per
question, then ``/attempt`` exactly once at the end with the full answer list.

Grading (synchronous, deterministic, **no model calls**) lives in
``deepvision/study/grading.py`` and is applied here via
:func:`~deepvision.study.grading.grade_answers`:

- ``multiple_choice`` / ``true_false`` — ``selected_option_id == correct_option_id``.
- ``short_answer`` — normalized exact match against ``accepted_answers``, else
  content-token overlap >= ``SHORT_ANSWER_TOKEN_OVERLAP_THRESHOLD`` (0.6).
- An omitted or empty answer grades as incorrect — not an error. A user may
  submit a partially finished quiz.
- ``score = correct / total`` where ``total == len(question_ids)``.
- For ``mode == 'retry_missed'``: the scope **is** the source attempt's
  ``missed_question_ids``, and every submitted ``question_id`` must be one of
  them (400 otherwise). Both halves matter — fixing the denominator to the
  recorded misses is what stops the mode from being used to fake a high score on
  a cherry-picked subset.

Generation also (1) sweeps the paper's whole chunk pool
(:func:`~deepvision.study.source_pool.pool_for_paper`, via ``_build_pool``) so
the generator sees material the report never quotes, chapter-scoping it with
the same :class:`~deepvision.rag.chapter_scope.PageScopedRetriever` a chapter
*report* uses when ``request.chapter_id`` is set; and (2) passes every prompt
already asked in a previous quiz for this paper (``_previous_prompts``) as
``avoid_prompts``, so regenerating a quiz stops re-asking the same questions.
Both degrade to nothing (empty pool / empty exclusion set) rather than failing
the job — see ``_build_pool``'s docstring.

Generation is queued through :func:`~deepvision.api.study_job_runner.launch_study_job`
— the same shared runner the flashcards router uses, following
``api/chapter_job_runner.py``'s shape: ``stages = []``, ``current_stage = None``,
progress as percent + log, and a single-critical-section in-flight reservation so
two POSTs for the same paper cannot both pay for a full generation. This module
supplies only the work itself, in :func:`_quiz_work`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select

from deepvision.agents.quiz_agent import QuizAgent
from deepvision.api.deps import get_llm, get_settings
from deepvision.api.study_job_runner import (
    JobAbort,
    ProgressFn,
    StudyWork,
    launch_study_job,
)
from deepvision.api.schemas import (
    QuizAttemptHistoryResponse,
    QuizAttemptRequest,
    QuizCheckRequest,
    QuizCheckResponse,
    QuizGenerateRequest,
    QuizGenerateResponse,
    QuizListResponse,
    QuizRetryRequest,
)
from deepvision.db import session_scope
from deepvision.db.schema import (
    PaperRow,
    QuizAnswerRow,
    QuizAttemptRow,
    QuizQuestionRow,
    QuizRow,
)
from deepvision.models import AppSettings, Section
from deepvision.models.chunks import AnyChunk
from deepvision.models.report import SectionName
from deepvision.models.study import (
    AttemptMode,
    Difficulty,
    GradedAnswer,
    QuestionKind,
    Quiz,
    QuizAttempt,
    QuizAttemptSummary,
    QuizCitation,
    QuizOption,
    QuizPublic,
    QuizQuestion,
    QuizQuestionPublic,
    QuizSummary,
)
from deepvision.study.grading import grade_answers, grade_question
from deepvision.utils import get_logger
from deepvision.utils.ids import new_id

__all__ = ["router"]

router = APIRouter(prefix="/quiz", tags=["quiz"])

log = get_logger(__name__)


# ==========================================================================
# Row <-> model mapping
# ==========================================================================
def _question_from_row(row: QuizQuestionRow) -> QuizQuestion:
    """Re-validate a persisted question row — **answer key included**.

    This is the server-side shape. Never hand the result of this function to a
    client except through :func:`_public_question` or a
    :class:`~deepvision.models.study.GradedAnswer`.
    """
    citation: Optional[QuizCitation] = None
    if row.citation_section:
        try:
            citation = QuizCitation(
                section=SectionName(row.citation_section),
                page=row.citation_page,
                chunk_id=row.citation_chunk_id,
                snippet=row.citation_snippet or "",
            )
        except ValueError:
            log.warning(
                "quiz question has an unknown citation section",
                extra={"question_id": row.id, "section": row.citation_section},
            )
    try:
        kind = QuestionKind(row.kind)
    except ValueError:
        kind = QuestionKind.MULTIPLE_CHOICE
    try:
        difficulty = Difficulty(row.difficulty)
    except ValueError:
        difficulty = Difficulty.RECALL

    return QuizQuestion(
        id=row.id,
        quiz_id=row.quiz_id,
        paper_id=row.paper_id,
        index=row.index,
        prompt=row.prompt or "",
        kind=kind,
        difficulty=difficulty,
        options=[
            QuizOption(id=str(o.get("id", "")), text=str(o.get("text", "")))
            for o in (row.options or [])
            if isinstance(o, dict)
        ],
        correct_option_id=row.correct_option_id,
        correct_answer_text=row.correct_answer_text,
        accepted_answers=list(row.accepted_answers or []),
        explanation=row.explanation or "",
        citation=citation,
        created_at=row.created_at,
    )


def _pointer_citation(citation: Optional[QuizCitation]) -> Optional[QuizCitation]:
    """Reduce a citation to a **pointer**: section + page + chunk, no snippet.

    ``QuizQuestionPublic`` carries a citation deliberately, as a "where does
    this come from" hint. But ``QuizCitation.snippet`` is a *verbatim quote of
    the source the question was generated from* — for a ``short_answer``
    question mined out of a Key Concepts definition, the accepted answer is
    typically sitting inside that quote word for word. Shipping it before the
    user commits hands over the answer to anyone who opens devtools, which is
    exactly the property the rest of this module exists to protect. (Verified
    empirically: an integration probe found short-answer keys inside pre-answer
    snippets.)

    So the pre-answer payload keeps the *pointer* — section, page and
    ``chunk_id``, enough for the UI to offer "show me where this is in the
    paper" — and drops the quote. The full snippet travels later, inside
    :class:`~deepvision.models.study.GradedAnswer`, where the answer is already
    revealed anyway. No UI regression: ``CitationChip`` is only ever rendered
    from a ``GradedAnswer``.

    Applied to every kind, not just ``short_answer``: a snippet can just as
    easily single out the right option of a multiple-choice question, and one
    unconditional rule is harder to get wrong later than a per-kind exception.
    """
    if citation is None:
        return None
    return QuizCitation(
        section=citation.section,
        page=citation.page,
        chunk_id=citation.chunk_id,
        snippet="",
    )


def _public_question(question: QuizQuestion) -> QuizQuestionPublic:
    """Project a question to the **hidden-until-answered** wire shape.

    Built field by field on purpose. ``model_dump(exclude={...})`` would look
    equivalent and would start leaking the day a new answer-bearing field is
    added to :class:`~deepvision.models.study.QuizQuestion` — the exclude list
    would simply not know about it. Here, adding a field to ``QuizQuestion``
    changes nothing: it cannot reach the client unless someone explicitly writes
    a line for it below, and ``QuizQuestionPublic`` has no field it could go in.

    The one field that needs more than copying is the citation — see
    :func:`_pointer_citation`.
    """
    return QuizQuestionPublic(
        id=question.id,
        index=question.index,
        prompt=question.prompt,
        kind=question.kind,
        difficulty=question.difficulty,
        options=[QuizOption(id=o.id, text=o.text) for o in question.options],
        citation=_pointer_citation(question.citation),
    )


def _quiz_public(
    quiz_row: QuizRow,
    questions: list[QuizQuestion],
    *,
    paper_title: str,
    mode: AttemptMode = AttemptMode.FULL,
    retry_of_attempt_id: Optional[str] = None,
) -> QuizPublic:
    return QuizPublic(
        id=quiz_row.id,
        paper_id=quiz_row.paper_id,
        paper_title=paper_title,
        title=quiz_row.title or "",
        questions=[_public_question(q) for q in questions],
        difficulty_mix=dict(quiz_row.difficulty_mix or {}),
        kind_mix=dict(quiz_row.kind_mix or {}),
        generated_at=quiz_row.generated_at,
        mode=mode,
        retry_of_attempt_id=retry_of_attempt_id,
    )


def _paper_title(session: Session, paper_id: str) -> str:
    row = session.get(PaperRow, paper_id)
    return (row.title or "") if row is not None else ""


def _load_quiz(session: Session, quiz_id: str) -> tuple[QuizRow, list[QuizQuestion]]:
    """Load a quiz and its questions in quiz order, or raise a 404."""
    quiz_row = session.get(QuizRow, quiz_id)
    if quiz_row is None:
        raise HTTPException(status_code=404, detail=f"quiz not found: {quiz_id}")
    rows = session.exec(
        select(QuizQuestionRow)
        .where(QuizQuestionRow.quiz_id == quiz_id)
        .order_by(QuizQuestionRow.index)
    ).all()
    return quiz_row, [_question_from_row(row) for row in rows]


def _attempt_summary(row: QuizAttemptRow) -> QuizAttemptSummary:
    return QuizAttemptSummary(
        id=row.id,
        quiz_id=row.quiz_id,
        paper_id=row.paper_id,
        mode=_attempt_mode(row.mode),
        total=row.total,
        correct=row.correct,
        score=row.score,
        missed_count=len(row.missed_question_ids or []),
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


def _attempt_mode(value: Optional[str]) -> AttemptMode:
    try:
        return AttemptMode(value or AttemptMode.FULL.value)
    except ValueError:
        return AttemptMode.FULL


# ==========================================================================
# Generation job
# ==========================================================================
# The launcher itself lives in ``api/study_job_runner.py`` — the SAME module the
# flashcards router uses. It was duplicated here while that file was still being
# built by another domain; keeping two copies of a threading.Lock-guarded
# in-flight reservation is how the two halves of Study drift apart. The one
# thing quiz generation needs that a deck does not is to mint ``quiz_id``
# *before* the work starts (the 202 response promises it), which the runner's
# ``payload`` carries: a second concurrent POST gets the winner's job **and**
# the winner's quiz id, so its poll resolves to the quiz actually being written.

#: Job kind — half of the study job runner's in-flight reservation key.
_JOB_KIND = "quiz"


def _load_sections(paper_id: str, chapter_id: Optional[str]) -> list[Section]:
    """Return the report sections the quiz is built from.

    A chapter-scoped quiz reads the **chapter** report, so its questions cannot
    come from pages outside the chapter. If that report has not been generated
    yet the job fails with a message saying so — quietly falling back to the
    whole paper would silently quiz material the user did not ask about.
    """
    if chapter_id:
        from deepvision.report.chapter_report_generator import ChapterReportGenerator

        try:
            report = ChapterReportGenerator().load(paper_id, chapter_id)
        except LookupError as exc:
            raise JobAbort(
                f"no chapter report for {chapter_id!r} yet — generate the chapter "
                "report first, then build a quiz from it"
            ) from exc
        return list(report.sections)

    from deepvision.report.report_generator import DefaultReportGenerator

    try:
        report = DefaultReportGenerator().load(paper_id)
    except LookupError as exc:
        raise JobAbort(
            "this paper has no report yet — open the Report tab to generate one, "
            "then build a quiz from it"
        ) from exc
    return list(report.sections)


def _previous_prompts(session: Session, paper_id: str) -> frozenset[str]:
    """Prompt text of every question in every quiz already generated for
    ``paper_id`` — the exclusion set that keeps a regenerated quiz from asking
    what a past one already asked (``QuizAgent.generate(avoid_prompts=...)``).

    Not chapter-scoped: :class:`~deepvision.db.schema.QuizRow` has no
    ``chapter_id`` column (a paper's quizzes are not partitioned by chapter),
    so this is every prior quiz for the paper regardless of scope.
    """
    rows = session.exec(
        select(QuizQuestionRow.prompt).where(QuizQuestionRow.paper_id == paper_id)
    ).all()
    return frozenset(prompt for prompt in rows if prompt)


def _build_pool(
    paper_id: str, chapter_id: Optional[str], settings: AppSettings
) -> list[AnyChunk]:
    """Build this quiz's whole-paper (or chapter-scoped) source pool.

    Never raises — a failure building the retriever, or resolving the chapter,
    degrades to an empty pool, which is exactly today's report-only generation
    (:func:`~deepvision.study.source_pool.pool_for_paper` already treats a
    ``None``/absent retriever that way).
    """
    from deepvision.report.agent_bridge import build_retriever
    from deepvision.study.source_pool import pool_for_paper

    try:
        retriever = build_retriever(settings)
        if chapter_id:
            # Same guarantee chapter *reports* get: nothing outside the
            # chapter's pages may reach the generator. See
            # ``report/chapter_report_generator.py``'s identical wrapping.
            from deepvision.rag.chapter_scope import PageScopedRetriever
            from deepvision.report.chapter_report_generator import resolve_chapter

            chapter = resolve_chapter(paper_id, chapter_id)
            retriever = PageScopedRetriever(
                retriever, page_start=chapter.page_start, page_end=chapter.page_end
            )
    except Exception:  # noqa: BLE001 - the pool is a bonus, never a hard requirement
        log.warning(
            "could not build a source pool for this quiz; falling back to the "
            "report alone",
            extra={"paper_id": paper_id, "chapter_id": chapter_id},
            exc_info=True,
        )
        return []
    return pool_for_paper(paper_id, retriever)


def _persist_quiz(quiz: Quiz) -> None:
    """Write the quiz and its questions. One transaction, answer key included."""
    with session_scope() as session:
        session.add(
            QuizRow(
                id=quiz.id,
                paper_id=quiz.paper_id,
                title=quiz.title,
                question_count=len(quiz.questions),
                difficulty_mix=dict(quiz.difficulty_mix),
                kind_mix=dict(quiz.kind_mix),
                model_used=quiz.model_used,
                generated_at=quiz.generated_at,
            )
        )
        for question in quiz.questions:
            citation = question.citation
            session.add(
                QuizQuestionRow(
                    id=question.id,
                    quiz_id=quiz.id,
                    paper_id=quiz.paper_id,
                    index=question.index,
                    prompt=question.prompt,
                    kind=question.kind.value,
                    difficulty=question.difficulty.value,
                    options=[{"id": o.id, "text": o.text} for o in question.options],
                    correct_option_id=question.correct_option_id,
                    correct_answer_text=question.correct_answer_text,
                    accepted_answers=list(question.accepted_answers),
                    explanation=question.explanation,
                    citation_section=citation.section.value if citation else None,
                    citation_page=citation.page if citation else None,
                    citation_chunk_id=citation.chunk_id if citation else None,
                    citation_snippet=citation.snippet if citation else "",
                    created_at=question.created_at,
                )
            )


def _quiz_work(
    quiz_id: str,
    paper_id: str,
    request: QuizGenerateRequest,
    settings: AppSettings,
) -> "StudyWork":
    """Build the unit of work the study job runner will execute on its thread.

    Milestones bracket the one opaque generation call. Raising
    :class:`~deepvision.api.study_job_runner.JobAbort` marks the job failed with
    a user-facing message and no stack trace; any other exception is caught by
    the runner's worker guard and logged as an incident. Either way nothing
    escapes to crash the thread.
    """

    def work(progress: ProgressFn) -> str:
        progress(5, "Loading the report this quiz will be built from")
        sections = _load_sections(paper_id, request.chapter_id)

        with session_scope() as session:
            paper_title = _paper_title(session, paper_id)
            avoid_prompts = _previous_prompts(session, paper_id)

        progress(10, "Sweeping the paper for material the report summarised away")
        extra_chunks = _build_pool(paper_id, request.chapter_id, settings)

        progress(
            15,
            f"Writing {request.question_count} questions "
            "(this can take a minute on a local model)",
        )
        agent = QuizAgent(get_llm(settings))
        quiz = agent.generate(
            paper_id=paper_id,
            sections=sections,
            quiz_id=quiz_id,
            paper_title=paper_title,
            question_count=request.question_count,
            kinds=request.kinds,
            difficulties=request.difficulties,
            settings=settings,
            on_progress=progress,
            extra_chunks=extra_chunks,
            avoid_prompts=avoid_prompts,
        )

        if not quiz.questions:
            # An empty quiz is worse than no quiz: it pollutes the quiz list and
            # every attempt against it scores 0/0. Fail the job honestly.
            raise JobAbort(
                "no quizzable material was found in this paper's report — "
                "re-generate the report (with a model configured) and try again"
            )

        progress(90, "Saving quiz")
        _persist_quiz(quiz)
        return f"Quiz ready: {len(quiz.questions)} questions"

    return work


# ==========================================================================
# Routes
# ==========================================================================
@router.post("/generate", response_model=QuizGenerateResponse, status_code=202)
def generate_quiz(request: QuizGenerateRequest) -> QuizGenerateResponse:
    """Queue quiz generation and return the pollable job (202).

    Always 202, never a cached 200: generating a quiz deliberately creates a
    *new* quiz rather than replacing the old one, so past attempt history stays
    meaningful. ``quiz_id`` is minted here and is resolvable through
    ``GET /quiz/{quiz_id}`` the moment the job reaches ``done``.
    """
    from deepvision.ingestion import repo  # local import: keeps router import cheap

    paper = repo.get_paper_meta(request.paper_id)
    if paper is None:
        raise HTTPException(
            status_code=404, detail=f"paper not found: {request.paper_id}"
        )

    settings = request.settings_override or get_settings()
    quiz_id = new_id("quiz")
    try:
        handle = launch_study_job(
            _JOB_KIND,
            request.paper_id,
            _quiz_work(quiz_id, request.paper_id, request, settings),
            arxiv_id=paper.arxiv_id or "",
            start_message="Preparing quiz generation",
            done_message="Quiz ready",
            payload=quiz_id,
        )
        # On a lost race this is the WINNER's quiz id, not the one minted above:
        # the caller must poll the quiz that is actually being generated.
        job, quiz_id = handle.job, handle.payload
    except Exception as exc:  # pragma: no cover - defensive
        log.error(
            "failed to queue quiz generation",
            extra={"paper_id": request.paper_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=500, detail=f"failed to queue quiz generation: {exc}"
        ) from exc

    return QuizGenerateResponse(paper_id=request.paper_id, quiz_id=quiz_id, job=job)


@router.get("/paper/{paper_id}", response_model=QuizListResponse)
def list_quizzes(paper_id: str) -> QuizListResponse:
    """List a paper's quizzes, newest first (summaries only).

    Attempt counts and best scores are folded in from **one** query over the
    paper's attempts rather than one query per quiz.
    """
    with session_scope() as session:
        quiz_rows = session.exec(
            select(QuizRow)
            .where(QuizRow.paper_id == paper_id)
            .order_by(QuizRow.generated_at.desc())
        ).all()
        if not quiz_rows:
            return QuizListResponse(paper_id=paper_id, quizzes=[], count=0)

        attempts = session.exec(
            select(
                QuizAttemptRow.quiz_id,
                QuizAttemptRow.score,
                QuizAttemptRow.finished_at,
                QuizAttemptRow.mode,
            ).where(QuizAttemptRow.paper_id == paper_id)
        ).all()
        # Snapshot the ORM rows before the session closes (session_scope commits
        # on exit, which expires every instance it loaded).
        quizzes = [
            {
                "id": row.id,
                "paper_id": row.paper_id,
                "title": row.title or "",
                "question_count": row.question_count,
                "difficulty_mix": dict(row.difficulty_mix or {}),
                "generated_at": row.generated_at,
            }
            for row in quiz_rows
        ]

    # ``best`` counts FULL attempts only. A retry_missed run is scored over a
    # smaller, harder denominator, so letting "3/3 on the ones I got wrong"
    # outrank a real 9/10 would make the Report launch card lie. Retries only
    # supply the best score when there is no full attempt to report at all.
    stats: dict[str, dict[str, Any]] = {}
    for quiz_id, score, finished_at, mode in attempts:
        entry = stats.setdefault(
            quiz_id, {"count": 0, "best_full": None, "best_any": None, "last": None}
        )
        entry["count"] += 1
        score = score or 0.0
        key = "best_full" if _attempt_mode(mode) is AttemptMode.FULL else "best_any"
        if entry[key] is None or score > entry[key]:
            entry[key] = score
        if entry["best_any"] is None or score > entry["best_any"]:
            entry["best_any"] = score
        if entry["last"] is None or (finished_at and finished_at > entry["last"]):
            entry["last"] = finished_at

    def _best(entry: dict[str, Any]) -> Optional[float]:
        return entry.get("best_full") if entry.get("best_full") is not None else entry.get("best_any")

    summaries = [
        QuizSummary(
            **row,
            attempt_count=stats.get(row["id"], {}).get("count", 0),
            best_score=_best(stats.get(row["id"], {})),
            last_attempted_at=stats.get(row["id"], {}).get("last"),
        )
        for row in quizzes
    ]
    return QuizListResponse(
        paper_id=paper_id, quizzes=summaries, count=len(summaries)
    )


@router.get("/attempts/{attempt_id}", response_model=QuizAttempt)
def get_attempt(attempt_id: str) -> QuizAttempt:
    """Return one graded attempt in full, with explanations.

    Declared **before** ``/{quiz_id}`` so ``/quiz/attempts/...`` cannot be
    swallowed by the quiz-id route.
    """
    with session_scope() as session:
        attempt_row = session.get(QuizAttemptRow, attempt_id)
        if attempt_row is None:
            raise HTTPException(
                status_code=404, detail=f"attempt not found: {attempt_id}"
            )
        answer_rows = session.exec(
            select(QuizAnswerRow).where(QuizAnswerRow.attempt_id == attempt_id)
        ).all()
        question_rows = session.exec(
            select(QuizQuestionRow)
            .where(QuizQuestionRow.quiz_id == attempt_row.quiz_id)
            .order_by(QuizQuestionRow.index)
        ).all()
        questions = {row.id: _question_from_row(row) for row in question_rows}
        return _attempt_from_rows(attempt_row, answer_rows, questions)


def _attempt_from_rows(
    attempt_row: QuizAttemptRow,
    answer_rows: list[QuizAnswerRow],
    questions: dict[str, QuizQuestion],
) -> QuizAttempt:
    """Rebuild a graded attempt from storage.

    The verdict comes from the stored ``is_correct`` — it is never re-derived,
    so a later edit to the grader cannot silently rewrite history. The
    explanation and citation come from the question, which is why this is the
    one read path allowed to surface them.
    """
    by_question = {row.question_id: row for row in answer_rows}
    graded: list[GradedAnswer] = []
    for question_id in attempt_row.question_ids or []:
        question = questions.get(question_id)
        answer_row = by_question.get(question_id)
        if question is None:
            continue
        graded.append(
            GradedAnswer(
                question_id=question.id,
                index=question.index,
                prompt=question.prompt,
                kind=question.kind,
                difficulty=question.difficulty,
                options=list(question.options),
                selected_option_id=answer_row.selected_option_id if answer_row else None,
                answer_text=answer_row.answer_text if answer_row else None,
                is_correct=bool(answer_row.is_correct) if answer_row else False,
                correct_option_id=question.correct_option_id,
                correct_answer_text=question.correct_answer_text,
                explanation=question.explanation,
                citation=question.citation,
            )
        )
    return QuizAttempt(
        id=attempt_row.id,
        quiz_id=attempt_row.quiz_id,
        paper_id=attempt_row.paper_id,
        mode=_attempt_mode(attempt_row.mode),
        retry_of_attempt_id=attempt_row.retry_of_attempt_id,
        question_ids=list(attempt_row.question_ids or []),
        answers=graded,
        total=attempt_row.total,
        correct=attempt_row.correct,
        score=attempt_row.score,
        started_at=attempt_row.started_at,
        finished_at=attempt_row.finished_at,
        missed_question_ids=list(attempt_row.missed_question_ids or []),
    )


@router.get("/{quiz_id}", response_model=QuizPublic)
def get_quiz(quiz_id: str) -> QuizPublic:
    """Return the quiz with every answer and explanation stripped."""
    with session_scope() as session:
        quiz_row, questions = _load_quiz(session, quiz_id)
        paper_title = _paper_title(session, quiz_row.paper_id)
        return _quiz_public(quiz_row, questions, paper_title=paper_title)


@router.post("/{quiz_id}/check", response_model=QuizCheckResponse)
def check_answer(quiz_id: str, request: QuizCheckRequest) -> QuizCheckResponse:
    """Grade ONE answer and return the verdict. **Writes nothing.**

    The read-only twin of :func:`submit_attempt`, for immediate per-question
    feedback while the user is still working through the quiz. Same grader, so
    the verdict here is always the verdict the recorded attempt will produce for
    the same input.

    No ``quiz_attempts`` / ``quiz_answers`` row is created — that is the entire
    point. The run is recorded once, by ``/attempt``, when the user finishes.
    """
    with session_scope() as session:
        _quiz_row, questions = _load_quiz(session, quiz_id)
        by_id = {question.id: question for question in questions}
        question = by_id.get(request.answer.question_id)
        if question is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "this question does not belong to this quiz: "
                    f"{request.answer.question_id}"
                ),
            )
        return QuizCheckResponse(
            quiz_id=quiz_id, graded=grade_question(question, request.answer)
        )


@router.post("/{quiz_id}/attempt", response_model=QuizAttempt)
def submit_attempt(quiz_id: str, request: QuizAttemptRequest) -> QuizAttempt:
    """Grade an attempt server-side and persist it. Synchronous.

    Deterministic and model-free, so it is fast enough to run inside the
    request. Any correctness claim in the request body is ignored — the verdict
    below comes from :func:`~deepvision.study.grading.grade_answers` alone.
    """
    finished_at = datetime.utcnow()
    started_at = _parse_started_at(request.started_at, finished_at)

    with session_scope() as session:
        quiz_row, questions = _load_quiz(session, quiz_id)
        if not questions:
            raise HTTPException(
                status_code=400, detail=f"quiz {quiz_id} has no questions"
            )
        by_id = {question.id: question for question in questions}

        mode = request.mode or AttemptMode.FULL
        retry_of_attempt_id: Optional[str] = None
        submitted_ids = {
            answer.question_id for answer in (request.answers or []) if answer
        }

        if mode is AttemptMode.RETRY_MISSED:
            if not request.retry_of_attempt_id:
                raise HTTPException(
                    status_code=400,
                    detail="retry_of_attempt_id is required when mode is 'retry_missed'",
                )
            source = session.get(QuizAttemptRow, request.retry_of_attempt_id)
            if source is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"attempt not found: {request.retry_of_attempt_id}",
                )
            if source.quiz_id != quiz_id:
                raise HTTPException(
                    status_code=400,
                    detail="retry_of_attempt_id belongs to a different quiz",
                )
            # dict.fromkeys de-duplicates while preserving order: a repeated id
            # would otherwise double-count in the score and collide on the
            # (attempt_id, question_id) unique constraint when persisting.
            missed = [
                qid
                for qid in dict.fromkeys(source.missed_question_ids or [])
                if qid in by_id
            ]
            if not missed:
                raise HTTPException(
                    status_code=400,
                    detail="that attempt has no missed questions to retry",
                )
            # The scope is the RECORDED misses, not whatever was submitted:
            # fixing the denominator here is what stops a cherry-picked subset
            # from producing a flattering score.
            unexpected = submitted_ids - set(missed)
            if unexpected:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "these questions were not missed in the source attempt: "
                        + ", ".join(sorted(unexpected))
                    ),
                )
            scope_ids = missed
            retry_of_attempt_id = source.id
        else:
            mode = AttemptMode.FULL
            unknown = submitted_ids - set(by_id)
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "these questions do not belong to this quiz: "
                        + ", ".join(sorted(unknown))
                    ),
                )
            scope_ids = [question.id for question in questions]

        scope = sorted((by_id[qid] for qid in scope_ids), key=lambda q: q.index)
        graded = grade_answers(scope, request.answers or [])

        correct = sum(1 for answer in graded if answer.is_correct)
        total = len(scope)
        missed_ids = [answer.question_id for answer in graded if not answer.is_correct]
        attempt_id = new_id("qa")

        session.add(
            QuizAttemptRow(
                id=attempt_id,
                quiz_id=quiz_id,
                paper_id=quiz_row.paper_id,
                mode=mode.value,
                retry_of_attempt_id=retry_of_attempt_id,
                question_ids=[question.id for question in scope],
                missed_question_ids=missed_ids,
                total=total,
                correct=correct,
                score=(correct / total) if total else 0.0,
                started_at=started_at,
                finished_at=finished_at,
            )
        )
        elapsed_by_question = {
            answer.question_id: answer.elapsed_ms
            for answer in (request.answers or [])
            if answer
        }
        for answer in graded:
            session.add(
                QuizAnswerRow(
                    id=new_id("qans"),
                    attempt_id=attempt_id,
                    question_id=answer.question_id,
                    quiz_id=quiz_id,
                    selected_option_id=answer.selected_option_id,
                    answer_text=answer.answer_text,
                    is_correct=answer.is_correct,
                    correct_option_id=answer.correct_option_id,
                    elapsed_ms=elapsed_by_question.get(answer.question_id),
                    answered_at=finished_at,
                )
            )

        return QuizAttempt(
            id=attempt_id,
            quiz_id=quiz_id,
            paper_id=quiz_row.paper_id,
            mode=mode,
            retry_of_attempt_id=retry_of_attempt_id,
            question_ids=[question.id for question in scope],
            answers=graded,
            total=total,
            correct=correct,
            score=(correct / total) if total else 0.0,
            started_at=started_at,
            finished_at=finished_at,
            missed_question_ids=missed_ids,
        )


@router.get("/{quiz_id}/attempts", response_model=QuizAttemptHistoryResponse)
def list_attempts(quiz_id: str) -> QuizAttemptHistoryResponse:
    """Attempt history for a quiz, newest first (summaries only)."""
    with session_scope() as session:
        quiz_row = session.get(QuizRow, quiz_id)
        if quiz_row is None:
            raise HTTPException(status_code=404, detail=f"quiz not found: {quiz_id}")
        # Read every ORM attribute *inside* the session: session_scope commits
        # on exit, which expires the instance, and touching it afterwards raises
        # DetachedInstanceError.
        paper_id = quiz_row.paper_id
        rows = session.exec(
            select(QuizAttemptRow)
            .where(QuizAttemptRow.quiz_id == quiz_id)
            .order_by(QuizAttemptRow.finished_at.desc())
        ).all()
        summaries = [_attempt_summary(row) for row in rows]

    # Best score across *full* attempts when there are any: a retry_missed
    # score is measured over a different (harder) denominator, so mixing them
    # would let "3/3 on the ones I got wrong" outrank a real 9/10.
    full_scores = [s.score for s in summaries if s.mode is AttemptMode.FULL]
    best = max(full_scores) if full_scores else (
        max((s.score for s in summaries), default=None)
    )
    return QuizAttemptHistoryResponse(
        quiz_id=quiz_id,
        paper_id=paper_id,
        attempts=summaries,
        count=len(summaries),
        best_score=best,
    )


@router.post("/{quiz_id}/retry", response_model=QuizPublic)
def retry_missed(quiz_id: str, request: QuizRetryRequest) -> QuizPublic:
    """Return only the questions missed in ``source_attempt_id`` (answers hidden).

    A retry payload is an ordinary :class:`QuizPublic` — same hidden shape as a
    first pass, just a shorter question list — flagged with
    ``mode='retry_missed'`` so the UI can label the score as not comparable.
    """
    with session_scope() as session:
        quiz_row, questions = _load_quiz(session, quiz_id)
        source = session.get(QuizAttemptRow, request.source_attempt_id)
        if source is None:
            raise HTTPException(
                status_code=404,
                detail=f"attempt not found: {request.source_attempt_id}",
            )
        if source.quiz_id != quiz_id:
            raise HTTPException(
                status_code=400, detail="that attempt belongs to a different quiz"
            )
        missed = set(source.missed_question_ids or [])
        subset = [question for question in questions if question.id in missed]
        if not subset:
            raise HTTPException(
                status_code=400, detail="that attempt has no missed questions to retry"
            )
        paper_title = _paper_title(session, quiz_row.paper_id)
        return _quiz_public(
            quiz_row,
            subset,
            paper_title=paper_title,
            mode=AttemptMode.RETRY_MISSED,
            retry_of_attempt_id=source.id,
        )


def _parse_started_at(raw: Optional[str], finished_at: datetime) -> datetime:
    """Parse the client's ISO ``started_at``, clamped to something sane.

    A bogus or absent value becomes ``finished_at`` rather than an error: the
    field is advisory (it drives a "took 4m12s" label), and rejecting a
    submission over it would throw away the user's answers.
    """
    if not raw:
        return finished_at
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return finished_at
    if parsed.tzinfo is not None:
        # Everything persisted here is naive UTC (``datetime.utcnow``), so an
        # offset-aware value is converted rather than merely stripped.
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed if parsed <= finished_at else finished_at
