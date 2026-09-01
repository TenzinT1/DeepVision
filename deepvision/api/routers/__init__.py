"""API routers, one module per endpoint group.

Each router exposes a module-level ``router: APIRouter``. Every router must be
listed in ``ALL_ROUTERS`` below to be mounted by ``api/main.py``.
"""

from deepvision.api.routers import (
    chat,
    flashcards,
    health,
    ingest,
    jobs,
    media,
    papers,
    quiz,
    report,
    search,
    settings,
    study,
    upload,
)

#: All routers, in the order they are included on the app.
ALL_ROUTERS = [
    health.router,
    search.router,
    ingest.router,
    upload.router,
    jobs.router,
    papers.router,
    report.router,
    chat.router,
    media.router,
    # Study layer — three routers, three owners, three disjoint prefixes:
    #   /api/study/*       cross-paper overview, due queue, per-paper stats
    #   /api/flashcards/*  deck generation, listing, SM-2 review, star
    #   /api/quiz/*        quiz generation, hidden-answer fetch, grading, history
    study.router,
    flashcards.router,
    quiz.router,
    settings.router,
]

__all__ = [
    "ALL_ROUTERS",
    "health",
    "search",
    "ingest",
    "upload",
    "jobs",
    "papers",
    "report",
    "chat",
    "media",
    "study",
    "flashcards",
    "quiz",
    "settings",
]
