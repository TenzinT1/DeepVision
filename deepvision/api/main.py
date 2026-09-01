"""FastAPI application factory and router wiring.

Architect-owned: the app, CORS, lifespan (DB init), and router includes are all
here and functional. Route *bodies* live in the router modules (currently 501
stubs owned by the builder domains). Every route is mounted under ``/api``.

Run with: ``uvicorn deepvision.api.main:app --reload``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deepvision import __version__
from deepvision.api.routers import ALL_ROUTERS
from deepvision.config import get_config
from deepvision.utils import get_logger

log = get_logger(__name__)

API_PREFIX = "/api"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize the database (create tables) and data dirs on startup."""
    cfg = get_config()
    cfg.ensure_dirs()
    try:
        from deepvision.db import init_db

        init_db(cfg)
        log.info("database initialized", extra={"path": str(cfg.sqlite_path)})
    except Exception as exc:  # pragma: no cover - defensive startup logging
        log.error("db init failed", extra={"error": str(exc)})
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    cfg = get_config()
    app = FastAPI(
        title="DeepVision API",
        version=__version__,
        description="Local-first multimodal research assistant.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in ALL_ROUTERS:
        app.include_router(router, prefix=API_PREFIX)
    return app


app = create_app()
