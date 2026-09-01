"""Database engine and session factory.

Provides the SQLite engine and a session context manager used by the API's
``get_session`` dependency and by the ingestion worker. The engine builder and
``init_db`` (create tables) are fully implemented so the schema can be
materialized; per-domain query logic lives in the builder services.

Vectors are NOT stored here — see the RAG layer's ChromaDB vector store.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from deepvision.config import Config, get_config
from deepvision.db import schema  # noqa: F401  (register tables on SQLModel.metadata)

__all__ = ["get_engine", "init_db", "session_scope", "schema"]

# Process-wide engine cache keyed by the resolved SQLite URL. A plain
# ``@lru_cache`` on ``get_engine`` cannot be used here because ``Config`` (a
# pydantic-settings model) is unhashable, so passing one — as the app's own
# ``main.py`` lifespan and ``init_db(cfg)`` do — would raise ``TypeError:
# unhashable type: 'Config'`` and (in the lifespan's try/except) leave the DB
# uninitialized on a clean boot. Caching on the URL string keeps the intended
# "one engine per database file" singleton behavior while accepting a Config.
_ENGINES: dict[str, Engine] = {}


def get_engine(config: Config | None = None) -> Engine:
    """Return the process-wide SQLite engine (cached per database URL).

    Ensures the data directory exists first. ``check_same_thread=False`` lets the
    async API and the ingestion worker share the engine.
    """
    cfg = config or get_config()
    cfg.ensure_dirs()
    url = f"sqlite:///{cfg.sqlite_path}"
    engine = _ENGINES.get(url)
    if engine is None:
        engine = create_engine(
            url, echo=False, connect_args={"check_same_thread": False}
        )
        _ENGINES[url] = engine
    return engine


def init_db(config: Config | None = None) -> None:
    """Create all tables defined in :mod:`deepvision.db.schema` if absent."""
    SQLModel.metadata.create_all(get_engine(config))


@contextmanager
def session_scope(config: Config | None = None) -> Iterator[Session]:
    """Yield a transactional :class:`~sqlmodel.Session`, committing on success.

    Rolls back and re-raises on error; always closes.
    """
    session = Session(get_engine(config))
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
