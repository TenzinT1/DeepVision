"""Process-level configuration loaded from the environment / ``.env``.

This is the *deployment* configuration (paths, ports, model names, default
provider modes) — distinct from :class:`deepvision.models.settings.AppSettings`,
which is the *user-editable* settings object persisted in the DB and edited via
the Settings screen. ``config.py`` seeds the initial ``AppSettings`` defaults and
provides everything that must be known before a DB row exists.. Pydantic-settings v2.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Config", "get_config"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config(BaseSettings):
    """Environment-driven configuration. All fields overridable via ``.env``.

    Env vars are prefixed ``DEEPVISION_`` (e.g. ``DEEPVISION_API_PORT=9000``).
    """

    model_config = SettingsConfigDict(
        env_prefix="DEEPVISION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Paths ----------------------------------------------------------
    project_root: Path = Field(
        default=_PROJECT_ROOT,
        description="Repository root; other paths derive from it by default.",
    )
    data_dir: Path = Field(
        default=_PROJECT_ROOT / "deepvision" / "data",
        description="Local file store for PDFs, figure crops and page images.",
    )
    sqlite_path: Path = Field(
        default=_PROJECT_ROOT / "deepvision" / "data" / "deepvision.db",
        description="SQLite database file for papers/jobs/reports/chat/settings.",
    )
    chroma_dir: Path = Field(
        default=_PROJECT_ROOT / "deepvision" / "data" / "chroma",
        description="Persistent ChromaDB directory for vector storage.",
    )

    # ---- API ------------------------------------------------------------
    api_host: str = Field(default="127.0.0.1", description="Uvicorn bind host.")
    api_port: int = Field(default=8000, description="Uvicorn bind port.")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins for the frontend dev servers.",
    )

    # ---- Provider selection --------------------------------------------
    strict_providers: bool = Field(
        default=True,
        description=(
            "When True, local stages use the REAL adapters (Ollama LLM, "
            "transformers vision, sentence-transformers embeddings). Set to "
            "False (DEEPVISION_STRICT_PROVIDERS=0) to force the offline "
            "Echo/Null/Hash stubs — used by the smoke check / CI so it runs "
            "with no models installed."
        ),
    )

    # ---- Default provider modes (seed AppSettings) ----------------------
    default_llm_mode: str = Field(
        default="local", description="'local' or 'api' — LLM default (local-first)."
    )
    default_vision_mode: str = Field(
        default="local", description="'local' or 'api' — Vision default (local-first)."
    )
    default_embedding_mode: str = Field(
        default="local", description="'local' or 'api' — Embeddings default (local-first)."
    )

    # ---- Local model identifiers ---------------------------------------
    ollama_host: str = Field(
        default="http://localhost:11434", description="Ollama server base URL."
    )
    local_llm_model: str = Field(
        default="llama3.1:8b", description="Default local Ollama LLM model tag."
    )
    local_llm_timeout: float = Field(
        default=600.0,
        gt=0,
        description=(
            "Per-call socket timeout (seconds) for the local Ollama LLM. "
            "Measured on this machine, a Standard report's 12 calls averaged "
            "~105s each on llama3.1:8b and a Detailed call is materially "
            "longer, so the old 120s ceiling sat right at the edge and six of "
            "twelve calls tripped it and fell back to the extractive draft. "
            "600s gives real headroom so the fallback fires only on a "
            "genuinely stuck server, not on normal slow generation. Override "
            "with DEEPVISION_LOCAL_LLM_TIMEOUT. Must be > 0: this value goes "
            "straight to urllib's socket timeout, where a negative raises "
            "ValueError mid-call and 0 makes every local call fail instantly, "
            "so both are rejected at startup rather than at generation time."
        ),
    )
    local_vision_model: str = Field(
        default="Qwen/Qwen2-VL-7B-Instruct",
        description="Default local HF vision model (Qwen2-VL / Florence-2).",
    )
    local_embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Default local sentence-transformers embedding model.",
    )
    embedding_dim: int = Field(
        default=1024, description="Vector dimension of the default embedding model."
    )

    # ---- API provider defaults (used only when a stage is 'api') --------
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    anthropic_base_url: str = Field(default="https://api.anthropic.com")
    api_llm_model: str = Field(default="claude-sonnet-5")
    api_vision_model: str = Field(default="claude-sonnet-5")
    api_embedding_model: str = Field(default="text-embedding-3-large")

    # NOTE: API keys are NOT read from process env into long-lived config;
    # they live in the user's persisted AppSettings (see models/settings.py)
    # and are supplied per-request to provider factories. This keeps secrets
    # out of the deployment config surface.

    def ensure_dirs(self) -> None:
        """Create the local data directories if they do not yet exist."""
        for path in (self.data_dir, self.chroma_dir, self.sqlite_path.parent):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the process-wide singleton :class:`Config` (cached)."""
    return Config()
