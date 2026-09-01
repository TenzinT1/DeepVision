"""User-editable application settings (the Settings screen).

``AppSettings`` is the persisted, user-facing configuration edited on the
Settings screen and read by the provider factory. It is distinct from
:class:`deepvision.config.Config` (deployment/environment). Provider stages are
each ``local`` or ``api``; API keys are only meaningful when a stage is ``api``.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ProviderMode",
    "OCRLanguage",
    "ReportDetail",
    "DETAIL_LEVEL_INFO",
    "ProviderKeys",
    "AppSettings",
]


class ProviderMode(str, Enum):
    """Where a provider stage runs: locally (free) or via an external API."""

    LOCAL = "local"
    API = "api"


class ReportDetail(str, Enum):
    """How long / detailed the generated report sections should be.

    Drives how much grounded material each section is written from and the
    target length the summarizer asks the LLM for.

    The string values are the wire contract (``"concise"`` / ``"standard"`` /
    ``"detailed"``) and must never be renamed — the UI dropdown and the
    persisted settings row both carry them verbatim.
    """

    CONCISE = "concise"
    STANDARD = "standard"
    DETAILED = "detailed"


#: The *contract* of each detail level, shared by the backend (the summarizer
#: turns ``blurb`` into a concrete length target it asks the LLM for) and the UI
#: (renders ``label`` in the dropdown and ``blurb`` as helper text next to it).
#: Surfaced on the wire as ``SettingsResponse.detail_levels`` so the two sides
#: cannot drift: whatever the user is promised here is what the generator aims
#: for. Concise vs. standard must *actually* differ in output length.
DETAIL_LEVEL_INFO: dict[ReportDetail, dict[str, str]] = {
    ReportDetail.CONCISE: {
        "label": "Concise",
        "blurb": "~1 short paragraph per section (2-3 sentences)",
    },
    ReportDetail.STANDARD: {
        "label": "Standard",
        "blurb": "~2 paragraphs per section (6-9 sentences)",
    },
    ReportDetail.DETAILED: {
        "label": "Detailed",
        "blurb": "~4-5 paragraphs per section (16-22 sentences)",
    },
}


class OCRLanguage(str, Enum):
    """OCR language options exposed in the UI.

    Values are tesseract language specs (``+`` joins multiple); ``auto`` requests
    auto-detection.
    """

    ENGLISH = "eng"
    ENGLISH_FRENCH = "eng+fra"
    CHINESE_SIMPLIFIED = "chi_sim"
    JAPANESE = "jpn"
    AUTO = "auto"


class ProviderKeys(BaseModel):
    """API keys for stages set to ``api``.

    Only consulted when the corresponding stage mode is ``api``. Stored locally
    in the settings row; never logged and never sent anywhere but the chosen
    provider endpoint.
    """

    llm_api_key: Optional[str] = Field(default=None, description="LLM provider key.")
    vision_api_key: Optional[str] = Field(
        default=None, description="Vision provider key (optional)."
    )
    embedding_api_key: Optional[str] = Field(
        default=None, description="Embedding provider key (optional)."
    )


class AppSettings(BaseModel):
    """The full user-editable settings object.

    Local-first defaults: all three provider stages default to ``local``. The
    provider factory reads this object to construct concrete providers.
    """

    model_config = ConfigDict(use_enum_values=False)

    # ---- Providers ------------------------------------------------------
    llm_mode: ProviderMode = Field(
        default=ProviderMode.LOCAL, description="LLM stage: local or api."
    )
    vision_mode: ProviderMode = Field(
        default=ProviderMode.LOCAL, description="Vision stage: local or api."
    )
    embedding_mode: ProviderMode = Field(
        default=ProviderMode.LOCAL, description="Embeddings stage: local or api."
    )

    # Concrete model identifiers. These are no longer free text: the UI picks a
    # ``ModelChoice.id`` from the backend-published catalog
    # (``deepvision.api.schemas.MODEL_CATALOG``, served on
    # ``SettingsResponse.model_catalog``), and selecting a choice sets BOTH the
    # ``<stage>_model`` id and the matching ``<stage>_mode``. ``None`` still
    # means "use the Config default for this mode" so existing settings rows
    # and non-UI callers keep working.
    llm_model: Optional[str] = Field(
        default=None,
        description=(
            "ModelChoice.id from the catalog's 'llm' list "
            "(e.g. 'claude-sonnet-5'); None means use the Config default."
        ),
    )
    vision_model: Optional[str] = Field(
        default=None,
        description="ModelChoice.id from the catalog's 'vision' list; None = Config default.",
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="ModelChoice.id from the catalog's 'embedding' list; None = Config default.",
    )

    keys: ProviderKeys = Field(
        default_factory=ProviderKeys, description="API keys (used only for api stages)."
    )

    # ---- Extraction -----------------------------------------------------
    ocr_language: OCRLanguage = Field(
        default=OCRLanguage.ENGLISH, description="OCR language selection."
    )
    chunk_size: int = Field(
        default=800, ge=256, le=2048, description="Chunk size in tokens (256-2048)."
    )
    chunk_overlap: int = Field(
        default=120, ge=0, le=400, description="Chunk overlap in tokens (0-400)."
    )
    image_embedding: bool = Field(
        default=False,
        description="Advanced: embed figure crops for visual retrieval (slower).",
    )

    # ---- Report generation ----------------------------------------------
    report_detail: ReportDetail = Field(
        default=ReportDetail.STANDARD,
        description="How long/detailed each report section should be.",
    )

    @model_validator(mode="after")
    def _check_overlap(self) -> "AppSettings":
        """Overlap must be strictly smaller than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        return self

    @property
    def needs_keys(self) -> bool:
        """True if any stage is set to ``api`` (drives the API-keys card)."""
        return ProviderMode.API in (
            self.llm_mode,
            self.vision_mode,
            self.embedding_mode,
        )
