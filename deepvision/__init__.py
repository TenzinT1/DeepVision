"""DeepVision — a local-first multimodal research assistant.

This package is the shared contract layer produced in Wave 0. It defines the
data models, provider interfaces, DB schema, HTTP API contract, and shared
utilities that all downstream builder domains (ingestion, rag, agents, report,
api, providers) build against.

Importing :mod:`deepvision` pulls in only the contract surface (models, config,
utils, provider interfaces) — never business logic — so it is always safe and
"""

from __future__ import annotations

__version__ = "0.9.0"

from deepvision import models  # noqa: F401  (re-export the model namespace)
from deepvision.config import Config, get_config

__all__ = ["models", "Config", "get_config", "__version__"]
