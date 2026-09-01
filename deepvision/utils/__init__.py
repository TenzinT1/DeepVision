"""Shared utilities: structured logging and id generation.

Everything here is fully implemented and safe to import from any layer.
"""

from deepvision.utils import ids, logger
from deepvision.utils.logger import get_logger

__all__ = ["ids", "logger", "get_logger"]
