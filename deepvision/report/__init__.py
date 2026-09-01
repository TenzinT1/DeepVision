"""Report domain — report generation, media/citation serving, and export.

Assembles the :class:`Report` (via the agents), serves media and citation
payloads for the interactive UI, and exports to Markdown/PDF.
"""

from deepvision.report.exporters import DefaultReportExporter, ReportExporter
from deepvision.report.figure_links import attach_referenced_figures
from deepvision.report.interactive_sections import (
    CitationResolver,
    DefaultCitationResolver,
    DefaultMediaService,
    MediaService,
    build_section,
    normalize_sections,
)
from deepvision.report.report_generator import (
    DefaultReportGenerator,
    PaperNotFoundError,
    ReportGenerator,
)

__all__ = [
    "ReportGenerator",
    "DefaultReportGenerator",
    "PaperNotFoundError",
    "MediaService",
    "DefaultMediaService",
    "CitationResolver",
    "DefaultCitationResolver",
    "ReportExporter",
    "DefaultReportExporter",
    "build_section",
    "normalize_sections",
    "attach_referenced_figures",
]
