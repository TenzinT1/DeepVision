"""Ingestion domain — the seven-stage async pipeline. Exposes the orchestrator plus one interface per pipeline stage:
Download -> Extract text -> Extract images -> OCR -> Vision analysis -> Embedding
-> Generate report (embedding itself lives in the RAG layer; report generation in
the report layer — both are *tracked stages* of the job, so ``state == done``
means the report exists). All bodies are NotImplementedError stubs against
fixed signatures.
"""

from deepvision.ingestion.arxiv_search import ArxivSearcher, ArxivSearcherService
from deepvision.ingestion.image_extractor import ImageExtractor, ImageExtractorService
from deepvision.ingestion.ocr_processor import OCRProcessor, OCRProcessorService
from deepvision.ingestion.orchestrator import (
    DefaultIngestionOrchestrator,
    IngestionOrchestrator,
)
from deepvision.ingestion.page_renderer import PageRenderer, PageRendererService
from deepvision.ingestion.pdf_downloader import PDFDownloader, PDFDownloaderService
from deepvision.ingestion.pdf_parser import PDFParser, PDFParserService
from deepvision.ingestion.vision_processor import VisionProcessor, VisionProcessorService

__all__ = [
    "IngestionOrchestrator",
    "DefaultIngestionOrchestrator",
    "ArxivSearcher",
    "ArxivSearcherService",
    "PDFDownloader",
    "PDFDownloaderService",
    "PDFParser",
    "PDFParserService",
    "ImageExtractor",
    "ImageExtractorService",
    "PageRenderer",
    "PageRendererService",
    "OCRProcessor",
    "OCRProcessorService",
    "VisionProcessor",
    "VisionProcessorService",
]
