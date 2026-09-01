"""Embedding-stage glue: chunk text/caption selection -> EmbeddingProvider.

Not part of the frozen ``Chunker``/``VectorStore``/``Retriever`` ABCs (those
signatures are fixed by). This is an additive convenience the
RAG domain owns to make the pipeline's Embedding stage
(``Chunker.chunk -> EmbeddingProvider.embed_texts -> VectorStore.upsert``,) a single call for the ingestion layer's orchestrator, instead of every
caller re-deriving which text to embed per modality.

Policy:
- Text-bearing chunks (``TextChunk``/``OCRChunk``/``VisionInsightChunk``) embed
  their ``.text`` via ``embed_texts`` (batched).
- ``ImageChunk``: when ``settings.image_embedding`` is on, embed the crop via
  ``EmbeddingProvider.embed_image`` (visual embedding). If that returns
  ``None`` (provider has no visual embedder, or the toggle is off, or the
  image file is missing) **or** returns a vector whose dimension doesn't
  match ``embedding_provider.dim`` (e.g. a CLIP visual embedder alongside a
  differently-sized text embedder), fall back to embedding the figure
  caption text so the figure remains retrievable by keyword/semantic caption
  search rather than being silently dropped -- and so the single dim-keyed
  vector store collection never receives mixed-dimension vectors.
- A chunk with no embeddable content at all (empty text, no caption, no image
  embedding) is dropped from the returned pair rather than upserted with a
  meaningless vector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from deepvision.config import Config, get_config
from deepvision.models.chunks import AnyChunk, ImageChunk
from deepvision.models.settings import AppSettings
from deepvision.providers.base import EmbeddingProvider
from deepvision.utils.logger import get_logger

__all__ = ["embed_chunks"]

log = get_logger(__name__)


def embed_chunks(
    chunks: Sequence[AnyChunk],
    settings: AppSettings,
    embedding_provider: EmbeddingProvider,
    *,
    config: Optional[Config] = None,
) -> tuple[list[AnyChunk], list[list[float]]]:
    """Embed ``chunks`` and return the ``(chunks, embeddings)`` pair to upsert.

    The returned lists are index-aligned and ready to pass straight to
    ``VectorStore.upsert(chunks, embeddings)``. Chunks that end up with no
    embeddable content are omitted from both lists (and logged).
    """
    if not chunks:
        return [], []

    cfg = config or get_config()
    base_dir = Path(cfg.data_dir)

    embeddings: list[Optional[list[float]]] = [None] * len(chunks)

    # 1) Image chunks with the advanced visual-embedding toggle on: try
    #    embed_image first; anything left unset falls through to caption text.
    if settings.image_embedding:
        for i, ch in enumerate(chunks):
            if not isinstance(ch, ImageChunk) or not ch.image_path:
                continue
            abs_path = base_dir / ch.image_path
            if not abs_path.exists():
                log.warning("image chunk path missing on disk", chunk_id=ch.id, path=str(abs_path))
                continue
            try:
                vec = embedding_provider.embed_image(str(abs_path))
            except Exception as exc:  # provider-level failure: degrade, don't crash
                log.warning(
                    "embed_image failed; will fall back to caption text",
                    chunk_id=ch.id,
                    error=str(exc),
                )
                vec = None
            if vec is not None and len(vec) != embedding_provider.dim:
                # The visual embedder (e.g. CLIP, 512-dim) can return a
                # different dimensionality than the provider's text vectors
                # (e.g. bge, 1024-dim). The vector store's collection is
                # keyed by a single dim (open_vector_store(embed_provider.dim)
                # in the orchestrator), so a mismatched vector can't be
                # upserted alongside text vectors without corrupting the
                # collection (Chroma raises InvalidDimension; the JSON
                # fallback store silently zip-truncates similarity). Discard
                # it and fall through to embedding the caption text instead.
                log.warning(
                    "discarding image embedding with mismatched dimension; "
                    "falling back to caption text",
                    chunk_id=ch.id,
                    image_dim=len(vec),
                    text_dim=embedding_provider.dim,
                )
                vec = None
            if vec is not None:
                embeddings[i] = vec

    # 2) Batch-embed everything still unembedded that has usable text
    #    (native .text for text-bearing chunks, caption for image/vision).
    text_indices: list[int] = []
    texts: list[str] = []
    for i, ch in enumerate(chunks):
        if embeddings[i] is not None:
            continue
        text = ch.text or getattr(ch, "caption", None) or ""
        text = text.strip()
        if text:
            text_indices.append(i)
            texts.append(text)

    if texts:
        vectors = embedding_provider.embed_texts(texts)
        for idx, vec in zip(text_indices, vectors):
            embeddings[idx] = vec

    out_chunks: list[AnyChunk] = []
    out_embeddings: list[list[float]] = []
    for ch, vec in zip(chunks, embeddings):
        if vec is None:
            log.debug("skipping chunk with no embeddable content", chunk_id=ch.id)
            continue
        out_chunks.append(ch)
        out_embeddings.append(vec)

    log.debug(
        "embedded chunks",
        total=len(chunks),
        embedded=len(out_chunks),
        skipped=len(chunks) - len(out_chunks),
    )
    return out_chunks, out_embeddings
