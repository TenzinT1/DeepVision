"""Vector store interface — persistent ChromaDB.

Wraps a persistent ChromaDB collection: upsert chunk embeddings, query by
vector, and delete a paper's vectors.
``chromadb`` is an optional-at-import dependency: the module always imports
cleanly, but :class:`ChromaVectorStore` needs it installed to actually talk to
a real Chroma collection. When it is not installed, ``ChromaVectorStore``
transparently falls back to a small dependency-free, JSON-persisted cosine
index living under the same ``chroma_dir`` so ingestion/retrieval still work
end-to-end (degraded, not a crash) during early build waves or minimal
environments.
"""

from __future__ import annotations

import abc
import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from deepvision.models.chunks import (
    AnyChunk,
    BBox,
    ImageChunk,
    Modality,
    OCRChunk,
    Provenance,
    SourceRef,
    TextChunk,
    VisionInsightChunk,
)
from deepvision.utils.logger import get_logger

__all__ = [
    "VectorHit",
    "VectorStore",
    "ChromaVectorStore",
    "chunk_to_metadata",
    "chunk_from_hit",
    "open_vector_store",
]

log = get_logger(__name__)

try:  # pragma: no cover - exercised only when chromadb is installed
    import chromadb

    _HAS_CHROMADB = True
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    _HAS_CHROMADB = False


@dataclass
class VectorHit:
    """One nearest-neighbour result from the vector store."""

    chunk_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    document: str = ""


class VectorStore(abc.ABC):
    """Persistent vector index keyed by chunk id, scoped per paper."""

    @abc.abstractmethod
    def upsert(self, chunks: Sequence[AnyChunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Upsert ``chunks`` with their aligned ``embeddings`` (paper_id in metadata)."""
        raise NotImplementedError

    @abc.abstractmethod
    def query(
        self,
        embedding: Sequence[float],
        *,
        paper_id: Optional[str] = None,
        k: int = 6,
        page_range: Optional[tuple[int, int]] = None,
    ) -> list[VectorHit]:
        """Return the ``k`` nearest chunks, optionally filtered to one paper.

        ``page_range`` is an inclusive ``(page_start, page_end)`` 1-based page
        filter applied on top of ``paper_id`` — the native half of chapter
        scoping (see ``deepvision.rag.chapter_scope``). Every chunk's ``page``
        is stored in metadata by :func:`chunk_to_metadata`, so both backends
        can filter on it. Default ``None`` leaves existing call sites (chat,
        compare, whole-paper reports) unchanged.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def delete_paper(self, paper_id: str) -> None:
        """Delete all vectors belonging to ``paper_id`` (used on delete/rerun)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Chunk <-> flat metadata (de)serialization
#
# Chroma metadata values must be scalar (str/int/float/bool); nested shapes
# (BBox, SourceRef) are flattened on the way in and reconstructed on the way
# out so retrieval can return real, typed `AnyChunk` model instances.
# ---------------------------------------------------------------------------


def chunk_to_metadata(chunk: AnyChunk) -> dict[str, Any]:
    """Flatten a chunk's provenance/location fields into Chroma-safe metadata."""
    meta: dict[str, Any] = {
        "paper_id": chunk.paper_id,
        "modality": chunk.modality.value,
        "provenance": chunk.provenance.value,
        "page": int(chunk.page),
        "ordinal": int(chunk.ordinal),
    }
    if chunk.bbox is not None:
        meta["bbox_x0"] = chunk.bbox.x0
        meta["bbox_y0"] = chunk.bbox.y0
        meta["bbox_x1"] = chunk.bbox.x1
        meta["bbox_y1"] = chunk.bbox.y1
    if chunk.image_path:
        meta["image_path"] = chunk.image_path
    if chunk.source_ref is not None:
        meta["source_section_label"] = chunk.source_ref.section_label
        meta["source_page_label"] = chunk.source_ref.page_label
        meta["source_page"] = int(chunk.source_ref.page)
    if chunk.token_count is not None:
        meta["token_count"] = int(chunk.token_count)

    if isinstance(chunk, OCRChunk):
        if chunk.ocr_confidence is not None:
            meta["ocr_confidence"] = float(chunk.ocr_confidence)
        if chunk.language:
            meta["language"] = chunk.language
    elif isinstance(chunk, VisionInsightChunk):
        if chunk.figure_label:
            meta["figure_label"] = chunk.figure_label
        if chunk.caption:
            meta["caption"] = chunk.caption
    elif isinstance(chunk, ImageChunk):
        if chunk.figure_label:
            meta["figure_label"] = chunk.figure_label
        if chunk.caption:
            meta["caption"] = chunk.caption
        if chunk.thumbnail_path:
            meta["thumbnail_path"] = chunk.thumbnail_path
        if chunk.width is not None:
            meta["width"] = int(chunk.width)
        if chunk.height is not None:
            meta["height"] = int(chunk.height)
    return meta


def _chunk_document(chunk: AnyChunk) -> str:
    """Text stored/indexed for keyword lookup and reconstruction on read."""
    if chunk.text:
        return chunk.text
    if isinstance(chunk, (ImageChunk, VisionInsightChunk)) and chunk.caption:
        return chunk.caption
    return ""


def _chunk_from_id_metadata(
    chunk_id: str, metadata: dict[str, Any], document: str
) -> Optional[AnyChunk]:
    modality = metadata.get("modality")
    if "paper_id" not in metadata or "page" not in metadata:
        log.warning("vector hit missing required metadata", chunk_id=chunk_id)
        return None

    bbox: Optional[BBox] = None
    if all(k in metadata for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")):
        bbox = BBox(
            x0=float(metadata["bbox_x0"]),
            y0=float(metadata["bbox_y0"]),
            x1=float(metadata["bbox_x1"]),
            y1=float(metadata["bbox_y1"]),
        )
    source_ref: Optional[SourceRef] = None
    if all(
        k in metadata
        for k in ("source_section_label", "source_page_label", "source_page")
    ):
        source_ref = SourceRef(
            section_label=str(metadata["source_section_label"]),
            page_label=str(metadata["source_page_label"]),
            page=int(metadata["source_page"]),
        )

    common: dict[str, Any] = dict(
        id=chunk_id,
        paper_id=metadata["paper_id"],
        text=document or "",
        page=int(metadata["page"]),
        bbox=bbox,
        image_path=metadata.get("image_path"),
        source_ref=source_ref,
        ordinal=int(metadata.get("ordinal", 0)),
        token_count=metadata.get("token_count"),
    )

    if modality == Modality.TEXT.value:
        return TextChunk(**common)
    if modality == Modality.OCR.value:
        return OCRChunk(
            **common,
            ocr_confidence=metadata.get("ocr_confidence"),
            language=metadata.get("language"),
        )
    if modality == Modality.VISION.value:
        return VisionInsightChunk(
            **common,
            figure_label=metadata.get("figure_label"),
            caption=metadata.get("caption"),
        )
    if modality == Modality.IMAGE.value:
        # ImageChunk (unlike the text/ocr/vision subtypes) does not pin a
        # provenance default on the model, so it must be supplied explicitly on
        # reconstruction — otherwise pydantic raises and the whole retrieval
        # result is dropped. The value was flattened into metadata on upsert.
        return ImageChunk(
            **common,
            provenance=metadata.get("provenance") or Provenance.VISION.value,
            figure_label=metadata.get("figure_label"),
            caption=metadata.get("caption"),
            thumbnail_path=metadata.get("thumbnail_path"),
            width=metadata.get("width"),
            height=metadata.get("height"),
        )
    log.warning("unknown modality in stored metadata", chunk_id=chunk_id, modality=modality)
    return None


def chunk_from_hit(hit: VectorHit) -> Optional[AnyChunk]:
    """Reconstruct a typed :data:`AnyChunk` from a :class:`VectorHit`."""
    return _chunk_from_id_metadata(hit.chunk_id, hit.metadata, hit.document)


def _meta_in_page_range(
    metadata: dict[str, Any], page_range: tuple[int, int]
) -> bool:
    """True if ``metadata['page']`` falls inside the inclusive ``page_range``.

    A record with no usable ``page`` is treated as **out** of range: chapter
    scoping must never leak a chunk whose location cannot be verified.
    """
    page = metadata.get("page")
    try:
        page_int = int(page)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    start, end = int(page_range[0]), int(page_range[1])
    return start <= page_int <= end


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


class _JsonFallbackIndex:
    """Dependency-free cosine-similarity index, persisted as one JSON file.

    Used only when ``chromadb`` is not installed. Not intended for large
    corpora (linear scan), but keeps the RAG pipeline functional offline.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._records = json.loads(self._path.read_text("utf-8"))
            except Exception:
                log.warning("failed to load fallback vector index; starting fresh", path=str(self._path))
                self._records = {}

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._records), encoding="utf-8")
        tmp.replace(self._path)

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        with self._lock:
            for cid, emb, meta, doc in zip(ids, embeddings, metadatas, documents):
                self._records[cid] = {
                    "embedding": list(emb),
                    "metadata": meta,
                    "document": doc,
                }
            self._save()

    def query(
        self,
        embedding: Sequence[float],
        *,
        paper_id: Optional[str],
        k: int,
        page_range: Optional[tuple[int, int]] = None,
    ) -> list[VectorHit]:
        with self._lock:
            candidates = list(self._records.items())
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for cid, rec in candidates:
            meta = rec.get("metadata", {})
            if paper_id is not None and meta.get("paper_id") != paper_id:
                continue
            if page_range is not None and not _meta_in_page_range(meta, page_range):
                continue
            score = _cosine(embedding, rec["embedding"])
            scored.append((score, cid, rec))
        scored.sort(key=lambda t: t[0], reverse=True)
        hits = [
            VectorHit(chunk_id=cid, score=score, metadata=rec.get("metadata", {}), document=rec.get("document", ""))
            for score, cid, rec in scored[:k]
        ]
        return hits

    def delete_paper(self, paper_id: str) -> None:
        with self._lock:
            self._records = {
                cid: rec
                for cid, rec in self._records.items()
                if rec.get("metadata", {}).get("paper_id") != paper_id
            }
            self._save()


class ChromaVectorStore(VectorStore):
    """Persistent ChromaDB-backed vector store.

    A single shared collection is used *per embedding dimension*
    (``chunks_dim{dim}``), filtered by ``paper_id`` metadata at query/delete
    time. This keeps the collection consistent with whichever embedding
    provider produced the vectors: switching ``embedding_mode``/model changes
    ``dim``, which routes to a different (empty) collection — the documented
    "switching embedding provider requires re-ingest" policy falls out of this
    naturally rather than needing an explicit migration step.

    Falls back to a JSON-persisted cosine index (see ``_JsonFallbackIndex``)
    when ``chromadb`` is not importable, so the module and its callers never
    crash for lack of the dependency.
    """

    def __init__(
        self,
        persist_dir: str | Path,
        dim: int,
        *,
        collection_prefix: str = "chunks",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.dim = dim
        self._collection_name = f"{collection_prefix}_dim{dim}"
        self._collection = None
        self._fallback: Optional[_JsonFallbackIndex] = None

        if _HAS_CHROMADB:
            try:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(self.persist_dir))
                self._collection = client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine", "embedding_dim": dim},
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.error(
                    "failed to open chromadb collection; falling back to JSON index",
                    error=str(exc),
                )
                self._collection = None

        if self._collection is None:
            fallback_path = self.persist_dir / f"{self._collection_name}.json"
            self._fallback = _JsonFallbackIndex(fallback_path)
            if not _HAS_CHROMADB:
                log.warning(
                    "chromadb not installed; using JSON fallback vector index",
                    path=str(fallback_path),
                )

    # -- VectorStore ---------------------------------------------------

    def upsert(self, chunks: Sequence[AnyChunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )
        if not chunks:
            return

        ids = [c.id for c in chunks]
        metadatas = [chunk_to_metadata(c) for c in chunks]
        documents = [_chunk_document(c) for c in chunks]
        vectors = [list(e) for e in embeddings]

        if self._collection is not None:
            try:
                self._collection.upsert(
                    ids=ids, embeddings=vectors, metadatas=metadatas, documents=documents
                )
                return
            except Exception as exc:  # pragma: no cover - defensive
                log.error("chromadb upsert failed", error=str(exc))
                raise
        assert self._fallback is not None
        self._fallback.upsert(ids, vectors, metadatas, documents)

    def query(
        self,
        embedding: Sequence[float],
        *,
        paper_id: Optional[str] = None,
        k: int = 6,
        page_range: Optional[tuple[int, int]] = None,
    ) -> list[VectorHit]:
        if self._collection is not None:
            try:
                hits = self._chroma_query(
                    embedding, paper_id=paper_id, k=k, page_range=page_range
                )
            except Exception as exc:  # pragma: no cover - defensive
                if page_range is None:
                    log.error("chromadb query failed", error=str(exc))
                    return []
                # The page predicate may be rejected by an older Chroma
                # metadata-filter dialect. Degrade to the paper-only filter and
                # let the Python-side page filter below do the scoping rather
                # than returning nothing at all.
                log.warning(
                    "chromadb page-range query failed; retrying unfiltered and "
                    "filtering pages in-process",
                    error=str(exc),
                )
                try:
                    hits = self._chroma_query(
                        embedding, paper_id=paper_id, k=k, page_range=None
                    )
                except Exception as exc2:  # pragma: no cover - defensive
                    log.error("chromadb query failed", error=str(exc2))
                    return []
            if page_range is not None:
                # Belt and braces: whatever the backend did with `where`, no
                # out-of-range chunk leaves this method when a page range was
                # requested.
                hits = [h for h in hits if _meta_in_page_range(h.metadata, page_range)]
            return hits

        assert self._fallback is not None
        return self._fallback.query(
            embedding, paper_id=paper_id, k=k, page_range=page_range
        )

    def _chroma_query(
        self,
        embedding: Sequence[float],
        *,
        paper_id: Optional[str],
        k: int,
        page_range: Optional[tuple[int, int]],
    ) -> list[VectorHit]:
        """One raw Chroma nearest-neighbour call, translated to ``VectorHit``s."""
        assert self._collection is not None
        if page_range is None:
            # Unchanged from the pre-page_range behaviour, byte for byte.
            where = {"paper_id": paper_id} if paper_id is not None else None
        else:
            clauses: list[dict[str, Any]] = []
            if paper_id is not None:
                clauses.append({"paper_id": {"$eq": paper_id}})
            clauses.append({"page": {"$gte": int(page_range[0])}})
            clauses.append({"page": {"$lte": int(page_range[1])}})
            where = {"$and": clauses}
        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=max(k, 1),
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[VectorHit] = []
        for cid, meta, doc, dist in zip(ids, metadatas, documents, distances):
            score = max(0.0, 1.0 - float(dist))
            hits.append(
                VectorHit(
                    chunk_id=cid,
                    score=score,
                    metadata=dict(meta or {}),
                    document=doc or "",
                )
            )
        return hits

    def delete_paper(self, paper_id: str) -> None:
        if self._collection is not None:
            try:
                self._collection.delete(where={"paper_id": paper_id})
                return
            except Exception as exc:  # pragma: no cover - defensive
                log.error("chromadb delete_paper failed", error=str(exc))
                raise
        assert self._fallback is not None
        self._fallback.delete_paper(paper_id)


def open_vector_store(
    dim: int, *, chroma_dir: Optional[str | Path] = None
) -> ChromaVectorStore:
    """Convenience constructor: open the ChromaDB store at ``config.chroma_dir``.

    ``dim`` must match the active :class:`EmbeddingProvider`'s ``dim`` (the
    collection is keyed by it). Pass ``chroma_dir`` explicitly to override the
    process config (e.g. in tests).
    """
    if chroma_dir is None:
        from deepvision.config import get_config

        chroma_dir = get_config().chroma_dir
    return ChromaVectorStore(chroma_dir, dim)
