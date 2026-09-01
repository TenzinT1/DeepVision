"""RAG domain — chunking, vector storage, and retrieval. Handles the Embedding pipeline stage (chunk -> embed -> upsert
into ChromaDB) and query-time retrieval for chat/agents.

- ``Chunker`` / ``VectorStore`` / ``Retriever``: the frozen ABC interfaces
  (signatures fixed by).
- ``SlidingWindowChunker`` / ``ChromaVectorStore`` / ``ChunkRetriever``: this
  domain's concrete implementations.
- ``embed_chunks``: additive glue for the Embedding pipeline stage (see
  ``embedding_pipeline.py`` docstring).
"""

from deepvision.rag.chunker import Chunker, SlidingWindowChunker, count_tokens
from deepvision.rag.embedding_pipeline import embed_chunks
from deepvision.rag.retrieval import ChunkRetriever, Retriever
from deepvision.rag.vector_store import (
    ChromaVectorStore,
    VectorHit,
    VectorStore,
    chunk_from_hit,
    chunk_to_metadata,
    open_vector_store,
)

__all__ = [
    "Chunker",
    "SlidingWindowChunker",
    "count_tokens",
    "VectorStore",
    "ChromaVectorStore",
    "VectorHit",
    "chunk_to_metadata",
    "chunk_from_hit",
    "open_vector_store",
    "Retriever",
    "ChunkRetriever",
    "embed_chunks",
]
