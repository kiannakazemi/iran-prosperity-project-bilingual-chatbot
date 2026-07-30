"""Retrieval stage of the RAG pipeline.

Re-exports the public retrieval API from ``pipeline`` so existing imports
(``from indexing.retrieval import ReliableRetriever``) keep working after the
move into a package.
"""

from indexing.retrieval.pipeline import (
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBED_MODEL,
    HEADER_ROUTER_MAX_CANDIDATES,
    HEADER_ROUTER_MAX_SELECTIONS,
    RERANK_CANDIDATE_POOL,
    ReliableRetriever,
    RetrievalResult,
)

__all__ = [
    "COLLECTION_NAME",
    "DENSE_TOP_K",
    "EMBED_MODEL",
    "HEADER_ROUTER_MAX_CANDIDATES",
    "HEADER_ROUTER_MAX_SELECTIONS",
    "RERANK_CANDIDATE_POOL",
    "ReliableRetriever",
    "RetrievalResult",
]
