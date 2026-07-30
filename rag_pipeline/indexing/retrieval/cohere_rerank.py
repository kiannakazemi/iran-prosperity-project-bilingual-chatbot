"""
Cohere reranker adapter for the retrieval pipeline.

Wraps Cohere ``rerank-v3.5`` (multilingual: English + Persian) so a dense
candidate pool can be reordered by query-document relevance before the final
``top_k`` cut. Relevance scores are calibrated 0-1.

Requires ``COHERE_API_KEY`` in the project-root ``.env``.
"""

from __future__ import annotations

import os
from typing import Any, List, Tuple

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)

COHERE_RERANK_MODEL = "rerank-v3.5"


class CohereReranker:
    """Thin wrapper around Cohere ``rerank-v3.5`` (multilingual)."""

    def __init__(self, model_name: str = COHERE_RERANK_MODEL, api_key: str | None = None):
        try:
            import cohere
        except ImportError as e:
            raise ImportError(
                "cohere package not installed. Run `pip install cohere`."
            ) from e
        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY not set in environment.")
        self._client: Any = cohere.Client(key)
        self._model_name = model_name

    def rerank(
        self, query: str, documents: List[str], top_n: int
    ) -> List[Tuple[int, float]]:
        """Return ``[(original_index, relevance_score), …]`` ordered best-first.

        ``original_index`` points back into the input ``documents`` list so the
        caller can recover the full hit object for each reranked result.
        """
        if not documents:
            return []
        resp = self._client.rerank(
            model=self._model_name,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )
        results = getattr(resp, "results", None) or resp.get("results")
        return [(r.index, float(r.relevance_score)) for r in results]


def make_cohere_reranker(model_name: str = COHERE_RERANK_MODEL) -> CohereReranker:
    """Build and return a Cohere reranker."""
    return CohereReranker(model_name=model_name)
