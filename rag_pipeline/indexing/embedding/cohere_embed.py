"""
Cohere embedding adapter for the indexing pipeline.

Wraps Cohere ``embed-multilingual-v3.0`` (1024-dim) as a LlamaIndex
``BaseEmbedding``. Used by ``embed_and_store`` for document vectors and by
``retrieval`` for query vectors.

Requires ``COHERE_API_KEY`` in the project-root ``.env``.
"""

from __future__ import annotations

import os
from typing import Any, List

from dotenv import find_dotenv, load_dotenv
from llama_index.core.base.embeddings.base import BaseEmbedding

load_dotenv(find_dotenv(usecwd=True), override=False)

COHERE_MODEL = "embed-multilingual-v3.0"
COHERE_DIM = 1024


class CohereEmbedding(BaseEmbedding):
    """Cohere ``embed-multilingual-v3.0`` for LlamaIndex.

    Indexing uses ``search_document``; queries use ``search_query`` so cosine
    similarity at retrieval time is well-calibrated.
    """

    _model_name: str = COHERE_MODEL
    _client: Any = None

    def __init__(
        self,
        model_name: str = COHERE_MODEL,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        try:
            import cohere
        except ImportError as e:
            raise ImportError(
                "cohere package not installed. Run `pip install cohere`."
            ) from e
        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY not set in environment.")
        object.__setattr__(self, "_client", cohere.Client(key))
        object.__setattr__(self, "_model_name", model_name)

    @classmethod
    def class_name(cls) -> str:
        return "CohereEmbedding"

    def _embed_one(self, text: str, input_type: str) -> List[float]:
        resp = self._client.embed(
            texts=[text], model=self._model_name, input_type=input_type
        )
        vecs = getattr(resp, "embeddings", None) or resp.get("embeddings")
        return list(vecs[0])

    def _embed_many(self, texts: List[str], input_type: str) -> List[List[float]]:
        resp = self._client.embed(
            texts=list(texts), model=self._model_name, input_type=input_type
        )
        vecs = getattr(resp, "embeddings", None) or resp.get("embeddings")
        return [list(v) for v in vecs]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_one(query, input_type="search_query")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_one(text, input_type="search_document")

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            out.extend(self._embed_many(batch, input_type="search_document"))
        return out


def make_cohere_embed_model(
    model_name: str = COHERE_MODEL,
) -> CohereEmbedding:
    """Return a configured Cohere embedding model."""
    return CohereEmbedding(model_name=model_name)
