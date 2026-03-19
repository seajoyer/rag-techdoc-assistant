"""
lc_adapter.py
-------------
LangChain ``Embeddings`` adapter over the project's embedder backends.

Both ``BGEM3Embedder`` (local) and ``HFInferenceEmbedder`` (API) share
the same ``embed(texts: list[str]) -> np.ndarray`` interface, so a single
thin adapter covers both without any conditionals.

Usage
~~~~~
    from src.embedding.lc_adapter import ProjectEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # works with either embedder backend
    lc_emb = ProjectEmbeddings(embedder)
    ragas_emb = LangchainEmbeddingsWrapper(lc_emb)

    # or directly in any LangChain chain / RAGAS metric
    metric = AnswerRelevancy(llm=judge_llm, embeddings=ragas_emb)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)


class ProjectEmbeddings(Embeddings):
    """
    LangChain ``Embeddings`` shim over ``BGEM3Embedder`` / ``HFInferenceEmbedder``.

    Both backends expose ``embed(texts: list[str]) -> np.ndarray`` returning
    a float32 array of shape ``(N, 1024)``.  This adapter converts that to
    the ``list[list[float]]`` contract that LangChain (and RAGAS) expect.

    Parameters
    ----------
    embedder:
        Any object that implements ``embed(texts: list[str]) -> np.ndarray``.
        Concretely: ``BGEM3Embedder`` or ``HFInferenceEmbedder``.
    """

    def __init__(self, embedder: object) -> None:
        self._emb = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents; returns list of float vectors."""
        log.debug("ProjectEmbeddings.embed_documents | n=%d", len(texts))
        return self._emb.embed(texts).tolist()  # type: ignore[union-attr]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string; returns a float vector."""
        log.debug("ProjectEmbeddings.embed_query | text=%r", text[:60])
        return self._emb.embed([text])[0].tolist()  # type: ignore[union-attr]
