"""
store.py
--------
Qdrant Cloud document store with hybrid dense + sparse retrieval.

Collection layout
~~~~~~~~~~~~~~~~~
Named dense vector:

    "dense"    → 1 024-dim float32, Distance.COSINE
                 Source: BAAI/bge-m3 via HuggingFace Inference

Named sparse vector:

    "keywords" → SparseVectorParams(modifier=Modifier.IDF)
                 Source: Chunk.keywords via feature-hashed raw-TF vectors
                 Qdrant applies IDF normalisation to queries at search time.

Payload (every field of Chunk.to_dict(), including ``text``):

    chunk_id, text, page_url, citation_url, anchor, page_title,
    section, kind, symbol, keywords, params, source_url,
    char_count, is_continuation, sub_index

Storing ``text`` in the payload keeps results self-contained — no secondary
disk read is needed to build the LangChain ``Document`` on retrieval.

Hybrid search with RRF fusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``hybrid_search()`` uses the Qdrant Query API (>= 1.9):

    prefetch[0]  → dense ANN  (top k*3 candidates)
    prefetch[1]  → sparse     (top k*3 candidates)
    query        → FusionQuery(fusion=Fusion.RRF)  (re-ranks with RRF)
    limit        → top_k final results

Reciprocal Rank Fusion is parameter-free and robust across different
vector score distributions, making it a safe default for hybrid retrieval.

LangChain integration
~~~~~~~~~~~~~~~~~~~~~
``QdrantDocStore.as_retriever()`` returns a ``HybridQdrantRetriever``
(``langchain_core.BaseRetriever`` subclass) ready to plug into any
LangChain LCEL chain::

    retriever = store.as_retriever(top_k=6)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

Each retrieved ``Document`` carries the full chunk payload in ``metadata``,
so citation URLs, symbol names, and section info are directly available to
the answer-formatting prompt.

Filtering
~~~~~~~~~
Both ``hybrid_search()`` and ``as_retriever()`` accept an optional
``filter_`` argument (a ``qdrant_client.models.Filter``) for section-
or kind-scoped retrieval::

    from qdrant_client import models
    api_only = models.Filter(
        must=[models.FieldCondition(
            key="kind",
            match=models.MatchExcept(**{"except": ["heading"]}),
        )]
    )
    docs = store.hybrid_search("DataLoader", filter_=api_only)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Sequence

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector as QdrantSparseVector,
    SparseVectorParams,
    VectorParams,
)

try:
    from ..chunking.chunker import Chunk
    from ..embedding.embedder import BGEM3Embedder
    from ..embedding.sparse import keywords_to_sparse
except ImportError:
    from chunking.chunker import Chunk  # type: ignore[no-redef]
    from embedding.embedder import BGEM3Embedder  # type: ignore[no-redef]
    from embedding.sparse import keywords_to_sparse  # type: ignore[no-redef]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "keywords"
DENSE_DIM = 1024


# ---------------------------------------------------------------------------
# Document store
# ---------------------------------------------------------------------------

class QdrantDocStore:
    """
    Hybrid Qdrant collection: dense (BGE-M3) + sparse (keyword IDF).

    Parameters
    ----------
    client:
        Connected ``QdrantClient`` (cloud or local).
    collection_name:
        Qdrant collection to use / create.
    embedder:
        ``BGEM3Embedder`` instance used for query-time dense encoding.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedder: BGEM3Embedder,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection with hybrid vector config.

        Parameters
        ----------
        recreate:
            Drop an existing collection before creating.  Use with care
            in production — this deletes all indexed points.
        """
        if recreate and self.client.collection_exists(self.collection_name):
            log.warning("Dropping existing collection %r", self.collection_name)
            self.client.delete_collection(self.collection_name)

        if self.client.collection_exists(self.collection_name):
            log.info("Collection %r already exists — skipping creation.", self.collection_name)
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR: VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                    on_disk=True,           # mmap for large collections
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: SparseVectorParams(
                    # IDF normalisation is applied to query vectors at search
                    # time.  Document vectors carry raw TF weights.
                    modifier=models.Modifier.IDF,
                )
            },
            # Optimiser / indexing settings tuned for a ~50 k-point collection
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=10_000,  # build HNSW after 10k points
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10_000,
            ),
        )
        log.info("Created collection %r", self.collection_name)

    def collection_info(self) -> dict:
        """Return a plain dict summarising collection stats."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
        }

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: np.ndarray,
        batch_size: int = 128,
    ) -> None:
        """
        Upsert ``chunks`` with their pre-computed ``dense_vectors``.

        Sparse vectors are built on-the-fly from ``chunk.keywords``.
        Upsert is idempotent — re-running with the same chunks overwrites
        existing points (matched by deterministic UUID derived from
        ``chunk.chunk_id``).

        Parameters
        ----------
        chunks:
            Ordered sequence of ``Chunk`` objects.
        dense_vectors:
            Float32 array of shape ``(len(chunks), 1024)``.
        batch_size:
            Points per Qdrant upsert call.  128 works well under the
            Qdrant Cloud free-tier gRPC payload limit.
        """
        if len(chunks) != len(dense_vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and dense_vectors ({len(dense_vectors)}) "
                "must have the same length."
            )

        total = len(chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_chunks = chunks[start:end]
            batch_vecs = dense_vectors[start:end]
            points = [
                _chunk_to_point(chunk, vec)
                for chunk, vec in zip(batch_chunks, batch_vecs)
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            log.info("Upserted %d–%d / %d points", start, end - 1, total)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = 6,
        filter_: models.Filter | None = None,
        prefetch_multiplier: int = 3,
    ) -> list[tuple[dict, float]]:
        """
        Hybrid dense + sparse search with RRF fusion.

        The query is embedded with BGE-M3 for the dense leg, and tokenised
        by whitespace with the same feature hash for the sparse leg.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Number of results to return after fusion.
        filter_:
            Optional Qdrant payload filter applied to both legs.
        prefetch_multiplier:
            Each leg fetches ``top_k * prefetch_multiplier`` candidates
            before fusion.

        Returns
        -------
        list of (payload_dict, rrf_score) tuples, highest score first.
        """
        # --- Dense query embedding ----------------------------------------
        q_dense = self.embedder.embed([query])[0].tolist()

        # --- Sparse query vector from whitespace tokens -------------------
        # Simple tokenisation is intentional: the sparse leg is a recall
        # booster for exact keyword matches, not a replacement for the dense
        # semantic leg.  More sophisticated tokenisation (sub-word, stemming)
        # rarely improves RRF results in practice for technical documentation.
        q_tokens = [t.strip(".,;:()[]\"'") for t in query.lower().split() if t]
        q_sparse_sv = keywords_to_sparse(q_tokens)

        candidate_limit = top_k * prefetch_multiplier

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=q_dense,
                    using=DENSE_VECTOR,
                    limit=candidate_limit,
                    filter=filter_,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=q_sparse_sv.indices,
                        values=q_sparse_sv.values,
                    ),
                    using=SPARSE_VECTOR,
                    limit=candidate_limit,
                    filter=filter_,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        return [(pt.payload, pt.score) for pt in results.points]

    # ------------------------------------------------------------------
    # LangChain integration
    # ------------------------------------------------------------------

    def as_retriever(
        self,
        top_k: int = 6,
        filter_: models.Filter | None = None,
        score_threshold: float | None = None,
    ) -> "HybridQdrantRetriever":
        """
        Return a LangChain ``BaseRetriever`` backed by this store.

        Parameters
        ----------
        top_k:
            Maximum number of documents returned per query.
        filter_:
            Optional Qdrant payload filter (section, kind, symbol, etc.).
        score_threshold:
            Drop results with RRF score below this value.  None = keep all.

        Returns
        -------
        HybridQdrantRetriever
            Ready to use in any LangChain LCEL chain.
        """
        return HybridQdrantRetriever(
            store=self,
            top_k=top_k,
            filter_=filter_,
            score_threshold=score_threshold,
        )


# ---------------------------------------------------------------------------
# LangChain retriever
# ---------------------------------------------------------------------------

class HybridQdrantRetriever(BaseRetriever):
    """
    LangChain ``BaseRetriever`` backed by ``QdrantDocStore.hybrid_search()``.

    Each ``Document`` returned has:

    * ``page_content`` — the chunk's Markdown text (ready for LLM context)
    * ``metadata``     — all other chunk fields plus ``score``

    The ``citation_url`` and ``symbol`` fields in metadata make it easy to
    build source-attributed answers in the downstream LLM chain.
    """

    store: Any = Field(repr=False)
    top_k: int = 6
    filter_: Any = Field(default=None, repr=False)
    score_threshold: float | None = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = self.store.hybrid_search(
            query=query,
            top_k=self.top_k,
            filter_=self.filter_,
        )
        docs: list[Document] = []
        for payload, score in results:
            if self.score_threshold is not None and score < self.score_threshold:
                continue
            text = payload.get("text", "")
            meta = {k: v for k, v in payload.items() if k != "text"}
            meta["score"] = score
            docs.append(Document(page_content=text, metadata=meta))
        return docs


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _chunk_to_point(chunk: Chunk, dense_vec: np.ndarray) -> PointStruct:
    """Build a Qdrant ``PointStruct`` from a chunk and its dense vector."""
    sparse_sv = keywords_to_sparse(chunk.keywords)

    # Deterministic UUID so re-runs are idempotent (upsert overwrites).
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))

    return PointStruct(
        id=point_id,
        vector={
            DENSE_VECTOR: dense_vec.tolist(),
            SPARSE_VECTOR: QdrantSparseVector(
                indices=sparse_sv.indices,
                values=sparse_sv.values,
            ),
        },
        payload=chunk.to_dict(),  # includes text, citation_url, kind, etc.
    )


def show_results(store, query: str, top_k: int = 4) -> None:
    results = store.hybrid_search(query, top_k=top_k)
    print(f"Query: {query!r}\n")
    print(f"{'#':<3} {'Score':>6}  {'Kind':<12} {'Symbol / Title':<35} Citation")
    print("─" * 110)
    for i, (payload, score) in enumerate(results, 1):
        label = payload.get("symbol") or payload.get("page_title", "—")
        print(
            f"{i:<3} {score:>6.4f}  "
            f"{payload.get('kind', '?'):<12} "
            f"{label[:45]:<35} "
            f"{payload.get('citation_url', '')}"
        )
    print()
