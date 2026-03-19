"""
reranker.py
-----------
Cross-encoder reranker for the hybrid retrieval pipeline.

Role in the pipeline
~~~~~~~~~~~~~~~~~~~~
The hybrid search (dense ANN + sparse IDF, fused with RRF) is a fast
*bi-encoder* system: query and documents are embedded independently and
matched by vector similarity.  That trade-off favours recall over
precision.

A *cross-encoder* sees the query and the full document text concatenated,
so it can model fine-grained query–document interactions that bi-encoders
miss.  The cost is that it must score every candidate pair individually
(O(N) forward passes), so it is applied only on the small shortlist
returned by the fast retriever, not on the full collection.

Integration
~~~~~~~~~~~
``CrossEncoderReranker`` is an optional component of
``HybridQdrantRetriever``.  When present:

1.  ``hybrid_search`` is called with ``top_k * rerank_top_k_multiplier``
    to fetch a wider candidate pool (e.g. 24 candidates for top_k=6).
2.  The reranker scores all candidates and re-orders them by
    cross-encoder score.
3.  The retriever returns the top ``top_k`` documents from the
    reranked list, continuing to the existing deduplication step.

Degrading gracefully
~~~~~~~~~~~~~~~~~~~~
If ``sentence_transformers`` is not installed (e.g. in the CPU Docker
image used with EMBEDDER_MODE=hf), the reranker falls back to a no-op
that preserves the original RRF order and logs a one-time warning.
Callers never need to handle an ImportError.

Model choices
~~~~~~~~~~~~~
``cross-encoder/ms-marco-MiniLM-L-6-v2``  — default; ~22 MB, fast, good
``cross-encoder/ms-marco-MiniLM-L-12-v2`` — ~33 MB, ~15 % better on BEIR
``cross-encoder/ms-marco-electra-base``   — ~110 MB, best quality

All three are trained on MS MARCO passage ranking and transfer well to
technical documentation retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Default model: lightweight, ships in ~22 MB, runs in < 1 s on CPU for
# batches of 24 passages up to 512 tokens.
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Log the unavailability warning at most once per process.
_WARNED_UNAVAILABLE = False


class CrossEncoderReranker:
    """
    Rerank a candidate list using a cross-encoder model.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Defaults to
        ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
    top_k_multiplier:
        How many times more candidates to request from the bi-encoder
        retriever than the final ``top_k``.  For example, with
        ``top_k=6`` and ``top_k_multiplier=4`` the retriever fetches 24
        candidates which the cross-encoder then re-orders.
        Higher values improve recall at the cost of reranking latency.
    max_length:
        Token truncation limit for the cross-encoder.  512 matches the
        BGE-M3 dense encoder and is safe for all three recommended models.
    batch_size:
        Number of (query, passage) pairs scored in one forward pass.
        32 fits comfortably on CPU; increase to 64+ on GPU.
    device:
        ``"cuda"``, ``"cpu"``, or ``None`` for automatic selection.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        top_k_multiplier: int = 4,
        max_length: int = 512,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.top_k_multiplier = top_k_multiplier
        self._model: Any | None = None
        self._available = False

        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            import torch as _torch
            _device = device or ("cuda" if _torch.cuda.is_available() else "cpu")

            log.info(
                "[Reranker] Loading cross-encoder | model=%s | device=%s",
                model_name, _device,
            )
            self._model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=_device,
            )
            self._batch_size = batch_size
            self._available = True
            log.info("[Reranker] Cross-encoder ready | model=%s", model_name)

        except ImportError:
            global _WARNED_UNAVAILABLE
            if not _WARNED_UNAVAILABLE:
                log.warning(
                    "[Reranker] sentence-transformers is not installed — "
                    "reranker will be a no-op.  "
                    "Install it with:  pip install sentence-transformers"
                )
                _WARNED_UNAVAILABLE = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True when the cross-encoder model loaded successfully."""
        return self._available

    def rerank(
        self,
        query: str,
        candidates: list[tuple[dict, float]],
    ) -> list[tuple[dict, float]]:
        """
        Re-score and re-order *candidates* using the cross-encoder.

        When the model is unavailable (``available == False``) the
        original *candidates* list is returned unchanged, preserving
        the RRF ordering as a fallback.

        Parameters
        ----------
        query:
            The original natural-language query string.
        candidates:
            List of ``(payload_dict, rrf_score)`` tuples as returned by
            ``QdrantDocStore.hybrid_search()``.

        Returns
        -------
        list[tuple[dict, float]]
            Same structure as the input but re-ordered by cross-encoder
            score (descending), with the cross-encoder score replacing
            the original RRF score so that downstream logging stays
            meaningful.
        """
        if not self._available or not candidates:
            return candidates

        texts = [payload.get("text", "") for payload, _ in candidates]
        pairs = [[query, text] for text in texts]

        log.info(
            "[Reranker] Scoring %d candidate(s) | query=%r",
            len(pairs), query,
        )

        scores: list[float] = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        result = [(payload, ce_score) for (payload, _rrf), ce_score in reranked]

        log.info(
            "[Reranker] Reranked %d candidates | top score=%.4f | bottom score=%.4f",
            len(result),
            result[0][1] if result else 0.0,
            result[-1][1] if result else 0.0,
        )
        for rank, (payload, score) in enumerate(result[:6], 1):
            log.info(
                "[Reranker]   [%d] ce_score=%.4f  kind=%-12s  symbol=%s",
                rank, score,
                payload.get("kind", "?"),
                payload.get("symbol") or "—",
            )

        return result
