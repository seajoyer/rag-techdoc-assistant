"""
reranker.py
-----------
Cross-encoder reranker with RRF-score fusion for hybrid retrieval.

Design
~~~~~~
After the Qdrant hybrid search produces an RRF-ranked candidate list, the
cross-encoder rescores each (query, passage) pair with a full attention pass.
The two signals are then blended:

    final = α · ce_norm + (1 − α) · rrf_norm

where both scores are min-max normalised to [0, 1] within the batch before
blending.  Neither signal fully overrides the other:

    dense / sparse / RRF  -> strong at exact symbol matching and recall
    cross-encoder         -> strong at semantic relevance and precision

Typical alpha values
~~~~~~~~~~~~~~~~~~~~
    α = 0.0   pure RRF  (CE is a no-op; model is never loaded)
    α = 0.5   equal weight
    α = 0.7   CE-leaning — good default for prose questions
    α = 1.0   pure CE  (RRF score ignored after retrieval)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from langchain_core.documents import Document

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_ALPHA = 0.7


# ---------------------------------------------------------------------------
# Score normalisation helpers
# ---------------------------------------------------------------------------


def _minmax_norm(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalise *scores* to [0, 1].

    When all scores are identical (including the degenerate single-item case)
    every element is mapped to 0.5 so the blending formula stays neutral
    rather than producing NaN or all-zeros.
    """
    lo, hi = float(scores.min()), float(scores.max())
    span = hi - lo
    if span < 1e-9:
        return np.full_like(scores, 0.5, dtype=np.float32)
    return ((scores - lo) / span).astype(np.float32)


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """
    Cross-encoder reranker that blends CE relevance with RRF retrieval scores.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model ID or local path.
        Defaults to ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
    max_length:
        Token truncation for the concatenated [query; passage] input.
    device:
        ``"cuda"``, ``"cpu"``, or ``None`` for auto-select.
    default_alpha:
        Default blend weight when callers do not pass one explicitly.
        ``1.0`` = pure CE; ``0.0`` = pure RRF; ``0.7`` = CE-leaning.
    """

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL,
        max_length: int = 512,
        device: str | None = None,
        default_alpha: float = _DEFAULT_ALPHA,
    ) -> None:
        self._model_name   = model_name_or_path
        self._max_length   = max_length
        self._device       = device     # None -> auto-selected at load time
        self.default_alpha = default_alpha
        self._model        = None       # lazy — loaded on first rerank() call

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the CrossEncoder model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install the reranker extra:  uv sync --extra reranker"
            ) from exc

        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"

        self._model = CrossEncoder(
            self._model_name,
            max_length=self._max_length,
            device=self._device,
        )
        log.info(
            "CrossEncoderReranker ready — model: %s | device: %s | max_length: %d",
            self._model_name, self._device, self._max_length,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        docs: list["Document"],
        rrf_scores: list[float],
        alpha: float | None = None,
    ) -> list["Document"]:
        """
        Rerank *docs* by blending cross-encoder relevance with RRF scores.

        Score metadata written to each document
        ----------------------------------------
        ``score``            — final blended score used for ranking downstream
        ``rrf_score``        — original (unnormalised) RRF score from Qdrant
        ``rrf_score_norm``   — RRF score normalised to [0, 1]
        ``ce_score_raw``     — raw CE logit (useful for threshold tuning)
        ``ce_score``         — CE score normalised to [0, 1]

        ``score`` is overwritten in-place so all downstream code (Telegram
        bot, ``show_results``, evaluation notebooks) keeps working unchanged.

        Parameters
        ----------
        query:
            The original user question — not the HyDE snippet.  The CE sees
            the same text the user typed for faithful relevance estimation.
        docs:
            Candidate documents in RRF rank order.
        rrf_scores:
            Parallel list of RRF scores corresponding to *docs*.
        alpha:
            Blend weight for the CE signal.  Falls back to
            ``self.default_alpha`` when *None*.

        Returns
        -------
        list[Document]
            The same documents, reordered by descending blended score.
        """
        if not docs:
            return docs

        alpha = self.default_alpha if alpha is None else float(alpha)

        # alpha == 0 -> pure RRF order; stamp metadata and return early.
        if alpha == 0.0:
            for doc, rrf in zip(docs, rrf_scores):
                doc.metadata["rrf_score"] = float(rrf)
            log.info("[Reranker] alpha=0.0 — skipping CE, returning RRF order as-is")
            return docs

        self._ensure_model()

        # ── Cross-encoder scoring ─────────────────────────────────────────
        pairs  = [[query, doc.page_content] for doc in docs]

        log.info(
            "[Reranker] Scoring %d pairs | model=%s | alpha=%.2f",
            len(pairs), self._model_name, alpha,
        )

        # CrossEncoder.predict() accepts a list of [query, passage] pairs and
        # returns a numpy array of raw logits.
        raw_ce_arr: np.ndarray = self._model.predict(pairs, show_progress_bar=False)
        raw_ce = raw_ce_arr.tolist()

        log.info(
            "[Reranker] CE raw scores | n=%d  min=%.3f  max=%.3f  mean=%.3f",
            len(raw_ce), min(raw_ce), max(raw_ce),
            sum(raw_ce) / len(raw_ce),
        )

        # ── Normalise both vectors to [0, 1] ──────────────────────────────
        ce_norm  = _minmax_norm(np.array(raw_ce,     dtype=np.float32))
        rrf_norm = _minmax_norm(np.array(rrf_scores, dtype=np.float32))

        # ── Blend ─────────────────────────────────────────────────────────
        blended = alpha * ce_norm + (1.0 - alpha) * rrf_norm

        # ── Write scores back to metadata ─────────────────────────────────
        for doc, rrf_raw, ce_raw, ce_n, rrf_n, blend in zip(
            docs, rrf_scores, raw_ce, ce_norm, rrf_norm, blended
        ):
            doc.metadata.update(
                rrf_score      = float(rrf_raw),
                rrf_score_norm = float(rrf_n),
                ce_score_raw   = float(ce_raw),
                ce_score       = float(ce_n),
                score          = float(blend),   # replaces pre-rerank RRF score
            )

        reranked = sorted(docs, key=lambda d: d.metadata["score"], reverse=True)

        log.info("[Reranker] Final ranking (alpha=%.2f):", alpha)
        for rank, doc in enumerate(reranked, 1):
            m = doc.metadata
            log.info(
                "[Reranker]   [%d] blend=%.4f  ce=%.4f  rrf=%.4f  symbol=%s",
                rank,
                m["score"],
                m["ce_score"],
                m["rrf_score_norm"],
                m.get("symbol") or "—",
            )

        return reranked
