"""
embedder.py
-----------
Dense text embeddings via BAAI/bge-m3 on HuggingFace Inference API.

BGE-M3 produces 1 024-dimensional, L2-normalised dense vectors that work
with cosine similarity.  The HuggingFace feature-extraction endpoint may
return either shape (N, dim) or (N, seq_len, dim); the latter requires CLS
pooling (index 0 along the sequence axis) before L2 normalisation.

Rate-limiting, exponential-backoff retry, and request batching are built in
so the embedder can be called directly from a notebook loop without extra
wrapper code.

Usage
~~~~~
::

    from src.embedding.embedder import BGEM3Embedder

    embedder = BGEM3Embedder(api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"])
    vectors = embedder.embed(["What is torch.autograd?", "How do I use DataLoader?"])
    # vectors.shape → (2, 1024)
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
from huggingface_hub import InferenceClient

log = logging.getLogger(__name__)


class BGEM3Embedder:
    """
    Dense text embedder backed by ``BAAI/bge-m3`` on HuggingFace Inference.

    Parameters
    ----------
    api_token:
        HuggingFace API token (a read-scope token is sufficient).
    batch_size:
        Texts per API call.  The HF free tier handles 16–32 comfortably;
        reduce if you hit 413 payload errors on long chunks.
    requests_per_second:
        Polite rate cap.  Stays within HF free-tier limits at ≤ 2.
    max_retries:
        Attempts before re-raising the last exception.  Uses 2^attempt
        second back-off between retries.
    """

    MODEL_ID = "BAAI/bge-m3"
    EMBED_DIM = 1024

    def __init__(
        self,
        api_token: str,
        batch_size: int = 32,
        requests_per_second: float = 2.0,
        max_retries: int = 5,
    ) -> None:
        self._client = InferenceClient(token=api_token)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._min_gap: float = 1.0 / max(requests_per_second, 1e-9)
        self._last_call: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed *texts* and return an (N, 1024) float32 array.

        Splits into batches of ``self.batch_size``, calls the HF endpoint
        for each, then concatenates and returns.

        Parameters
        ----------
        texts:
            Plain-text strings to embed.  Empty strings are accepted but
            return a zero vector (the API behaviour); callers should filter
            them out if zero vectors are undesirable.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), EMBED_DIM)``, dtype float32, L2-normalised.
        """
        if not texts:
            return np.empty((0, self.EMBED_DIM), dtype=np.float32)

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        parts: list[np.ndarray] = []
        for batch in batches:
            parts.append(self._embed_with_retry(batch))

        return np.concatenate(parts, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _embed_with_retry(self, batch: list[str]) -> np.ndarray:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                raw = self._client.feature_extraction(
                    batch,
                    model=self.MODEL_ID,
                )
                self._last_call = time.monotonic()
                arr = np.asarray(raw, dtype=np.float32)

                # HF may return (N, seq_len, dim) — CLS-pool if so.
                if arr.ndim == 3:
                    arr = arr[:, 0, :]

                if arr.ndim == 1:
                    # Single-text call returned a flat vector
                    arr = arr[np.newaxis, :]

                return _l2_normalise(arr)

            except Exception as exc:
                last_exc = exc
                log.warning(
                    "Embed attempt %d/%d failed for batch of %d: %s",
                    attempt,
                    self.max_retries,
                    len(batch),
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))

        raise RuntimeError(
            f"Embedding failed after {self.max_retries} retries "
            f"for batch of {len(batch)}: {last_exc}"
        ) from last_exc

    def _throttle(self) -> None:
        """Sleep until the minimum inter-request gap has elapsed."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_gap:
            time.sleep(self._min_gap - elapsed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalise(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero vectors are left as-is."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms
