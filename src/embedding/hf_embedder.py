"""
hf_embedder.py
--------------
Dense text embeddings via HuggingFace Inference API (BAAI/bge-m3).

Drop-in fallback for BGEM3Embedder when local GPU inference isn't available.
The public interface is intentionally identical: both expose an ``embed()``
method that accepts a list of strings and returns a float32 (N, 1024) array.

Error handling
~~~~~~~~~~~~~~
``requests`` is used directly to keep the dependency footprint minimal.
HTTP errors raise ``RuntimeError`` with a diagnostic message.  The caller
(``bot/services.py``) is responsible for catching them at startup.

Rate limits / cold starts
~~~~~~~~~~~~~~~~~~~~~~~~~
HuggingFace free-tier endpoints can be cold.  ``wait_for_model=True`` is
sent with every request so the call blocks until the model is warm rather
than returning a 503 immediately.  Set ``timeout`` to a generous value
(default: 120 s) when the collection is large.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import requests

log = logging.getLogger(__name__)

_HF_BASE_URL = "https://api-inference.huggingface.co/models"


class HFInferenceEmbedder:
    """
    Dense text embedder backed by the HuggingFace Inference API.

    Parameters
    ----------
    api_token:
        HuggingFace API token (``HUGGINGFACEHUB_API_TOKEN``).
    model_id:
        HuggingFace model repository ID.  Defaults to ``BAAI/bge-m3``.
    batch_size:
        Texts per API request.  Free-tier endpoints accept up to ~32
        short texts per call; reduce if you hit payload-size errors.
    timeout:
        Per-request timeout in seconds.  Increase for slow cold-starts.
    """

    MODEL_ID  = "BAAI/bge-m3"
    EMBED_DIM = 1_024

    def __init__(
        self,
        api_token: str,
        model_id: str = MODEL_ID,
        batch_size: int = 16,
        timeout: int = 120,
    ) -> None:
        if not api_token:
            raise ValueError(
                "HFInferenceEmbedder requires a non-empty api_token. "
                "Set HUGGINGFACEHUB_API_TOKEN in your .env file."
            )
        self._url     = f"{_HF_BASE_URL}/{model_id}"
        self._headers = {"Authorization": f"Bearer {api_token}"}
        self.batch_size = batch_size
        self._timeout   = timeout

        log.info(
            "HFInferenceEmbedder ready — model: %s | batch: %d | timeout: %ds",
            model_id, batch_size, timeout,
        )

    # ------------------------------------------------------------------
    # Public interface (identical to BGEM3Embedder)
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed *texts* and return an ``(N, 1024)`` float32 array.

        Parameters
        ----------
        texts:
            Plain-text strings to embed.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), 1024)``, dtype float32, L2-normalised.
        """
        if not texts:
            return np.empty((0, self.EMBED_DIM), dtype=np.float32)

        parts: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            parts.append(self._call_api(batch))

        arr = np.concatenate(parts, axis=0).astype(np.float32)
        return self._l2_normalize(arr)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _call_api(self, texts: list[str]) -> np.ndarray:
        """POST one batch to the HF Inference API and return raw vectors."""
        payload = {
            "inputs":  texts,
            "options": {"wait_for_model": True},
        }
        log.debug("HF Inference request | batch=%d", len(texts))
        resp = requests.post(
            self._url,
            headers=self._headers,
            json=payload,
            timeout=self._timeout,
        )
        if not resp.ok:
            raise RuntimeError(
                f"HF Inference API error {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()

        # The API returns either:
        #   (a) list[list[float]]          — one vector per text  (mean-pooled)
        #   (b) list[list[list[float]]]    — token-level outputs  (needs pooling)
        arr = np.array(data, dtype=np.float32)
        if arr.ndim == 3:
            # token-level → mean-pool over the token dimension
            arr = arr.mean(axis=1)
        if arr.shape[-1] != self.EMBED_DIM:
            raise RuntimeError(
                f"Unexpected embedding dim {arr.shape[-1]} (expected {self.EMBED_DIM}). "
                "Make sure the model_id points to BAAI/bge-m3."
            )
        return arr

    @staticmethod
    def _l2_normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return arr / norms
