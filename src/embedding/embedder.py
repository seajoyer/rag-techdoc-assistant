"""
embedder.py
-----------
Dense text embeddings via BAAI/bge-m3 running locally on CUDA (or CPU).

Drop-in replacement for the original HuggingFace Inference-API embedder.
Public interface is identical — nothing else in the project needs to change.

Requires
--------
    pip install FlagEmbedding

The model weights (~2.2 GB) are downloaded from HuggingFace Hub on first
use and cached in ~/.cache/huggingface/.

Why FlagEmbedding over sentence-transformers
--------------------------------------------
FlagEmbedding is BAAI's own reference implementation.  It uses the exact
inference path from training, and it's the only library that guarantees
correct dense/sparse/colbert output formats for BGE-M3.

GPU memory guide  (fp16 weights ≈ 2.2 GB on-device, max_length=512)
--------------------------------------------------------------------
VRAM      batch_size   Notes
 4 GB     16           Safe baseline; headroom for OOM spikes
 8 GB     64           Good default for most consumer cards
16 GB     128          High-throughput
24 GB+    256          Diminishing returns above this
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

log = logging.getLogger(__name__)


class BGEM3Embedder:
    """
    Dense text embedder backed by ``BAAI/bge-m3`` running locally on CUDA.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model ID or local path.  Defaults to ``BAAI/bge-m3``.
    batch_size:
        Sequences per forward pass.  See module docstring for VRAM guide.
    use_fp16:
        fp16 weights — ~2× faster on Ampere+ (RTX 30xx / A100), negligible
        quality loss (<0.1% on BEIR).  Auto-disabled on CPU.
    max_length:
        Truncation in tokens.  BGE-M3 supports up to 8 192, but 512 is a
        safe ceiling for retrieval chunks.
    devices:
        ``"cuda"``, ``"cuda:1"``, ``["cuda:0","cuda:1"]``, or ``None``
        for auto-select (CUDA if available, else CPU).
    """

    MODEL_ID = "BAAI/bge-m3"
    EMBED_DIM = 1024

    def __init__(
        self,
        model_name_or_path: str = MODEL_ID,
        batch_size: int = 16,
        use_fp16: bool = True,
        max_length: int = 512,
        devices: str | list[str] | None = None,
    ) -> None:
        if devices is None:
            devices = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = BGEM3FlagModel(
            model_name_or_path,
            use_fp16=use_fp16 and str(devices) != "cpu",
            devices=devices,
        )
        self.batch_size = batch_size
        self.max_length = max_length

        _dev = devices if isinstance(devices, str) else ", ".join(devices)
        log.info(
            "BGEM3Embedder ready — model: %s | device: %s | fp16: %s | batch: %d",
            model_name_or_path, _dev, use_fp16, batch_size,
        )

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

        output = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,       # sparse lives in sparse.py
            return_colbert_vecs=False,
        )
        # FlagEmbedding already returns L2-normalised float32.
        return np.asarray(output["dense_vecs"], dtype=np.float32)
