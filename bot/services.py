"""
services.py
-----------
Lazy-initialised RAG pipeline singleton for the Telegram bot.

Initialization
~~~~~~~~~~~~~~
The RAG chain is expensive to build:
  - BGEM3Embedder downloads and loads a ~2.3 GB model on first use.
  - QdrantClient opens a network connection.
  - HyDETransformer initialises the Groq client.

``get_chain()`` builds the chain exactly once and caches it for the
lifetime of the process.  Concurrent callers block on an asyncio lock
until the chain is ready (double-checked locking pattern).

Embedder selection (EMBEDDER_MODE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  "local"  → BGEM3Embedder (torch + FlagEmbedding must be installed)
  "hf"     → HFInferenceEmbedder (HUGGINGFACEHUB_API_TOKEN required)
  "auto"   → try local, fall back to HF on ImportError / RuntimeError

The chain is built in a thread-pool executor so the asyncio event loop
stays responsive while the heavy model loads.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

log = logging.getLogger(__name__)

# ── Module-level state ────────────────────────────────────────────────────
_chain: Any | None = None
_chain_lock: asyncio.Lock | None = None   # created lazily (needs running loop)
_embedder_label: str = "unknown"          # human-readable; shown in /status


def _get_lock() -> asyncio.Lock:
    global _chain_lock
    if _chain_lock is None:
        _chain_lock = asyncio.Lock()
    return _chain_lock


# ── Public API ────────────────────────────────────────────────────────────

async def get_chain() -> Any:
    """
    Return the RAG chain, initialising it on first call.

    Safe for concurrent use: the first coroutine to acquire the lock
    builds the chain; all others wait and reuse it.
    """
    global _chain
    if _chain is not None:
        return _chain

    async with _get_lock():
        if _chain is not None:          # another coroutine beat us to it
            return _chain
        log.info("[Service] Building RAG pipeline …")
        loop = asyncio.get_running_loop()
        _chain = await loop.run_in_executor(None, _build_chain)
        log.info("[Service] RAG pipeline ready (embedder: %s).", _embedder_label)

    return _chain


def is_ready() -> bool:
    """True once the chain has been initialised successfully."""
    return _chain is not None


def embedder_label() -> str:
    """Human-readable embedder description for /status."""
    return _embedder_label


# ── Internal builder (runs in a thread pool) ──────────────────────────────

def _build_chain() -> Any:
    """
    Construct and return the full RAG chain.

    This function is *blocking* and should only be called via
    ``run_in_executor``.
    """
    from bot.config import settings
    from qdrant_client import QdrantClient
    from src.vectorstore import QdrantDocStore
    from src.rag import build_rag_chain

    embedder = _make_embedder()

    log.info("[Service] Connecting to Qdrant at %s …", settings.qdrant_url)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    store = QdrantDocStore(
        client=client,
        collection_name=settings.collection_name,
        embedder=embedder,
    )

    hyde = None
    if settings.hyde_enabled:
        try:
            from src.retrieval import HyDETransformer
            hyde = HyDETransformer(groq_api_key=settings.groq_api_key)
            log.info("[Service] HyDE enabled.")
        except Exception as exc:
            log.warning("[Service] HyDE init failed (%s) — disabled.", exc)

    retriever = store.as_retriever(top_k=settings.top_k, hyde=hyde)

    chain = build_rag_chain(
        retriever=retriever,
        groq_api_key=settings.groq_api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    log.info("[Service] RAG chain built successfully.")
    return chain


def _make_embedder():
    """
    Instantiate the appropriate embedder based on ``EMBEDDER_MODE``.

    Sets ``_embedder_label`` as a side-effect for use in /status.
    """
    global _embedder_label
    from bot.config import settings

    mode = settings.embedder_mode.strip().lower()

    if mode in ("local", "auto"):
        try:
            import torch
            from src.embedding import BGEM3Embedder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("[Service] Loading local BGEM3Embedder on %s …", device)
            embedder = BGEM3Embedder(batch_size=1)
            _embedder_label = f"BGEM3 (local · {device})"
            log.info("[Service] Local embedder ready.")
            return embedder

        except (ImportError, Exception) as exc:
            if mode == "local":
                raise RuntimeError(
                    f"EMBEDDER_MODE=local but local embedder is unavailable: {exc}\n"
                    "Install torch and FlagEmbedding, or set EMBEDDER_MODE=hf."
                ) from exc
            log.warning(
                "[Service] Local embedder unavailable (%s) — falling back to HF Inference.",
                exc,
            )

    # ── HF Inference fallback ────────────────────────────────────────────
    if not settings.hf_api_token:
        raise RuntimeError(
            "HF Inference embedder requires HUGGINGFACEHUB_API_TOKEN.\n"
            "Add it to your .env file, or set EMBEDDER_MODE=local (requires torch)."
        )
    from src.embedding.hf_embedder import HFInferenceEmbedder
    log.info("[Service] Using HFInferenceEmbedder.")
    embedder = HFInferenceEmbedder(api_token=settings.hf_api_token)
    _embedder_label = "BGEM3 (HF Inference API)"
    return embedder
