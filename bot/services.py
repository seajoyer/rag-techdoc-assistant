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

Streaming support
~~~~~~~~~~~~~~~~~
``astream_answer(question)`` is an async generator that yields raw ``str``
token chunks followed by a single ``RAGResult`` as the last item.  It
bridges the blocking ``stream_rag`` generator (which runs in a thread-pool
executor) to the asyncio event loop via an ``asyncio.Queue``.

The retriever instance and LLM parameters are cached alongside the chain
after the first build so ``astream_answer`` can reuse them without any
additional initialisation cost.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, AsyncGenerator

log = logging.getLogger(__name__)

# ── Module-level state ────────────────────────────────────────────────────
_chain: Any | None = None
_chain_lock: asyncio.Lock | None = None   # created lazily (needs running loop)
_embedder_label: str = "unknown"          # human-readable; shown in /status

# Cached for streaming path — set inside _build_chain (same executor call).
_retriever: Any | None = None
_rag_params: dict | None = None           # groq_api_key, model, temperature, max_tokens


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


async def astream_answer(
    question: str,
) -> AsyncGenerator[Any, None]:
    """
    Async generator: yields raw LLM token ``str`` chunks, then a ``RAGResult``.

    The last item is always a ``RAGResult``; callers detect it with
    ``isinstance(item, RAGResult)``.

    This function bridges the blocking ``stream_rag`` generator to the
    asyncio event loop:

    1. Ensures the pipeline is initialised (no-op after the first call).
    2. Spawns a daemon thread that runs ``stream_rag`` and enqueues each
       yielded value via ``asyncio.run_coroutine_threadsafe``.
    3. The async generator drains the queue, re-raising any exception that
       propagated from the thread.

    The queue has a bounded size (``_STREAM_QUEUE_MAXSIZE``) to provide
    back-pressure: the producer thread blocks once the consumer falls
    behind, preventing unbounded memory growth for slow Telegram connections.
    """
    from src.rag.chain import stream_rag

    # Ensure the pipeline (and therefore _retriever / _rag_params) is ready.
    await get_chain()

    if _retriever is None or _rag_params is None:
        raise RuntimeError(
            "Retriever not initialised after get_chain() — this should not happen."
        )

    _STREAM_QUEUE_MAXSIZE = 128
    sentinel = object()
    queue: asyncio.Queue = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
    loop = asyncio.get_running_loop()

    def _run_stream() -> None:
        """Blocking worker: runs in the thread pool."""
        try:
            for item in stream_rag(question, _retriever, **_rag_params):
                # run_coroutine_threadsafe + .result() gives us back-pressure:
                # this blocks the producer thread until the queue has space.
                future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                future.result()          # propagate CancelledError / timeout
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

    thread = threading.Thread(target=_run_stream, daemon=True, name="rag-stream")
    thread.start()

    while True:
        item = await queue.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


# ── Internal builder (runs in a thread pool) ──────────────────────────────

def _build_chain() -> Any:
    """
    Construct and return the full RAG chain.

    Side-effects
    ~~~~~~~~~~~~
    Sets the module-level ``_retriever``, ``_rag_params``, and
    ``_embedder_label`` globals so that ``astream_answer`` can reuse the
    already-constructed retriever without rebuilding the pipeline.

    This function is *blocking* and should only be called via
    ``run_in_executor``.
    """
    global _retriever, _rag_params

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

    # ── Optional cross-encoder reranker ──────────────────────────────────────
    reranker = None
    if settings.reranker_enabled:
        try:
            from src.retrieval import CrossEncoderReranker
            reranker = CrossEncoderReranker(default_alpha=settings.reranker_alpha)
            log.info(
                "[Service] CrossEncoderReranker enabled | alpha=%.2f | candidate_top_k=%d",
                settings.reranker_alpha, settings.reranker_top_k,
            )
        except Exception as exc:
            log.warning("[Service] Reranker init failed (%s) — disabled.", exc)

    retriever = store.as_retriever(
        top_k=settings.reranker_top_k if reranker else settings.top_k,
        hyde=hyde,
        reranker=reranker,
        reranker_alpha=settings.reranker_alpha,
        final_top_k=settings.top_k,   # always return top_k docs to the LLM
    )

    # ── Cache components needed by astream_answer ─────────────────────────
    _retriever = retriever
    _rag_params = {
        "groq_api_key": settings.groq_api_key,
        "temperature":  settings.temperature,
        "max_tokens":   settings.max_tokens,
        # model is intentionally omitted so stream_rag uses its own default,
        # but we pass it explicitly here for consistency.
        "model": "llama-3.3-70b-versatile",
    }

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
