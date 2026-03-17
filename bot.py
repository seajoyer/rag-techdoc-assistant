"""
bot.py
------
Dead-simple synchronous Telegram bot replicating 04_rag_chain.ipynb.

Usage
-----
1.  Make sure your .env (or environment) has:
        TELEGRAM_BOT_TOKEN=...
        QDRANT_URL=...
        QDRANT_API_KEY=...
        GROQ_API_KEY=...

2.  Run from the project root so src/ is importable:
        python bot.py

Extra dependency:
    pip install pyTelegramBotAPI
"""

import logging
import os
import sys
import time
from pathlib import Path

import telebot
from dotenv import load_dotenv
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding import BGEM3Embedder
from src.rag import RAGResult, build_rag_chain
from src.retrieval import HyDETransformer
from src.vectorstore import QdrantDocStore

# ── config (notebook Section 2) ──────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")

COLLECTION_NAME = "pytorch_docs"
TOP_K           = 6
MAX_TOKENS      = 1024
TEMPERATURE     = 0.0

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_KEY = os.environ["QDRANT_API_KEY"]
GROQ_KEY   = os.environ["GROQ_API_KEY"]
BOT_TOKEN  = os.environ["TELEGRAM_BOT_TOKEN"]

# ── logging ───────────────────────────────────────────────────────────────────
# Root logger stays at WARNING so third-party libraries stay quiet.
# Our own pipeline namespaces are promoted to INFO so every tagged log line
# ([HyDE], [Embed], [Sparse], [Search], [Retriever], [Chain]) is visible.
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(name)-35s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

_PIPELINE_LOGGERS = [
    "src.retrieval.hyde",
    "src.embedding.embedder",
    "src.embedding.cache",
    "src.vectorstore.store",
    "src.rag.chain",
]
for _name in _PIPELINE_LOGGERS:
    logging.getLogger(_name).setLevel(logging.INFO)

log = logging.getLogger(__name__)

# ── build the chain once at startup (notebook Sections 3-4) ──────────────────
print("Connecting to Qdrant …")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
embedder      = BGEM3Embedder(batch_size=1)
store         = QdrantDocStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedder=embedder,
)
info = store.collection_info()
print(f"Collection `{info['name']}`: {info['points_count']:,} points, status: {info['status']}")

hyde      = HyDETransformer(groq_api_key=GROQ_KEY)
retriever = store.as_retriever(top_k=TOP_K, hyde=hyde)
chain     = build_rag_chain(
    retriever=retriever,
    groq_api_key=GROQ_KEY,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)
print("Chain ready:", chain)

# ── bot ───────────────────────────────────────────────────────────────────────
bot = telebot.TeleBot(BOT_TOKEN)


def _format_result(result: RAGResult) -> str:
    """Same output as print_result() in the notebook."""
    lines = ["=" * 72, result.answer]
    if result.sources:
        lines += ["", "Sources", "-" * 40]
        for src in result.sources:
            label = src.symbol or src.title or src.url
            lines.append(f"  [{src.index}] {label}")
            lines.append(f"       {src.url}")
    lines.append("=" * 72)
    return "\n".join(lines)


def _format_retrieval(result: RAGResult) -> str:
    lines = [f"Retrieved {len(result.context_docs)} chunks:"]
    for i, doc in enumerate(result.context_docs, 1):
        m = doc.metadata
        lines.append(f"  [{i}] kind={m.get('kind'):<10}  score={m.get('score', 0):.4f}")
        lines.append(f"symbol={m.get('symbol') or '—'}")
        lines.append(f"       {m.get('citation_url')}")
    return "\n".join(lines)


@bot.message_handler(func=lambda _: True)
def handle_message(message):
    question = message.text.strip()
    log.warning("[Bot] Incoming question | user=%s | question=%r", message.from_user.id, question)

    t0 = time.perf_counter()
    result = chain.invoke(question)
    elapsed = time.perf_counter() - t0

    log.warning(
        "[Bot] Pipeline complete | elapsed=%.2fs | sources=%d | answer_chars=%d",
        elapsed, len(result.sources), len(result.answer),
    )

    reply     = _format_result(result)
    retrieval = _format_retrieval(result)

    # Telegram caps messages at 4096 chars
    for i in range(0, len(reply), 4096):
        bot.send_message(message.chat.id, reply[i : i + 4096])
        bot.send_message(message.chat.id, retrieval[i : i + 4096])

if __name__ == "__main__":
    print("Bot is running. Press Ctrl-C to stop.")
    bot.infinity_polling()
