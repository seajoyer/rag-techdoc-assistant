"""
incremental.py
--------------
Resumable chunk splitting: wraps ChunkSplitter with I/O so the pure
chunker module stays testable without a filesystem.

On subsequent runs, pages whose URL already appears in _chunks.jsonl are
silently skipped.  New chunks are appended, and the complete list
(existing + new) is returned.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from tqdm.auto import tqdm

from .chunker import Chunk, ChunkSplitter

try:
    from ..data_acquisition.pipeline import DocPage
except ImportError:
    from data_acquisition.pipeline import DocPage  # type: ignore[no-redef]

log = logging.getLogger(__name__)


def split_incremental(
    pages: list[DocPage],
    chunks_path: Path,
    splitter: ChunkSplitter | None = None,
    show_progress: bool = True,
) -> list[Chunk]:
    """
    Split pages into chunks, skipping pages already present in chunks_path.

    Parameters
    ----------
    pages:
        All DocPage objects to consider. Pages whose URL already appears in
        chunks_path are silently skipped.
    chunks_path:
        JSONL file produced by previous runs. Created (and its parent dirs)
        if absent — i.e. the first call behaves like a full split.
    splitter:
        ChunkSplitter instance. Defaults to ChunkSplitter() (1500/120/200).
    show_progress:
        Show a tqdm bar over new pages.

    Returns
    -------
    list[Chunk]
        All chunks (existing + newly split), in file order.
    """
    if splitter is None:
        splitter = ChunkSplitter()

    # ── Load existing chunks & the URLs they cover ────────────────────────
    existing_chunks: list[Chunk] = []
    processed_urls: set[str] = set()

    if chunks_path.exists():
        existing_chunks, processed_urls = _load_chunks_and_urls(chunks_path)
        log.info(
            "Incremental chunking: %d existing chunks across %d pages.",
            len(existing_chunks), len(processed_urls),
        )

    # ── Identify new pages ────────────────────────────────────────────────
    new_pages = [p for p in pages if p.url not in processed_urls]

    if not new_pages:
        log.info("All %d pages already chunked — nothing to do.", len(pages))
        return existing_chunks

    log.info(
        "Chunking %d new page(s) (skipping %d already done).",
        len(new_pages), len(processed_urls),
    )

    # ── Split new pages ───────────────────────────────────────────────────
    new_chunks: list[Chunk] = []
    iterator = (
        tqdm(new_pages, desc="Chunking new pages", unit="page")
        if show_progress else new_pages
    )
    for page in iterator:
        new_chunks.extend(splitter.split(page))

    # ── Append to JSONL ───────────────────────────────────────────────────
    if new_chunks:
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with chunks_path.open("a", encoding="utf-8") as f:
            for chunk in new_chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        log.info("Appended %d new chunks -> %s", len(new_chunks), chunks_path)

    return existing_chunks + new_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_chunks_and_urls(
    chunks_path: Path,
) -> tuple[list[Chunk], set[str]]:
    """Deserialise chunks from JSONL; return chunks and the page_url set."""
    chunks: list[Chunk] = []
    urls: set[str] = set()

    with chunks_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                d.pop("char_count", None)   # computed in __post_init__
                chunks.append(Chunk(**d))
                urls.add(d["page_url"])
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                log.warning("Skipping malformed chunk at line %d: %s", lineno, exc)

    return chunks, urls
