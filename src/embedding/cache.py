"""
cache.py
--------
Disk-backed vector cache for incremental embedding.

Storage
~~~~~~~
Two parallel files in the output directory:

    _vectors.npy          float32 ndarray, shape (N, 1024)
    _vector_ids.json      JSON list of N chunk_id strings

Only chunks whose chunk_id is absent from _vector_ids.json are sent to
the embedder; the rest are loaded from the .npy file.  The final array
is re-ordered to match the caller's chunk list, so this is a drop-in
replacement for the old np.load(VECTORS_PATH) pattern.

Usage
~~~~~
    cache = VectorCache(OUTPUT_DIR)
    dense_vectors = cache.embed_missing(chunks, embedder)
    store.upsert_chunks(chunks, dense_vectors)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

try:
    from .embedder import BGEM3Embedder
    from ..chunking.chunker import Chunk
except ImportError:
    from embedder import BGEM3Embedder          # type: ignore[no-redef]
    from chunking.chunker import Chunk          # type: ignore[no-redef]

log = logging.getLogger(__name__)

_MACRO_BATCH = 50       # chunks per progress-bar tick
_EMBED_DIM   = 1_024    # BGE-M3 output dimension


class VectorCache:
    """
    Incremental dense-vector cache backed by .npy + .json files.

    Parameters
    ----------
    cache_dir:
        Directory that holds (or will hold) the cache files.  Typically
        the same ``output_dir`` used by the data-acquisition pipeline.
    vectors_filename / ids_filename:
        Override when managing multiple caches in the same directory.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        vectors_filename: str = "_vectors.npy",
        ids_filename: str    = "_vector_ids.json",
    ) -> None:
        self._dir      = Path(cache_dir)
        self._vec_path = self._dir / vectors_filename
        self._ids_path = self._dir / ids_filename

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def exists(self) -> bool:
        """True when both cache files are present."""
        return self._vec_path.exists() and self._ids_path.exists()

    @property
    def size(self) -> int:
        """Number of vectors currently stored (0 if cache is absent)."""
        if not self.exists:
            return 0
        try:
            return len(json.loads(self._ids_path.read_text(encoding="utf-8")))
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    def load(self) -> tuple[list[str], np.ndarray]:
        """
        Load the full cache from disk.

        Returns
        -------
        (chunk_ids, vectors)
            Parallel structures: ``vectors[i]`` is the embedding for
            ``chunk_ids[i]``.  Returns empty structures when cache is absent
            or corrupted (with a warning logged in the latter case).
        """
        _empty = ([], np.empty((0, _EMBED_DIM), dtype=np.float32))

        if not self.exists:
            return _empty

        try:
            chunk_ids: list[str] = json.loads(
                self._ids_path.read_text(encoding="utf-8")
            )
            vectors = np.load(self._vec_path)

            if len(chunk_ids) != len(vectors):
                log.warning(
                    "Cache mismatch: %d IDs vs %d vectors — ignoring cache.",
                    len(chunk_ids), len(vectors),
                )
                return _empty

            log.info("Loaded %d cached vectors from %s", len(chunk_ids), self._vec_path)
            return chunk_ids, vectors

        except Exception as exc:
            log.warning("Failed to load vector cache (%s) — starting fresh.", exc)
            return _empty

    def save(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        """Persist chunk_ids and vectors to disk (overwrites existing files)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        np.save(self._vec_path, vectors.astype(np.float32))
        self._ids_path.write_text(
            json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Saved %d vectors -> %s", len(chunk_ids), self._vec_path)

    # ------------------------------------------------------------------
    # High-level: incremental embedding  (main entry point)
    # ------------------------------------------------------------------

    def embed_missing(
        self,
        chunks: list[Chunk],
        embedder: "BGEM3Embedder",
        macro_batch: int = _MACRO_BATCH,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed any chunks not already in the cache; return an aligned array.

        Workflow
        --------
        1. Load existing cache -> {chunk_id: row_index} lookup.
        2. Identify chunks whose chunk_id is absent.
        3. Embed missing chunks in macro_batch-sized progress ticks.
        4. Merge new vectors into the cache and persist to disk.
        5. Return float32 (len(chunks), 1024) aligned to the chunks order.

        Parameters
        ----------
        chunks:
            The full chunk list that the caller will pass to upsert_chunks().
            Every chunk_id must be representable — either from the cache or
            just embedded.
        embedder:
            BGEM3Embedder used to compute dense vectors.
        macro_batch:
            Chunks per progress-bar update (does not affect embedder.batch_size).
        show_progress:
            Show a tqdm bar when embedding new chunks.

        Returns
        -------
        np.ndarray
            float32 array of shape (len(chunks), 1024) with row i
            corresponding to chunks[i].
        """
        cached_ids, cached_vecs = self.load()
        id_to_idx: dict[str, int] = {cid: i for i, cid in enumerate(cached_ids)}

        # ── Find missing chunks ───────────────────────────────────────────
        missing = [c for c in chunks if c.chunk_id not in id_to_idx]

        if not missing:
            log.info(
                "All %d chunks already cached — no embedding needed.", len(chunks)
            )
        else:
            log.info(
                "Embedding %d new chunk(s); %d already cached.",
                len(missing), len(cached_ids),
            )
            new_vecs = self._embed_chunks(missing, embedder, macro_batch, show_progress)

            # Merge: extend the parallel lists and persist
            merged_ids  = cached_ids + [c.chunk_id for c in missing]
            merged_vecs = (
                np.concatenate([cached_vecs, new_vecs], axis=0)
                if len(cached_ids) > 0
                else new_vecs
            )
            self.save(merged_ids, merged_vecs)

            # Refresh lookup with the full merged state
            id_to_idx   = {cid: i for i, cid in enumerate(merged_ids)}
            cached_vecs = merged_vecs

        # ── Align to caller's chunk order ─────────────────────────────────
        try:
            aligned = np.stack([cached_vecs[id_to_idx[c.chunk_id]] for c in chunks])
        except KeyError as exc:
            raise KeyError(
                f"chunk_id {exc} not found in cache after embedding — "
                "this should not happen; check embedder output."
            ) from exc

        return aligned.astype(np.float32)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _embed_chunks(
        self,
        chunks: list[Chunk],
        embedder: "BGEM3Embedder",
        macro_batch: int,
        show_progress: bool,
    ) -> np.ndarray:
        texts = [c.text for c in chunks]
        parts: list[np.ndarray] = []

        pbar = tqdm(
            total=len(texts), desc="Embedding", unit="chunk",
            disable=not show_progress,
        )
        for start in range(0, len(texts), macro_batch):
            batch = texts[start : start + macro_batch]
            parts.append(embedder.embed(batch))
            pbar.update(len(batch))
        pbar.close()

        return (
            np.concatenate(parts, axis=0) if parts
            else np.empty((0, _EMBED_DIM), dtype=np.float32)
        )
