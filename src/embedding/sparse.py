"""
sparse.py
---------
Converts pre-computed keyword lists into Qdrant-compatible sparse vectors
for hybrid keyword + semantic search.
"""

from __future__ import annotations

import re
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# 2^17 = 131 072 dimensions.  Sparse enough that Qdrant stores them
# efficiently as a list of (index, value) pairs.
VOCAB_SIZE: int = 2**17


# ---------------------------------------------------------------------------
# Token normalisation
# ---------------------------------------------------------------------------

_PUNCT_BORDER: re.Pattern[str] = re.compile(r"^[^\w]+|[^\w.]+$")


def tokenize_query(query: str) -> list[str]:
    """
    Normalise a free-text query into sparse-search tokens.

    This function is the **single source of truth** for query-side token
    normalisation.  It must produce tokens that hash to the same FNV-1a
    buckets as the keyword lists stored at index time by
    ``extractor._build_keywords``.

    Steps
    -----
    1. Lowercase.
    2. Split on whitespace.
    3. Strip any leading punctuation and trailing punctuation that is not
       part of a Python identifier or dotted qualified name.
       Leading dots are stripped (method-call syntax fix).
       Trailing dots are preserved (qualified-name safety).
    4. Discard empty residues.

    Parameters
    ----------
    query:
        Raw natural-language query, e.g. ``'What is torch.cos?'``.

    Returns
    -------
    list[str]
        Clean tokens ready to pass to ``keywords_to_sparse``.
    """
    tokens: list[str] = []
    for raw in query.lower().split():
        tok = _PUNCT_BORDER.sub("", raw)
        if tok:
            tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# Public types & API
# ---------------------------------------------------------------------------

class SparseVector(NamedTuple):
    """Lightweight sparse vector ready for Qdrant upsert."""
    indices: list[int]
    values: list[float]


def keywords_to_sparse(keywords: list[str]) -> SparseVector:
    """
    Convert a keyword list to a raw-TF sparse vector.

    Duplicate tokens in *keywords* are counted (TF > 1); the caller can
    pass ``list(set(keywords))`` if binary weights are preferred.

    Parameters
    ----------
    keywords:
        Token strings.

    Returns
    -------
    SparseVector
        ``indices``: feature-hashed bucket indices (deduplicated, sorted).
        ``values``:  raw term-frequency float weights.
    """
    counts: dict[int, float] = {}
    for kw in keywords:
        idx = _fnv1a(kw) % VOCAB_SIZE
        counts[idx] = counts.get(idx, 0.0) + 1.0

    if not counts:
        return SparseVector(indices=[], values=[])

    pairs = sorted(counts.items())
    return SparseVector(
        indices=[i for i, _ in pairs],
        values=[v for _, v in pairs],
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _fnv1a(s: str) -> int:
    """
    FNV-1a 32-bit hash of a UTF-8 string.

    Deterministic, parameter-free, no seed — identical results across Python
    versions, platforms, and process restarts.
    """
    h = 0x811C9DC5
    for b in s.encode():
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h
