"""
sparse.py
---------
Converts pre-computed keyword lists into Qdrant-compatible sparse vectors
for hybrid keyword + semantic search.

Why feature hashing?
~~~~~~~~~~~~~~~~~~~~
A vocabulary-based approach (sklearn TfidfVectorizer, etc.) requires a
two-pass over the entire corpus to build the vocabulary before any point
can be upserted.  This is inconvenient for incremental ingestion and
prevents parallel upsert pipelines.

Feature hashing (the "hashing trick") maps each token to a bucket index
with a deterministic, parameter-free hash.  Indices are stable across runs,
which means:

* Existing collection points never need to be re-embedded when new pages
  are added — the sparse component is backward compatible.
* No vocabulary artefact needs to be stored alongside the collection.

Collision probability
~~~~~~~~~~~~~~~~~~~~~
With VOCAB_SIZE = 2**17 (131 072) and a typical keyword list of ~50 tokens
per chunk, the expected number of collisions per chunk is ≈ 0.009 (< 1%).
In practice the keyword lists are short enough that collisions are negligible.

IDF weighting
~~~~~~~~~~~~~
Document vectors carry **raw term frequency** (count of each token in the
keyword list).  Qdrant's ``SparseVectorParams(modifier=Modifier.IDF)`` then
applies collection-level IDF normalisation to the *query* vector at search
time, giving BM25-like scoring without any offline IDF computation.

This asymmetric TF–IDF approach (TF in docs, IDF applied to queries) is the
standard pattern recommended in the Qdrant documentation for hybrid search.

Token normalisation contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Both index-time keyword lists (built by ``extractor._build_keywords``) and
query-time token lists (produced by ``tokenize_query``) must go through the
**same normalisation rules** before being hashed, or sparse scores silently
collapse to zero for tokens that differ only in surrounding punctuation.

``tokenize_query`` is the authoritative implementation of those rules.
Any future change to token normalisation belongs here and nowhere else.

Usage
~~~~~
::

    from src.embedding.sparse import keywords_to_sparse, tokenize_query

    # Query side
    tokens = tokenize_query("What is torch.cos?")
    # → ["what", "is", "torch.cos"]

    sv = keywords_to_sparse(tokens)
    # sv.indices → list[int]
    # sv.values  → list[float]   (raw TF, ready for Qdrant query)

    # Index side (keywords come pre-built from extractor._build_keywords)
    sv = keywords_to_sparse(["torch", "cos", "torch.cos"])
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

# Strips any leading/trailing characters that are *not* part of a Python
# identifier or a dotted qualified name (e.g. "torch.cos", "fork_rng").
# This covers question marks, exclamation points, quotes, brackets, commas,
# colons, semicolons — anything that natural-language wrapping adds around
# an otherwise clean symbol or keyword.
#
# Characters kept at token boundaries: [a-zA-Z0-9_.]
#   · word chars (\w) cover letters, digits, and underscores
#   · dots are kept to preserve qualified names like "torch.random.fork_rng"
_PUNCT_BORDER: re.Pattern[str] = re.compile(r"^[^\w.]+|[^\w.]+$")


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
    3. Strip any leading/trailing punctuation that is not part of a Python
       identifier or dotted qualified name (``[a-z0-9_.]``).
    4. Discard empty residues.

    Parameters
    ----------
    query:
        Raw natural-language query, e.g. ``'What is torch.cos?'``.

    Returns
    -------
    list[str]
        Clean tokens ready to pass to ``keywords_to_sparse``.

    Examples
    --------
    >>> tokenize_query("What is torch.cos?")
    ['what', 'is', 'torch.cos']

    >>> tokenize_query('"fork_rng" — how does it work?')
    ['fork_rng', 'how', 'does', 'it', 'work']

    >>> tokenize_query("torch.nn.functional.relu(input)")
    ['torch.nn.functional.relu(input']   # interior parens not stripped — fine,
                                          # they won't match any stored keyword
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
        Token strings.  Typically ``Chunk.keywords`` — module components,
        symbol names, parameter names, and heading words.

    Returns
    -------
    SparseVector
        Parallel ``indices`` / ``values`` lists sorted by index, suitable
        for ``QdrantSparseVector(indices=…, values=…)``.
        Returns empty lists when *keywords* is empty.
    """
    if not keywords:
        return SparseVector([], [])

    # Accumulate raw term frequencies per bucket
    tf: dict[int, float] = {}
    for token in keywords:
        idx = _fnv1a_hash(token)
        tf[idx] = tf.get(idx, 0.0) + 1.0

    # Sort by index (Qdrant requires sorted sparse vectors)
    indices = sorted(tf)
    values = [tf[i] for i in indices]
    return SparseVector(indices, values)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _fnv1a_hash(token: str) -> int:
    """
    FNV-1a 32-bit hash, mapped into [0, VOCAB_SIZE).

    FNV-1a is simple, fast, has no external dependencies, and distributes
    natural-language tokens well.  The modulo step into VOCAB_SIZE may
    introduce secondary collisions but these are negligible at 2^17.
    """
    h = 2_166_136_261  # FNV offset basis (32-bit)
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 16_777_619) & 0xFFFF_FFFF  # FNV prime, keep 32-bit
    return h % VOCAB_SIZE
