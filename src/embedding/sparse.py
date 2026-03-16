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

# Strips leading characters that are NOT part of a Python identifier or
# dotted qualified name.  Crucially, the leading-strip arm uses ``[^\w]+``
# (no dot exception) while the trailing-strip arm uses ``[^\w.]+$`` (dots
# kept so qualified names like "torch.random.fork_rng" survive trailing noise).
#
# Why the asymmetry?
#   Leading dots are Python method-call syntax (".backward()"), not part of
#   a valid Python name.  They appear in queries like "calling .backward()"
#   and must be stripped so ".backward" → "backward", matching the stored
#   keyword.
#
#   Trailing dots DO occur in qualified names ("torch.nn.") and are harmless
#   noise — keeping them in the trailing rule avoids false stripping of the
#   dot that separates a method name from its module prefix in edge cases.
#
# Characters kept at token boundaries (trailing side only): [a-zA-Z0-9_.]
#   · word chars (\w) cover letters, digits, and underscores
#   · dots are kept to preserve qualified names like "torch.random.fork_rng"
#
# FIX (vs original): original used ``^[^\w.]+`` which kept leading dots,
# causing ".backward" to not match stored keyword "backward".
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

    Examples
    --------
    >>> tokenize_query("What is torch.cos?")
    ['what', 'is', 'torch.cos']

    >>> tokenize_query('"fork_rng" — how does it work?')
    ['fork_rng', 'how', 'does', 'it', 'work']

    >>> tokenize_query("torch.autograd.grad differ from calling .backward()?")
    ['torch.autograd.grad', 'differ', 'from', 'calling', 'backward']

    >>> tokenize_query("torch.nn.functional.relu(input)")
    ['torch.nn.functional.relu']
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
