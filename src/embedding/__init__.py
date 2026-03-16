"""
Embedding package — dense and sparse vector production for RAG.
"""

from .embedder import BGEM3Embedder
from .sparse import keywords_to_sparse, tokenize_query, SparseVector
from .cache import VectorCache

__all__ = [
    "BGEM3Embedder",
    "keywords_to_sparse",
    "tokenize_query",
    "SparseVector",
    "VectorCache",
]
