from .sparse import keywords_to_sparse, tokenize_query, SparseVector
from .cache import VectorCache
from .hf_embedder import HFInferenceEmbedder

def __getattr__(name: str):
    if name == "BGEM3Embedder":
        from .embedder import BGEM3Embedder
        return BGEM3Embedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BGEM3Embedder",
    "HFInferenceEmbedder",
    "keywords_to_sparse",
    "tokenize_query",
    "SparseVector",
    "VectorCache",
]
