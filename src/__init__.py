"""
Root package init — exposes all sub-packages.
"""

from . import data_acquisition, chunking, embedding, retrieval, vectorstore, rag, evaluation

__all__ = [
    "data_acquisition",
    "chunking",
    "embedding",
    "retrieval",
    "vectorstore",
    "rag",
    "evaluation",
]
