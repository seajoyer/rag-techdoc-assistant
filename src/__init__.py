"""
Root package init — exposes all sub-packages.
"""

from . import data_acquisition, chunking, embedding, vectorstore, rag

__all__ = [
    "data_acquisition",
    "chunking",
    "embedding",
    "vectorstore",
    "rag",
]
