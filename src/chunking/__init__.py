"""
Chunking package — splits processed DocPages into RAG-ready Chunk objects.
"""

from .chunker import Chunk, ChunkSplitter
from .incremental import split_incremental

__all__ = ["Chunk", "ChunkSplitter", "split_incremental"]
