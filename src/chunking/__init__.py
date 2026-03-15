"""
Chunking package — splits processed DocPages into RAG-ready Chunk objects.
"""

from .chunker import Chunk, ChunkSplitter

__all__ = ["Chunk", "ChunkSplitter"]
