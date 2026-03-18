"""
RAG package — citation-aware LLM chain wired to the hybrid Qdrant retriever.
"""

from .chain import RAGResult, SourceRef, build_rag_chain, print_result, stream_rag

__all__ = ["RAGResult", "SourceRef", "build_rag_chain", "print_result", "stream_rag"]
