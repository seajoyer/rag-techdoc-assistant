"""
RAG package — citation-aware LLM chain wired to the hybrid Qdrant retriever.
"""

from .chain import RAGResult, SourceRef, build_rag_chain, print_result

__all__ = ["RAGResult", "SourceRef", "build_rag_chain", "print_result"]
