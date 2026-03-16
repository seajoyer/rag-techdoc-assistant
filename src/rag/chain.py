"""
chain.py
--------
Citation-aware RAG chain: HybridQdrantRetriever → Groq (llama-3.3-70b-versatile).

Design goals
~~~~~~~~~~~~
1.  **Structured output** — the chain returns a ``RAGResult`` dataclass
    containing the answer string *and* a deduplicated, rank-ordered list of
    ``SourceRef`` objects.  Callers never need to parse markdown footnotes.

2.  **Citation-aware prompt** — each context block is prefixed with a
    ``[N]`` index marker.  The system prompt instructs the model to cite
    every factual claim inline as ``[N]``.  The chain then resolves those
    markers back to real URLs.

3.  **LCEL wiring** — the full pipeline is expressed as a single
    LangChain LCEL chain so it composes naturally with other runnables
    (e.g. memory, guardrails, streaming).

4.  **Separation of concerns** — prompt engineering, context formatting,
    and source extraction live in small, independently testable functions.

Usage
~~~~~
::

    from src.rag.chain import build_rag_chain, RAGResult
    from src.vectorstore import QdrantDocStore

    store    = QdrantDocStore(client, "pytorch_docs", embedder)
    chain    = build_rag_chain(store, groq_api_key="gsk_...")
    result   = chain.invoke("How does torch.autograd.grad differ from .backward?")

    print(result.answer)
    for src in result.sources:
        print(f"  [{src.index}] {src.title} — {src.url}")

Streaming
~~~~~~~~~
For token-level streaming of the answer only (sources resolved afterward)::

    chain_stream = build_rag_chain(store, groq_api_key="...", streaming=True)
    for chunk in chain_stream.stream("What is torch.compile?"):
        print(chunk, end="", flush=True)

Environment
~~~~~~~~~~~
Set ``GROQ_API_KEY`` in your ``.env`` (or pass it explicitly).
Install extras::

    pip install langchain-groq langchain-core
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class SourceRef:
    """
    A single cited source returned alongside the answer.

    Attributes
    ----------
    index:
        The ``[N]`` marker as it appears in the answer text.
    url:
        Deep-link citation URL (``page_url + anchor``).
    title:
        Human-readable page title.
    symbol:
        Fully-qualified Python symbol, e.g. ``torch.nn.Linear``.
        Empty string for prose/heading chunks.
    kind:
        Chunk type: ``"heading"`` | ``"function"`` | ``"class"`` | …
    """

    index: int
    url: str
    title: str
    symbol: str
    kind: str


@dataclass
class RAGResult:
    """
    Complete output of the RAG chain.

    Attributes
    ----------
    answer:
        LLM-generated answer with inline ``[N]`` citation markers.
    sources:
        Deduplicated list of cited sources, ordered by first appearance.
        Only sources actually referenced in the answer are included.
    context_docs:
        The raw retrieved ``Document`` objects (useful for debugging
        or evaluation pipelines).
    """

    answer: str
    sources: list[SourceRef]
    context_docs: list[Document] = field(repr=False)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# The numbered-citation contract: every context block is prefixed [N],
# and the model must cite every factual claim with the matching [N].
_SYSTEM_PROMPT = """\
You are a precise technical assistant for the PyTorch documentation.

Answer the user's question using ONLY the context passages provided below.
Each passage is prefixed with a citation marker [N].

Rules:
- Cite every factual claim with its marker, e.g. "torch.Tensor is the \
central data structure [1]."
- A single sentence may carry multiple markers if supported by several \
passages, e.g. "[1][3]".
- If the context does not contain enough information to answer, say so \
explicitly — do not hallucinate.
- Prefer concise, technically accurate prose over bullet lists unless a \
list is clearly the best format.
- Preserve exact PyTorch symbol names, parameter names, and version notes \
as they appear in the context.
"""

_HUMAN_TEMPLATE = """\
## Context

{context}

---

## Question

{question}
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_TEMPLATE),
    ]
)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def _format_docs(docs: list[Document]) -> str:
    """
    Render retrieved documents into numbered citation blocks.

    Each block looks like::

        [1] **torch.nn.Linear** (pytorch.org/docs/stable/generated/torch.nn.Linear.html)
        Applies a linear transformation to the incoming data: y = xA^T + b.
        …

    The ``[N]`` prefix is what the system prompt trains the model to cite.
    """
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header_parts = [f"[{i}]"]

        # Surface the most useful identity signal for the LLM
        if meta.get("symbol"):
            header_parts.append(f"**{meta['symbol']}**")
        elif meta.get("page_title"):
            header_parts.append(f"**{meta['page_title']}**")

        if meta.get("citation_url"):
            header_parts.append(f"({meta['citation_url']})")

        blocks.append(" ".join(header_parts) + "\n" + doc.page_content.strip())

    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------


def _extract_sources(
    answer: str,
    docs: list[Document],
) -> list[SourceRef]:
    """
    Resolve ``[N]`` markers in *answer* back to ``SourceRef`` objects.

    Only markers that actually appear in the answer are included, and
    duplicates are deduplicated.  The result is ordered by first
    appearance in the answer.
    """
    # Find all [N] markers in order of first appearance
    seen: set[int] = set()
    ordered_indices: list[int] = []
    for m in re.finditer(r"\[(\d+)\]", answer):
        n = int(m.group(1))
        if n not in seen and 1 <= n <= len(docs):
            seen.add(n)
            ordered_indices.append(n)

    sources: list[SourceRef] = []
    for n in ordered_indices:
        meta = docs[n - 1].metadata
        sources.append(
            SourceRef(
                index=n,
                url=meta.get("citation_url", meta.get("page_url", "")),
                title=meta.get("page_title", ""),
                symbol=meta.get("symbol", ""),
                kind=meta.get("kind", ""),
            )
        )
    return sources


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------


def build_rag_chain(
    retriever: Any,
    *,
    groq_api_key: str | None = None,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    streaming: bool = False,
) -> Any:
    """
    Build and return the full RAG chain.

    The returned chain accepts a plain question string and returns a
    ``RAGResult``.  When ``streaming=True`` it returns raw token chunks
    (``str``) instead — use ``.stream()`` rather than ``.invoke()``.

    Parameters
    ----------
    retriever:
        Any LangChain ``BaseRetriever``.  Typically
        ``QdrantDocStore.as_retriever(top_k=top_k)``.
    groq_api_key:
        Groq API key.  Falls back to the ``GROQ_API_KEY`` environment
        variable if not supplied.
    model:
        Groq model identifier.  Defaults to ``llama-3.3-70b-versatile``.
    temperature:
        Sampling temperature.  0.0 gives deterministic, factual answers.
    max_tokens:
        Maximum tokens in the generated answer.
    top_k:
        Passed through to the retriever if it exposes a ``top_k``
        attribute (ignored otherwise — configure the retriever directly).
    streaming:
        If ``True``, skip the ``RAGResult`` wrapper and return raw string
        tokens suitable for ``chain.stream(question)``.

    Returns
    -------
    A LangChain ``Runnable``.
    """
    llm = ChatGroq(
        model=model,
        api_key=groq_api_key,  # None → reads GROQ_API_KEY from env
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )

    # ------------------------------------------------------------------
    # Step 1: retrieve documents and keep them accessible downstream
    # ------------------------------------------------------------------
    # We need the raw Document list both for context formatting AND for
    # source resolution after the LLM responds.  RunnablePassthrough lets
    # us carry the docs through without duplicating the retrieval call.

    def retrieve_and_pack(question: str) -> dict:
        docs = retriever.invoke(question)
        return {"question": question, "docs": docs}

    retrieve_step = RunnableLambda(retrieve_and_pack)

    # ------------------------------------------------------------------
    # Step 2: format context + run LLM
    # ------------------------------------------------------------------

    def build_prompt_input(packed: dict) -> dict:
        return {
            "context": _format_docs(packed["docs"]),
            "question": packed["question"],
            "docs": packed["docs"],  # pass through for source extraction
        }

    format_step = RunnableLambda(build_prompt_input)

    # ------------------------------------------------------------------
    # Step 3: wrap answer + docs into RAGResult
    # ------------------------------------------------------------------

    def pack_result(inputs: dict) -> RAGResult:
        answer: str = inputs["answer"]
        docs: list[Document] = inputs["docs"]
        return RAGResult(
            answer=answer,
            sources=_extract_sources(answer, docs),
            context_docs=docs,
        )

    if streaming:
        # Streaming mode: return raw token stream, skip RAGResult wrapper.
        chain = (
            retrieve_step
            | format_step
            | RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]})
            | _PROMPT
            | llm
            | StrOutputParser()
        )
        return chain

    # Non-streaming: full RAGResult with resolved citations.
    def llm_step(inputs: dict) -> dict:
        prompt_input = {"context": inputs["context"], "question": inputs["question"]}
        answer = (_PROMPT | llm | StrOutputParser()).invoke(prompt_input)
        return {"answer": answer, "docs": inputs["docs"]}

    chain = retrieve_step | format_step | RunnableLambda(llm_step) | RunnableLambda(pack_result)
    return chain


# ---------------------------------------------------------------------------
# Convenience: pretty-print a RAGResult
# ---------------------------------------------------------------------------


def print_result(result: RAGResult) -> None:
    """Print a ``RAGResult`` in a readable format (useful in scripts)."""
    print("=" * 72)
    print(result.answer)
    if result.sources:
        print("\nSources")
        print("-" * 40)
        for src in result.sources:
            label = src.symbol or src.title or src.url
            print(f"  [{src.index}] {label}")
            print(f"       {src.url}")
    print("=" * 72)
