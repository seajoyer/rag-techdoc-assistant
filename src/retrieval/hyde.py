"""
hyde.py
-------
Hypothetical Document Embeddings (HyDE) query transformer for documentation RAG.

Problem it solves
~~~~~~~~~~~~~~~~~
BGE-M3 embeds a natural-language question ("How does torch.autograd.grad differ
from .backward()?") into a very different region of the 1024-d vector space than
a terse API reference chunk ("torch.autograd.grad(*outputs, ...) — Computes
and returns the sum of gradients...").  The question framing dilutes the key
symbol tokens, so the nearest-neighbour search misses the relevant chunk entirely.

How HyDE works
~~~~~~~~~~~~~~
Instead of embedding the raw question, we ask an LLM to write a short
*hypothetical documentation snippet* that would directly answer the question.
That snippet "sounds like" a real doc chunk — terse, symbol-centric, reference
style — so its embedding lands in the same neighbourhood as the actual chunk.

Split-channel design
~~~~~~~~~~~~~~~~~~~~
This module is used **only for the dense embedding leg** of the hybrid search.
The sparse (keyword IDF) leg keeps the original query, because:

* The sparse leg already excels at exact symbol matching when the symbol appears
  verbatim in the question.
* A hypothetical snippet may introduce new tokens not in the original query,
  which could inflate sparse recall for the wrong chunks.

Using different representations for each leg is the key insight:
    dense leg  → embed(hypothetical_snippet)  # catches semantic similarity
    sparse leg → tokenize(original_query)     # catches exact keyword matches

Usage
~~~~~
::

    from src.retrieval.hyde import HyDETransformer

    hyde = HyDETransformer(groq_api_key="gsk_...")

    # Returns a short documentation-style snippet:
    snippet = hyde.transform("How does torch.autograd.grad differ from .backward()?")
    # → "torch.autograd.grad(*outputs, inputs, grad_outputs=None, ...) computes
    #    and returns the sum of gradients of outputs with respect to the inputs.
    #    Unlike .backward(), which accumulates gradients into .grad attributes,
    #    grad() returns the gradients as tensors and does not modify .grad."

    # The snippet is then embedded (in QdrantDocStore.hybrid_search) and used
    # as the dense query vector, while the original question is used for sparse.

Error handling
~~~~~~~~~~~~~~
If the Groq API call fails (network, rate-limit, invalid key), the transformer
logs a warning and falls back to the original query unchanged.  This means HyDE
degrades gracefully — a failure never breaks the retrieval pipeline, it just
reverts to vanilla dense search.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a technical documentation assistant specialising in PyTorch.
Given a question about PyTorch, write a short documentation-style snippet
(3–6 sentences) that would directly answer it.

Write only the snippet itself — no preamble, no "here is...", no markdown headers.
Use precise PyTorch API names, parameter names, and technical terminology exactly
as they appear in the official documentation.
Be concise and factual, as if writing a paragraph from the official docs."""

_USER_TEMPLATE = "Question: {question}\n\nDocumentation snippet:"


class HyDETransformer:
    """
    Transforms a natural-language question into a hypothetical documentation
    snippet suitable for dense vector search.

    Parameters
    ----------
    groq_api_key:
        Groq API key.  Falls back to the ``GROQ_API_KEY`` environment
        variable when *None*.
    model:
        Groq model identifier.  ``llama-3.1-8b-instant`` is fast and cheap
        for this task; accuracy requirements are low because even an imperfect
        hypothetical snippet embeds closer to the right chunk than a question does.
    max_tokens:
        Maximum output tokens for the generated snippet.  64–128 is enough;
        longer snippets add latency without improving retrieval quality.
    """

    def __init__(
        self,
        groq_api_key: str | None = None,
        model: str = "llama-3.1-8b-instant",
        max_tokens: int = 128,
    ) -> None:
        try:
            from groq import Groq  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "groq package is required for HyDETransformer.  "
                "Install it with: pip install groq"
            ) from exc

        self._client = Groq(
            api_key=groq_api_key or os.environ.get("GROQ_API_KEY")
        )
        self.model = model
        self.max_tokens = max_tokens

    def transform(self, query: str) -> str:
        """
        Generate a hypothetical documentation snippet for *query*.

        Returns the original *query* unchanged on API failure so the caller
        always gets a usable dense query string.

        Parameters
        ----------
        query:
            Natural-language question, e.g.
            ``"How does torch.autograd.grad differ from .backward()?"``

        Returns
        -------
        str
            Hypothetical documentation snippet (3–6 sentences), or *query*
            on failure.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _USER_TEMPLATE.format(question=query)},
                ],
            )
            snippet = response.choices[0].message.content.strip()
            if not snippet:
                log.warning("HyDETransformer: empty response — falling back to raw query")
                return query
            log.debug("HyDE snippet for %r: %r", query[:60], snippet[:120])
            return snippet

        except Exception as exc:
            log.warning(
                "HyDETransformer: API error (%s) — falling back to raw query",
                exc,
            )
            return query
