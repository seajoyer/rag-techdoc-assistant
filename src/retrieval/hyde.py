"""
hyde.py
-------
Hypothetical Document Embeddings (HyDE) query transformer for documentation RAG.

Split-channel design
~~~~~~~~~~~~~~~~~~~~
This module is used **only for the dense embedding leg** of the hybrid search.
The sparse (keyword IDF) leg keeps the original query.

Using different representations for each leg is the key insight:
    dense leg  -> embed(hypothetical_snippet)  # catches semantic similarity
    sparse leg -> tokenize(original_query)     # catches exact keyword matches

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
You are producing a retrieval key for a semantic search index over the \
PyTorch documentation.

Given a user question, write a short passage (3–5 sentences) of the kind \
that would appear verbatim in the official PyTorch API reference or \
developer guide as the direct answer.

Requirements:
- Use exact PyTorch identifiers: module paths, class names, method names, \
and parameter names as they appear in the docs \
(e.g. torch.nn.utils.clip_grad_norm_, Tensor.detach, optimizer.zero_grad).
- Include concrete details: default parameter values, return types, or \
behavioural caveats where they are central to the answer.
- Write in the declarative, third-person style of API reference prose — \
not conversational, not instructional ("you should…").
- Output the passage only. No preamble, no headings, no code fences.\
"""

_USER_TEMPLATE = "Question: {question}\n\nDocumentation passage:"


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
        log.info("[HyDE] Generating hypothetical snippet | model=%s | query=%r", self.model, query)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.0,
                seed=42,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _USER_TEMPLATE.format(question=query)},
                ],
            )
            snippet = response.choices[0].message.content.strip()
            if not snippet:
                log.warning("[HyDE] Empty response from model — falling back to raw query")
                return query
            usage = response.usage
            log.info(
                "[HyDE] Snippet generated | tokens_in=%d tokens_out=%d | snippet=%r",
                usage.prompt_tokens if usage else -1,
                usage.completion_tokens if usage else -1,
                snippet,
            )
            return snippet

        except Exception as exc:
            log.warning("[HyDE] API error (%s) — falling back to raw query", exc)
            return query
