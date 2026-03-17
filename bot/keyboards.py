"""
keyboards.py
------------
Telegram inline-keyboard builders and HTML message formatters.

Citation rendering
~~~~~~~~~~~~~~~~~~
The LLM returns plain text with ``[N]`` markers, e.g.::

    torch.Tensor is the central data structure [1]. It supports
    autograd [2][3].

``format_rag_response()`` converts this into Telegram HTML:

  1. HTML-escapes the whole answer (protects against < / > / &).
  2. Replaces each ``[N]`` with a clickable ``<a href="url">[N]</a>`` link.
  3. Appends a *Sources* footer with titled links.
  4. Wraps any inline code (``...``) and fenced blocks in <code>/<pre>.

Inline keyboard
~~~~~~~~~~~~~~~
``sources_keyboard()`` builds one URL button per cited source so users
can open documentation pages with a single tap.
"""

from __future__ import annotations

import html
import re

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

try:
    from src.rag.chain import RAGResult, SourceRef
except ImportError:
    from rag.chain import RAGResult, SourceRef  # type: ignore[no-redef]

# Maximum Telegram message length (hard limit: 4096).
# We stay a bit below to leave room for the sources footer.
_MAX_ANSWER_CHARS = 3_500

# Unicode superscripts for [1]–[9]; fall back to plain [N] beyond that.
_SUPERSCRIPTS = {1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵",
                 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹"}


# ---------------------------------------------------------------------------
# Keyboard builder
# ---------------------------------------------------------------------------

def sources_keyboard(sources: list[SourceRef]) -> InlineKeyboardMarkup | None:
    """
    Build an inline keyboard with one URL button per cited source.

    Returns ``None`` when *sources* is empty so callers can pass it
    directly to ``reply_markup`` without an extra check.
    """
    if not sources:
        return None

    buttons: list[list[InlineKeyboardButton]] = []
    for src in sources:
        label = src.symbol or src.title or f"Source {src.index}"
        # Telegram button text: 64-char limit, prefix with index
        btn_text = f"[{src.index}] {label}"
        if len(btn_text) > 60:
            btn_text = btn_text[:57] + "…"
        buttons.append([InlineKeyboardButton(text=btn_text, url=src.url)])

    return InlineKeyboardMarkup(inline_keyboard=buttons)


# ---------------------------------------------------------------------------
# Message formatter
# ---------------------------------------------------------------------------

def format_rag_response(result: RAGResult) -> str:
    """
    Convert a ``RAGResult`` into Telegram HTML (parse_mode='HTML').

    Parameters
    ----------
    result:
        The chain output to render.

    Returns
    -------
    str
        HTML-safe string ≤ 4 096 characters, ready to pass to
        ``message.answer(..., parse_mode='HTML')``.
    """
    answer = result.answer

    # ── Truncate very long answers before escaping ─────────────────────
    if len(answer) > _MAX_ANSWER_CHARS:
        answer = answer[:_MAX_ANSWER_CHARS].rsplit(" ", 1)[0] + " …\n\n<i>(answer truncated)</i>"

    # ── Convert minimal Markdown → HTML ───────────────────────────────
    answer = _md_to_html(answer)

    # ── Inject clickable citation links ───────────────────────────────
    url_map: dict[int, str] = {src.index: src.url for src in result.sources}
    def _replace_citation(m: re.Match) -> str:
        n = int(m.group(1))
        url = url_map.get(n)
        sup = _SUPERSCRIPTS.get(n, f"[{n}]")
        if url:
            return f'<a href="{url}">{sup}</a>'
        return sup

    # Match [N] that were NOT already wrapped in an <a> tag by _md_to_html
    answer = re.sub(r"\[(\d+)\]", _replace_citation, answer)

    # ── Sources footer ─────────────────────────────────────────────────
    if result.sources:
        footer_lines = ["\n\n📚 <b>Sources</b>"]
        for src in result.sources:
            label = html.escape(src.symbol or src.title or src.url)
            sup   = _SUPERSCRIPTS.get(src.index, f"[{src.index}]")
            footer_lines.append(f'{sup} <a href="{src.url}">{label}</a>')
        answer += "\n".join(footer_lines)

    return answer


def format_error(exc: Exception) -> str:
    """Render a user-facing error message."""
    return (
        "❌ <b>Something went wrong.</b>\n\n"
        f"<code>{html.escape(str(exc)[:300])}</code>\n\n"
        "Please try again or rephrase your question."
    )


# ---------------------------------------------------------------------------
# Internal: Markdown → Telegram HTML
# ---------------------------------------------------------------------------

def _md_to_html(text: str) -> str:
    """
    Convert the subset of Markdown commonly produced by the LLM to
    Telegram-compatible HTML.

    Handles (in order):
      1. Fenced code blocks  ``` … ```
      2. Inline code         ` … `
      3. Bold                **…**
      4. HTML-escaping of everything else
    """
    # We process segment by segment to avoid double-escaping code blocks.
    # Strategy: split on ``` boundaries, escape prose, wrap code verbatim.
    parts: list[str] = []
    segments = re.split(r"(```(?:\w+)?\n?.*?```)", text, flags=re.DOTALL)

    for seg in segments:
        if seg.startswith("```"):
            # Extract optional language tag and body
            inner = re.sub(r"^```\w*\n?", "", seg)
            inner = re.sub(r"```$", "", inner)
            parts.append(f"<pre><code>{html.escape(inner)}</code></pre>")
        else:
            # Escape HTML in prose, then apply inline formatting
            seg = html.escape(seg)
            # Inline code: `...`  (escaped backtick is still ` in HTML)
            seg = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", seg)
            # Bold: **...**
            seg = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", seg)
            parts.append(seg)

    return "".join(parts)
