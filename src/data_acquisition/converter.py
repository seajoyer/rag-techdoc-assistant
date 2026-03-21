"""
converter.py
------------
Converts a cleaned BeautifulSoup Tag into well-formed Markdown.

Extends markdownify with PyTorch / Sphinx-specific handling:
  - Renders Sphinx API signatures (dl.py > dt.sig) as fenced Python code blocks
  - Detects fenced-block language from ancestor highlight-<lang> divs
  - Converts field-list <dl>/<dt>/<dd> (Parameters / Returns / Return type)
  - Renders admonition divs (note / warning / tip …) as Markdown blockquotes
  - Normalises whitespace / blank lines

Chunk-split markers
~~~~~~~~~~~~~~~~~~~
Every structural boundary — headings and API-object blocks — is preceded by
an HTML comment of the form::

    <!-- section: module-torch.random -->
    ## torch.random

    <!-- api: torch.random.fork_rng -->
    ```python
    torch.random.fork_rng(devices=None, ...)
    ```

These comments are invisible in rendered Markdown but give the downstream
chunker unambiguous, machine-readable split points without relying on
fragile text-pattern heuristics.  They also carry the anchor fragment
needed for citation link construction:

    base_url + "#" + anchor  ->  https://pytorch.org/…#torch.random.fork_rng
"""

from __future__ import annotations

import re

from bs4 import Tag
from markdownify import MarkdownConverter


# ---------------------------------------------------------------------------
# Custom converter
# ---------------------------------------------------------------------------

class PyTorchMarkdownConverter(MarkdownConverter):
    """markdownify subclass tuned for PyTorch / Sphinx-generated HTML."""

    # -- <pre> / <code> ------------------------------------------------------

    def convert_pre(self, el: Tag, text: str, **kwargs) -> str:  # type: ignore[override]
        """
        Render <pre> as a fenced code block.

        Language is inferred from an ancestor ``<div class="highlight-python …">``
        (Pygments wrapping) rather than from the <code> child — after the
        cleaner has unwrapped Pygments spans the code text is already plain.
        """
        lang = _detect_language_from_pre(el)
        code = text.strip()
        return f"\n\n```{lang}\n{code}\n```\n\n"

    def convert_code(self, el: Tag, text: str, convert_as_inline: bool = False, **kwargs) -> str:
        """
        Inline <code> -> backtick span.

        When <code> is a direct child of <pre>, convert_pre() owns the
        fencing — we return the raw text so it is not double-wrapped.
        """
        if el.parent and el.parent.name == "pre":
            return text  # convert_pre handles the block
        content = text.strip()
        return f"`{content}`" if content else ""

    # -- Sphinx API signatures and field lists -------------------------------

    def convert_dt(self, el: Tag, text: str, **kwargs) -> str:  # type: ignore[override]
        """
        Two distinct <dt> flavours appear in PyTorch Sphinx docs:

        **Signature dt** — ``<dt class="sig sig-object py" id="torch.fn">``
            Rendered as a fenced ``python`` code block so the full signature
            is clearly distinguished from prose and is syntax-highlightable.
            get_text() is used instead of `text` to avoid markdownify wrapping
            ``<em class="sig-param">`` children in *italic* markers.

            A ``<!-- api: {id} -->`` comment is prepended so the downstream
            chunker can split on API-object boundaries without text heuristics.

        **Field-list dt** — ``<dt class="field-odd">`` / ``<dt class="field-even">``
            Labels like "Parameters", "Returns", "Return type".
            Rendered as a bold inline label without a trailing colon.
        """
        classes = el.get("class", [])

        if "sig" in classes:
            # get_text("") gives compact text without inter-tag spaces,
            # which reads like real Python: fork_rng(devices=None, ...)
            raw = el.get_text("", strip=True)
            # Re-add a space after each comma for readability
            raw = re.sub(r",(?!\s)", ", ", raw)
            # Strip any residual headerlink character (# or pilcrow ¶) and
            # the [source] link text added by Sphinx
            raw = re.sub(r"(\[source\]|[\s#\u00b6])+$", "", raw).strip()

            # Emit a machine-readable chunk-split marker before the fenced block.
            symbol_id = el.get("id", "")
            marker = f"\n<!-- api: {symbol_id} -->\n" if symbol_id else ""

            return f"{marker}\n```python\n{raw}\n```\n"

        # Field-list label — strip trailing colon added by a <span class="colon">
        label = re.sub(r":$", "", text.strip())
        return f"\n**{label}**\n"

    def convert_dd(self, el: Tag, text: str, **kwargs) -> str:  # type: ignore[override]
        """
        Render <dd> as a plain block.

        For field-list <dd> elements (parameter / return descriptions) the
        inner content is already a <ul> which markdownify renders as a proper
        list — we just ensure clean surrounding newlines.
        """
        return f"\n{text.strip()}\n\n"

    # -- Headings with chunk-split markers -----------------------------------

    def convert_hN(self, n: int, el: Tag, text: str, parent_tags=None, **kwargs) -> str:  # type: ignore[override]
        """
        Render ``<hN>`` as an ATX heading, prepending a ``<!-- section: … -->``
        chunk-split marker derived from the closest ancestor ``<section id="…">``.

        The headerlink ``<a>`` tags are stripped by the cleaner before this
        method is called, so we recover the anchor from the ``<section>`` id
        instead — which is always the canonical fragment for that section.
        """
        parent_sec = el.find_parent("section")
        anchor = ""
        if isinstance(parent_sec, Tag):
            raw_id = parent_sec.get("id", "")
            if raw_id:
                anchor = raw_id

        marker = f"\n<!-- section: {anchor} -->\n" if anchor else ""

        # Delegate heading text rendering to the base class (handles ATX /
        # underline / closed-ATX styles via the heading_style option).
        base = super().convert_hN(n, el, text, parent_tags)
        return marker + base

    # -- Tables --------------------------------------------------------------

    def convert_table(self, el: Tag, text: str, convert_as_inline: bool = False, **kwargs) -> str:  # type: ignore[override]
        result = super().convert_table(el, text, **kwargs)
        return f"\n{result.strip()}\n\n"

    # -- Admonitions ---------------------------------------------------------

    def convert_div(self, el: Tag, text: str, convert_as_inline: bool = False, **kwargs) -> str:  # type: ignore[override]
        """
        Convert Sphinx admonition divs to Markdown blockquotes.

        Sphinx emits:
            <div class="admonition note">
              <p class="admonition-title">Note</p>
              <p>… body …</p>
            </div>

        The admonition-title text is already included in `text` so we strip
        the first line if it duplicates the label, avoiding double-printing.
        """
        classes = el.get("class", [])
        admonition_map = {
            "note":           "Note",
            "warning":        "Warning",
            "tip":            "Tip",
            "deprecated":     "Deprecated",
            "versionadded":   "Version added",
            "versionchanged": "Version changed",
            "danger":         "Danger",
            "caution":        "Caution",
            "important":      "Important",
        }

        for cls, default_label in admonition_map.items():
            if cls in classes:
                # Prefer the explicit admonition-title element if present
                title_el = el.find("p", class_="admonition-title")
                label = title_el.get_text(strip=True) if title_el else default_label

                # Remove the duplicated title from the converted body
                body_lines = text.strip().splitlines()
                if body_lines and body_lines[0].strip() == label:
                    body_lines = body_lines[1:]

                body = "\n".join(f"> {line}" for line in "\n".join(body_lines).strip().splitlines())
                return f"\n> **{label}**\n{body}\n\n"

        return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def html_to_markdown(content_el: Tag) -> str:
    """
    Convert a cleaned BeautifulSoup element to normalised Markdown.

    The output includes ``<!-- section: … -->`` and ``<!-- api: … -->``
    HTML comments at every structural boundary (headings and API-object
    blocks).  These comments are invisible when rendered but act as
    machine-readable chunk-split points for the RAG pipeline.

    Parameters
    ----------
    content_el:
        A Tag that has already been processed by cleaner.clean_html().

    Returns
    -------
    str
        UTF-8 Markdown string, ready to be written to disk or chunked.
    """
    converter = PyTorchMarkdownConverter(
        heading_style="ATX",    # ## headings, not underline style
        bullets="-",            # consistent unordered list marker
        strip=["a"],            # drop bare <a> wrappers, keep their text
    )
    md = converter.convert_soup(content_el)
    return _normalise_whitespace(md)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Maps Pygments / highlight.js class fragments -> fenced-block language ids
_LANG_CLASS_MAP: dict[str, str] = {
    "python":   "python",
    "py":       "python",
    "cpp":      "cpp",
    "c++":      "cpp",
    "bash":     "bash",
    "sh":       "bash",
    "shell":    "bash",
    "text":     "text",
    "output":   "text",
    "default":  "text",
}


def _detect_language_from_pre(pre_el: Tag) -> str:
    """
    Infer a fenced-block language hint for a <pre> element.

    Strategy (in priority order):
    1. Walk up ancestor tags looking for ``<div class="highlight-<lang> …">``.
       Pygments wraps every code block in such a div, e.g.
       ``<div class="highlight-python notranslate">``.
    2. Check classes on an inner ``<code>`` child.
    3. Default to "python" — the dominant language in PyTorch documentation.
    """
    # 1. Ancestor highlight div (stop at article/section boundary)
    for parent in pre_el.parents:
        if not isinstance(parent, Tag):
            continue
        for cls in parent.get("class", []):
            if cls.startswith("highlight-"):
                fragment = cls[len("highlight-"):]
                # "notranslate" is a utility class, not a language
                if fragment in ("notranslate",):
                    continue
                return _LANG_CLASS_MAP.get(fragment, fragment)
        if parent.name in ("article", "section", "body"):
            break

    # 2. Inner <code> element
    code_el = pre_el.find("code")
    if isinstance(code_el, Tag):
        for cls in code_el.get("class", []):
            fragment = cls.lower().replace("language-", "").replace("highlight-", "")
            if fragment in _LANG_CLASS_MAP:
                return _LANG_CLASS_MAP[fragment]

    return "python"


def _normalise_whitespace(text: str) -> str:
    """Collapse 3+ consecutive blank lines -> 2, strip trailing spaces per line."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()
