"""
cleaner.py
----------
Strips PyTorch / Sphinx chrome from a parsed BeautifulSoup tree and
extracts the main content element.

All functions are pure transformations — they receive a soup object and
return (a possibly mutated) soup or a Tag.  No I/O happens here.

Pipeline order matters:
    1. extract_main_content(soup)   — find the article element FIRST
    2. clean_html(article_el)       — strip noise WITHIN the article only

Running clean_html on the full soup before extraction is unsafe because
some page-level wrappers (e.g. #header-holder in the pydata-sphinx-theme)
enclose the entire <body>, and decomposing them would delete the article too.
"""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup, Comment, Tag

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noise catalogue
# ---------------------------------------------------------------------------

# CSS selectors for Sphinx UI artefacts found *inside* the article element.
# These are safe to run on the extracted content tag without risk of nuking
# sibling layout elements.
_NOISE_SELECTORS: list[str] = [
    # Sphinx heading anchors and source-link buttons
    ".headerlink",              # paragraph / # link next to every heading
    ".viewcode-link",           # [source] jump-to-source button
    "a.reference.external + .viewcode-link",
    # On-this-page / secondary TOC that Sphinx sometimes injects into the article
    ".bd-toc",
    ".bd-sidebar-secondary",
    # Search boxes that Sphinx duplicates inside the article region
    "#sphinx-search",
    "#google-search",
    "#searchbox",
    # Version / deprecation banners (kept in admonition form but strip the
    # raw "New in version X" / "Changed in version X" paragraphs if desired)
    # ".versionmodified",  # uncomment to drop version-change notices
    # Misc chrome
    ".edit-on-github",
    "div.clearer",
    ".toggle-header",
    "#cookie-consent-banner",
    "div[role='search']",
    "form.search",
]

# Tags whose entire sub-tree is meaningless for RAG
_DROP_TAGS: frozenset[str] = frozenset({"script", "style", "noscript", "iframe", "svg", "img"})

# Pygments inline span classes — unwrapping them keeps text, drops colour tags.
# Covers single-letter token classes (n, k, o, p, s, c …) plus two-char
# variants and common numeric token types.
_PYGMENTS_SPAN_RE = re.compile(
    r"^(pre|n[a-z]?|k[a-z]?|o[a-z]?|p[a-z]?|s[12a-z]?|mi|mf|mo|mb|c[a-z]?)$"
)

# Selectors tried in order to locate the main content region.
# The pydata-sphinx-theme (PyTorch 2.x) uses:
#     <article id="pytorch-article" class="bd-article">
# which is NOT matched by any of the legacy selectors — it must come first.
_MAIN_CONTENT_SELECTORS: list[str] = [
    # pydata-sphinx-theme (PyTorch 2.x+)
    "article#pytorch-article",
    "article.bd-article",
    # legacy pytorch-sphinx-theme
    "div.pytorch-article",
    # generic Sphinx / RST layouts
    "article[role='main']",
    "div[role='main']",
    "div.body",
    "section#pytorch-documentation",
    "div#pytorch-documentation",
    # last-resort generic containers (may include sidebar chrome)
    "main",
    "div.document",
]

# Selectors tried in order to find the page <h1>.
_H1_SELECTORS: list[str] = [
    "article#pytorch-article h1",
    "article.bd-article h1",
    "div[role='main'] h1",
    "article h1",
    ".pytorch-article h1",
]

# Headerlink artefacts appended to heading text (¶ U+00B6, or literal #).
_HEADING_TRAIL_RE = re.compile(r"[\s#\u00b6]+$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_title(soup: BeautifulSoup) -> str:
    """
    Return the best human-readable title for the page.

    Must be called on the *original* soup, before clean_html() mutates it,
    so that we can find the <h1> before noise elements are removed.

    Falls back to the <title> tag, stripping the " — PyTorch … documentation"
    suffix.  Trailing headerlink artefacts (¶ / #) are always stripped.
    """
    for sel in _H1_SELECTORS:
        el = soup.select_one(sel)
        if el:
            raw = el.get_text(" ", strip=True)
            return _HEADING_TRAIL_RE.sub("", raw).strip()

    title_tag = soup.find("title")
    if title_tag:
        raw = title_tag.get_text(strip=True)
        return re.sub(r"\s*[—–\-]\s*PyTorch.*$", "", raw).strip()

    return "Untitled"


def extract_main_content(soup: BeautifulSoup) -> Tag:
    """
    Return the Tag representing the page's main content region.

    Call this BEFORE clean_html() — the selectors operate on the full
    unmodified tree so that layout wrappers do not confuse the search.

    Falls back gracefully through multiple selector strategies and
    ultimately returns <body> so the caller always receives a Tag.
    """
    for sel in _MAIN_CONTENT_SELECTORS:
        el = soup.select_one(sel)
        if el is not None:
            log.debug("Main content found via selector %r", sel)
            return el

    log.warning("Could not find a main content container; falling back to <body>")
    return soup.find("body") or soup  # type: ignore[return-value]


def clean_html(content: Tag) -> Tag:
    """
    Remove Sphinx UI artefacts from an *already-extracted* content Tag.

    Operates on the article / main-content element, not the full page soup,
    so page-level layout wrappers are never at risk of being decomposed.

    Mutates *content* in-place and returns it so calls can be chained:
        md = html_to_markdown(clean_html(extract_main_content(soup)))
    """
    # 1. Strip HTML comments
    for comment in content.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # 2. Drop entire tag families (scripts, styles, media …)
    for tag_name in _DROP_TAGS:
        for tag in content.find_all(tag_name):
            tag.decompose()

    # 3. Drop noisy Sphinx UI elements
    for sel in _NOISE_SELECTORS:
        for el in content.select(sel):
            el.decompose()

    # 4. Unwrap Pygments inline colour spans — keep text, lose the tag
    for span in content.find_all("span", class_=_PYGMENTS_SPAN_RE):
        span.unwrap()

    return content
