"""
extractor.py
------------
Extracts structured markers from a cleaned BeautifulSoup content Tag.

Produces a ``PageMarkers`` object that captures everything a downstream
RAG pipeline needs beyond raw Markdown:

* ``module``    – Python module/namespace for the page (e.g. ``torch.random``).
* ``symbols``   – Fully-qualified API names defined on the page.
* ``sections``  – Ordered list of ``SectionMarker`` records — one per heading
                  or API-object block — which drives chunk splitting and lets
                  each chunk be annotated with its citation anchor and kind.
* ``keywords``  – Flat, de-duplicated token list (module path parts, symbol
                  short-names, parameter names) for BM25 / keyword search.

Design notes
~~~~~~~~~~~~
``extract_markers()`` accepts a Tag that has *already* been processed by
``cleaner.clean_html()``, so headerlink ``<a>`` tags are gone.  Section
anchors are therefore read from ``<section id="…">`` attributes, which the
cleaner leaves intact.  API anchors come from ``<dt class="sig" id="…">``.

This module is intentionally free of I/O and has no dependency on the
converter or pipeline — it can be unit-tested in isolation against any
BeautifulSoup Tag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from bs4 import Tag


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

# All object-types that Sphinx / autodoc can emit inside a ``dl.py`` block.
_SPHINX_OBJECT_KINDS: frozenset[str] = frozenset({
    "function",
    "class",
    "method",
    "staticmethod",
    "classmethod",
    "attribute",
    "property",
    "data",
    "exception",
    "decorator",
    "decoratormethod",
})


@dataclass
class SectionMarker:
    """
    One chunk-split boundary within a documentation page.

    Attributes
    ----------
    anchor:
        URL fragment that links directly to this section, e.g.
        ``"#module-torch.random"`` or ``"#torch.random.fork_rng"``.
        Prepend the page URL to form a full citation link.
    label:
        Human-readable heading text or API symbol name.
    level:
        Heading depth (1–6).  API-object blocks use ``0`` — they are
        structural boundaries that don't map to a heading hierarchy.
    kind:
        ``"heading"`` for HTML headings; one of the Sphinx object types
        (``"function"``, ``"class"``, ``"method"``, ``"attribute"``, …)
        for API blocks.
    symbol:
        Fully-qualified Python name for API objects (e.g.
        ``"torch.random.fork_rng"``); empty string for plain headings.
    params:
        Ordered list of parameter *names* (no defaults/types) for callable
        API objects. Empty for non-callables and headings.
    source_url:
        GitHub source permalink extracted from the ``[source]`` link
        in the signature block, if present; empty string otherwise.
    """

    anchor: str
    label: str
    level: int
    kind: str
    symbol: str = ""
    params: list[str] = field(default_factory=list)
    source_url: str = ""


@dataclass
class PageMarkers:
    """
    All structural metadata extracted from one documentation page.

    Attributes
    ----------
    module:
        Python module/namespace for this page, inferred from
        ``<section id="module-…">``.  Empty string when unavailable
        (e.g. conceptual/tutorial pages that don't expose an API module).
    symbols:
        Ordered list of fully-qualified API names defined on the page —
        a convenient flat index without navigating ``sections``.
    sections:
        ``SectionMarker`` records in document order.  Every heading and
        every API object block is represented.  Use this to split the
        companion Markdown document into chunks: each marker's ``anchor``
        corresponds to a ``<!-- section: … -->`` or ``<!-- api: … -->``
        comment in the Markdown (injected by the converter).
    keywords:
        De-duplicated token list for BM25 / keyword-search indexing.
        Contains: module path components, qualified symbol names, short
        symbol names (last dotted component), and parameter names.
    """

    module: str
    symbols: list[str]
    sections: list[SectionMarker]
    keywords: list[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_markers(content_el: Tag) -> PageMarkers:
    """
    Extract structural markers from an already-cleaned content Tag.

    Parameters
    ----------
    content_el:
        A ``Tag`` previously processed by ``cleaner.clean_html()``.
        The tree is **not** mutated.

    Returns
    -------
    PageMarkers
        Fully populated markers object.
    """
    module = _extract_module(content_el)
    sections = _extract_sections(content_el)
    symbols = [s.symbol for s in sections if s.symbol]
    keywords = _build_keywords(module, sections)

    return PageMarkers(
        module=module,
        symbols=symbols,
        sections=sections,
        keywords=keywords,
    )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_module(content_el: Tag) -> str:
    """
    Return the Python module name for the page.

    Sphinx emits ``<section id="module-torch.random">`` for module-level
    reference pages.  We strip the ``"module-"`` prefix to get the dotted
    module name.  Returns ``""`` for pages without such a section.
    """
    sec = content_el.find("section", id=re.compile(r"^module-"))
    if sec and isinstance(sec, Tag):
        return sec["id"][len("module-"):]  # type: ignore[index]
    return ""


def _extract_sections(content_el: Tag) -> list[SectionMarker]:
    """
    Walk the document in source order, collecting headings and API blocks.

    Strategy
    --------
    We iterate over the direct children of every ``<section>`` element
    (and the content root itself) to capture headings, then scan all
    ``<dl class="py …">`` blocks for API signatures.  The result is
    de-duplicated and sorted by document position via a single DFS pass.
    """
    markers: list[SectionMarker] = []

    # We visit every element once in tree order. Headings contribute a
    # "heading" marker; dl.py blocks contribute an API marker (skipping
    # nested dl.field-list which appear *inside* dl.py and carry parameter
    # descriptions, not new symbols).
    seen_anchors: set[str] = set()

    def _visit(el: Tag) -> None:
        if not isinstance(el, Tag):
            return

        tag = el.name

        # --- Headings -------------------------------------------------------
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            # Anchor lives on the closest ancestor <section> element because
            # the headerlink <a> tags have been stripped by the cleaner.
            parent_sec = el.find_parent("section")
            anchor = ""
            if isinstance(parent_sec, Tag):
                raw_id = parent_sec.get("id", "")
                # Only use this section's anchor if we haven't seen it yet
                # (multiple headings might share a section, pick the first).
                anchor = f"#{raw_id}" if raw_id else ""

            label = _clean_heading_text(el.get_text(" ", strip=True))

            if anchor and anchor not in seen_anchors:
                seen_anchors.add(anchor)
                markers.append(SectionMarker(
                    anchor=anchor,
                    label=label,
                    level=level,
                    kind="heading",
                ))
            elif not anchor and label:
                # Headings without a resolvable anchor still serve as split
                # points; use the slugified label as a synthetic anchor.
                synthetic = "#" + re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
                if synthetic not in seen_anchors:
                    seen_anchors.add(synthetic)
                    markers.append(SectionMarker(
                        anchor=synthetic,
                        label=label,
                        level=level,
                        kind="heading",
                    ))

        # --- API object blocks (dl.py.*) ------------------------------------
        elif tag == "dl" and "py" in el.get("class", []):
            # Skip field-list dl elements (Parameters / Returns / etc.) which
            # are nested inside dl.py and do NOT introduce new symbols.
            if "field-list" in el.get("class", []):
                return

            kind = _api_kind(el)
            dt = el.find("dt", class_="sig")
            if not isinstance(dt, Tag):
                return

            symbol_id = dt.get("id", "")  # e.g. "torch.random.fork_rng"
            if not symbol_id:
                return

            anchor = f"#{symbol_id}"
            if anchor in seen_anchors:
                return
            seen_anchors.add(anchor)

            label = _clean_symbol_text(dt.get_text("", strip=True))
            params = _extract_params(dt)
            source_url = _extract_source_url(dt)

            markers.append(SectionMarker(
                anchor=anchor,
                label=label,
                level=0,
                kind=kind,
                symbol=symbol_id,
                params=params,
                source_url=source_url,
            ))
            # Do NOT recurse into dl.py children — nested dl.field-list are
            # parameter descriptions, not new structural boundaries.
            return

        # Recurse into children for everything else
        for child in el.children:
            if isinstance(child, Tag):
                _visit(child)

    _visit(content_el)
    return markers


def _api_kind(dl_el: Tag) -> str:
    """
    Extract the Sphinx object type from a ``<dl class="py function">`` etc.

    Returns the first class that matches a known Sphinx object kind, or
    ``"object"`` as a safe fallback.
    """
    for cls in dl_el.get("class", []):
        if cls in _SPHINX_OBJECT_KINDS:
            return cls
    return "object"


def _extract_params(dt_el: Tag) -> list[str]:
    """
    Return the ordered parameter *names* from a signature ``<dt>``.

    Sphinx wraps each parameter in ``<em class="sig-param">`` and the name
    itself in ``<span class="n">``.  We extract the ``span.n`` text to get
    clean names without defaults or type annotations.

    Example
    -------
    ``fork_rng(devices=None, enabled=True)``  ->  ``["devices", "enabled"]``
    """
    params: list[str] = []
    for em in dt_el.find_all("em", class_="sig-param"):
        name_span = em.find("span", class_="n")
        if isinstance(name_span, Tag):
            name = name_span.get_text("", strip=True)
            if name:
                params.append(name)
    return params


def _extract_source_url(dt_el: Tag) -> str:
    """
    Return the GitHub source permalink from a ``[source]`` link, or ``""``.

    The link is an ``<a class="reference external">`` wrapping a
    ``<span class="viewcode-link">`` — present before the cleaner runs.
    After cleaning the viewcode-link span is gone but the ``<a>`` may
    still be present (the cleaner strips ``.viewcode-link`` spans, not
    their parent anchors).
    """
    a = dt_el.find("a", class_="reference")
    if isinstance(a, Tag):
        href = a.get("href", "")
        # Only return links that look like source-code permalinks
        if href and ("github.com" in href or "/blob/" in href):
            return str(href)
    return ""


# ---------------------------------------------------------------------------
# Keyword builder
# ---------------------------------------------------------------------------

def _build_keywords(module: str, sections: Sequence[SectionMarker]) -> list[str]:
    """
    Build a de-duplicated keyword list for BM25 / keyword-search indexing.

    Includes:
    * Module path components  (``torch``, ``random`` from ``torch.random``)
    * Qualified symbol names  (``torch.random.fork_rng``)
    * Short symbol names      (``fork_rng``)
    * Parameter names         (``devices``, ``enabled``, …)
    * Section heading words   (split on non-word characters, length ≥ 3)
    """
    tokens: list[str] = []

    # Module components
    if module:
        tokens.append(module)
        tokens.extend(module.split("."))

    for sec in sections:
        if sec.kind == "heading":
            # Split heading into meaningful words
            words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", sec.label)
            tokens.extend(w for w in words if len(w) >= 3)
        else:
            # Full qualified symbol name
            if sec.symbol:
                tokens.append(sec.symbol)
                # Short name (last component)
                short = sec.symbol.split(".")[-1]
                if short and short != sec.symbol:
                    tokens.append(short)
            # Parameter names
            tokens.extend(sec.params)

    # De-duplicate while preserving first-occurrence order
    seen: set[str] = set()
    unique: list[str] = []
    for tok in tokens:
        if tok and tok not in seen:
            seen.add(tok)
            unique.append(tok)
    return unique


# ---------------------------------------------------------------------------
# Text-cleaning utilities
# ---------------------------------------------------------------------------

# Strips ¶ / # headerlink artefacts and trailing whitespace
_HEADING_TRAIL_RE = re.compile(r"[\s#\u00b6]+$")

# Strips "[source]#" artefacts that may appear at the end of sig get_text()
_SIG_TRAIL_RE = re.compile(r"(\[source\]|[\s#\u00b6])+$")


def _clean_heading_text(raw: str) -> str:
    return _HEADING_TRAIL_RE.sub("", raw).strip()


def _clean_symbol_text(raw: str) -> str:
    """Return the human-readable label for an API signature."""
    cleaned = _SIG_TRAIL_RE.sub("", raw).strip()
    # Re-add space after each comma for readability
    cleaned = re.sub(r",(?!\s)", ", ", cleaned)
    return cleaned
