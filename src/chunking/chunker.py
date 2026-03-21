"""
chunker.py
----------
Splits a processed DocPage into retrieval-ready Chunk objects.

Split strategy
~~~~~~~~~~~~~~
1.  **Primary split** on the ``<!-- section: … -->`` and ``<!-- api: … -->``
    HTML comments that the converter embeds at every structural boundary.
    Each comment maps 1-to-1 to a ``SectionMarker`` in ``page.markers``,
    so every chunk inherits rich metadata with zero extra parsing.

2.  **Merge forward** — segments shorter than ``min_chars`` are prepended to
    the *following* segment.  In practice this catches module-header stubs
    (e.g. ``# torch.random\\n\\nCreated On: Aug 07, 2019 …``) and short
    section intros that carry no standalone retrieval signal.
    The absorbing segment keeps its own anchor/citation URL; the prefix
    text provides context for the embedding model.

3.  **Sub-split** — segments longer than ``max_chars`` are sliced at
    paragraph boundaries (``\\n\\n``) with a configurable ``overlap_chars``
    tail prepended to each continuation sub-chunk, preserving cross-boundary
    context.  Code fences (``` … ```) are treated as atomic paragraphs: they
    are never split mid-fence.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Sequence

try:
    from ..data_acquisition.pipeline import DocPage
    from ..data_acquisition.extractor import PageMarkers, SectionMarker
except ImportError:
    from data_acquisition.pipeline import DocPage          # type: ignore[no-redef]
    from data_acquisition.extractor import PageMarkers, SectionMarker  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Regex: matches both ``<!-- section: anchor-id -->``
#        and     ``<!-- api: fully.qualified.name -->``
# Capturing groups: (kind, anchor_id)
# ---------------------------------------------------------------------------
_MARKER_RE = re.compile(r"<!--\s*(section|api):\s*([^\s>]+)\s*-->")

# Paragraph boundary: two or more newlines.
_PARA_SPLIT_RE = re.compile(r"\n{2,}")

# Fenced code block detector (to avoid splitting inside them).
_FENCE_RE = re.compile(r"^```", re.MULTILINE)


# ---------------------------------------------------------------------------
# Chunk dataclass — the unit passed to embedding / vector-store layers
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    One retrieval unit for the RAG pipeline.

    Attributes
    ----------
    chunk_id:
        Deterministic string ID built from the anchor slug and a short hash
        of the page URL, so IDs are stable across re-runs and unique across
        pages even when anchors collide (e.g. two pages with ``#overview``).
        Format: ``"{anchor_slug}__{sub_index}__{url_hash6}"``.

    text:
        Markdown content ready for embedding.  Does **not** include the
        leading ``<!-- marker -->`` comment.  Continuation sub-chunks are
        prefixed with the tail of the previous sub-chunk (overlap window).

    page_url:
        Canonical source URL of the parent documentation page.

    citation_url:
        Direct deep-link for source attribution: ``page_url + anchor``.
        For continuation sub-chunks the anchor still points to the section
        start (not the middle), which is the most useful citation target.

    anchor:
        URL fragment, e.g. ``"#torch.random.fork_rng"``.

    page_title:
        Human-readable title of the parent page (from ``DocPage.title``).

    section:
        Top-level section inferred from the URL path, e.g. ``"random"`` or
        ``"generated"``.

    kind:
        ``"heading"`` for prose/conceptual sections; one of the Sphinx object
        types (``"function"``, ``"class"``, ``"method"``, ``"attribute"``, …)
        for API-reference chunks.

    symbol:
        Fully-qualified Python name for API chunks, e.g.
        ``"torch.random.fork_rng"``.  Empty string for prose/heading chunks.

    keywords:
        Union of the page-level BM25 token list and this chunk's own
        symbol components + parameter names.  Feed directly into a BM25
        index or sparse retriever.

    params:
        Ordered parameter names for callable API chunks; empty list otherwise.
        Useful for function-signature–aware retrieval ("which function takes a
        ``device_type`` argument?").

    source_url:
        GitHub source permalink extracted from the ``[source]`` link, if the
        signature block had one.  Empty string otherwise.

    char_count:
        ``len(text)`` — computed automatically in ``__post_init__``.

    is_continuation:
        ``True`` when this chunk is a sub-split (Nth piece of an oversized
        section).  Useful for filtering or boosting in retrieval.

    sub_index:
        ``0`` for the primary (or only) chunk from a section; ``1``, ``2``, …
        for continuation pieces produced by sub-splitting.
    """

    chunk_id: str
    text: str
    page_url: str
    citation_url: str
    anchor: str
    page_title: str
    section: str
    kind: str
    symbol: str
    keywords: list[str]
    params: list[str]
    source_url: str
    char_count: int = field(init=False)
    is_continuation: bool = False
    sub_index: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable dict for vector-store upsert.
        """
        return {
            "chunk_id":        self.chunk_id,
            "text":            self.text,
            "page_url":        self.page_url,
            "citation_url":    self.citation_url,
            "anchor":          self.anchor,
            "page_title":      self.page_title,
            "section":         self.section,
            "kind":            self.kind,
            "symbol":          self.symbol,
            "keywords":        self.keywords,
            "params":          self.params,
            "source_url":      self.source_url,
            "char_count":      self.char_count,
            "is_continuation": self.is_continuation,
            "sub_index":       self.sub_index,
        }


# ---------------------------------------------------------------------------
# ChunkSplitter
# ---------------------------------------------------------------------------

class ChunkSplitter:
    """
    Converts a ``DocPage`` into a list of ``Chunk`` objects.

    Parameters
    ----------
    max_chars:
        Maximum characters per chunk.  Segments exceeding this are sub-split
        at paragraph boundaries.  Default 1 500 ≈ 375 tokens.
    min_chars:
        Segments below this threshold are merged into the next segment rather
        than emitted as standalone chunks.  Default 120.
    overlap_chars:
        When sub-splitting, this many characters from the *tail* of the
        preceding sub-chunk are prepended to the next one.  The overlap is
        drawn from whole paragraphs (never split mid-sentence) and is marked
        with a ``<!-- overlap -->`` comment so it can be stripped before
        display if desired.  Default 200.
    """

    def __init__(
        self,
        max_chars: int = 1_500,
        min_chars: int = 120,
        overlap_chars: int = 200,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if min_chars < 0:
            raise ValueError("min_chars must be non-negative")
        if overlap_chars < 0 or overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be in [0, max_chars)")

        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_chars = overlap_chars

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def split(self, page: DocPage) -> list[Chunk]:
        """
        Split a ``DocPage`` into ``Chunk`` objects.

        Parameters
        ----------
        page:
            A fully processed ``DocPage`` (markdown + markers populated).

        Returns
        -------
        list[Chunk]
            Ordered list of chunks, preserving document order.
            An empty list is returned if the page markdown is empty or
            contains no recognisable split markers.
        """
        # Step 1: Parse raw segments from the embedded marker comments.
        raw_segments = _parse_markers(page.markdown)
        if not raw_segments:
            # Fallback: treat the entire page as one chunk with a synthetic anchor.
            return self._fallback_chunk(page)

        # Step 2: Build anchor -> SectionMarker lookup from page metadata.
        marker_index = _build_marker_index(page.markers)

        # Step 3: Merge undersized segments forward so stubs don't pollute the index.
        merged = _merge_small(raw_segments, self.min_chars)

        # Step 4: Sub-split oversized segments and materialise Chunk objects.
        chunks: list[Chunk] = []
        for anchor, text, raw_kind in merged:
            marker = marker_index.get(anchor)
            sub_texts = _sub_split(text, self.max_chars, self.overlap_chars)
            for sub_idx, sub_text in enumerate(sub_texts):
                chunks.append(
                    _make_chunk(
                        page=page,
                        anchor=anchor,
                        marker=marker,
                        raw_kind=raw_kind,
                        text=sub_text,
                        sub_index=sub_idx,
                    )
                )

        return chunks

    def _fallback_chunk(self, page: DocPage) -> list[Chunk]:
        """Return a single whole-page chunk when no markers are found."""
        anchor = "#top"
        url_hash = _url_hash(page.url)
        return [
            Chunk(
                chunk_id=f"top__0__{url_hash}",
                text=page.markdown.strip(),
                page_url=page.url,
                citation_url=page.url,
                anchor=anchor,
                page_title=page.title,
                section=page.section,
                kind="heading",
                symbol="",
                keywords=list(page.keywords),
                params=[],
                source_url="",
            )
        ]


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _parse_markers(markdown: str) -> list[tuple[str, str, str]]:
    """
    Split markdown on embedded ``<!-- section/api: id -->`` comments.

    Returns
    -------
    list of (anchor, text, raw_kind) tuples where:

    * ``anchor``   — ``"#" + id``, e.g. ``"#torch.random.fork_rng"``
    * ``text``     — markdown content *following* this marker up to the next one
    * ``raw_kind`` — ``"section"`` or ``"api"`` (from the comment itself)

    Text before the very first marker is silently discarded — it is always
    empty in well-formed converter output.
    """
    # re.split with capturing groups produces:
    #   [pre, kind₁, id₁, text₁, kind₂, id₂, text₂, …]
    parts = _MARKER_RE.split(markdown)

    segments: list[tuple[str, str, str]] = []
    # parts[0] is pre-marker preamble (discard); then groups of 3 follow.
    for i in range(1, len(parts) - 2, 3):
        raw_kind = parts[i]          # "section" or "api"
        anchor_id = parts[i + 1]     # e.g. "torch.random.fork_rng"
        text = parts[i + 2].strip()
        segments.append((f"#{anchor_id}", text, raw_kind))

    return segments


def _build_marker_index(markers: PageMarkers) -> dict[str, SectionMarker]:
    """Return an anchor-keyed lookup from a page's structural markers."""
    return {sec.anchor: sec for sec in markers.sections}


def _merge_small(
    segments: list[tuple[str, str, str]],
    min_chars: int,
) -> list[tuple[str, str, str]]:
    """
    Merge segments shorter than ``min_chars`` into the *following* segment.

    The short segment's text is prepended as a context prefix; the following
    segment's anchor, kind, and SectionMarker are kept — they represent the
    primary retrieval target.

    If the very last segment is undersized and there is no following segment
    to absorb it, it is merged *backward* into the previous one instead.

    Parameters
    ----------
    segments:
        List of ``(anchor, text, raw_kind)`` tuples from ``_parse_markers``.
    min_chars:
        Minimum character count to qualify as a standalone chunk.

    Returns
    -------
    list of ``(anchor, text, raw_kind)`` tuples, guaranteed non-empty if the
    input is non-empty.
    """
    if not segments:
        return segments

    # Forward pass: carry undersized text into the next segment.
    carry_text: str = ""
    result: list[tuple[str, str, str]] = []

    for anchor, text, raw_kind in segments:
        combined = (carry_text + "\n\n" + text).strip() if carry_text else text

        if len(combined) < min_chars:
            # Still too small — keep accumulating.
            carry_text = combined
            continue

        result.append((anchor, combined, raw_kind))
        carry_text = ""

    # Flush remaining carry: merge backward into the last accepted segment.
    if carry_text:
        if result:
            last_anchor, last_text, last_kind = result[-1]
            result[-1] = (last_anchor, (last_text + "\n\n" + carry_text).strip(), last_kind)
        else:
            # Edge case: entire page is one tiny segment.
            first_anchor, _, first_kind = segments[0]
            result.append((first_anchor, carry_text, first_kind))

    return result


def _sub_split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """
    Split ``text`` into sub-chunks of at most ``max_chars`` characters,
    splitting only at paragraph boundaries (``\\n\\n``).

    Code fences (``` … ```) are treated as single atomic paragraphs: a fence
    is never broken mid-block, even if it exceeds ``max_chars`` on its own.
    In that case it occupies its own sub-chunk regardless of size.

    Overlap
    ~~~~~~~
    When a new sub-chunk starts, the *tail* of the previous sub-chunk is
    prepended as an ``<!-- overlap -->``-delimited prefix.  The overlap is
    assembled from whole trailing paragraphs of the previous window up to
    ``overlap_chars`` characters.

    Parameters
    ----------
    text:
        The segment text to split.
    max_chars:
        Hard upper limit (in characters) per sub-chunk, before overlap is added.
    overlap_chars:
        Target character budget for the overlap prefix.

    Returns
    -------
    list[str]
        One or more sub-chunk strings.  If ``text`` fits within ``max_chars``,
        a single-element list is returned (no overhead).
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = _split_paragraphs(text)
    sub_chunks: list[str] = []
    window: list[str] = []       # paragraphs in the current sub-chunk
    window_len: int = 0

    def _flush() -> None:
        if window:
            sub_chunks.append("\n\n".join(window))

    for para in paragraphs:
        para_len = len(para)

        # If a single paragraph exceeds max_chars, emit it alone.
        if para_len > max_chars:
            _flush()
            window = []
            window_len = 0
            sub_chunks.append(para)
            continue

        # Would adding this paragraph exceed the budget?
        # (+2 for the \n\n separator)
        would_be = window_len + (2 if window else 0) + para_len
        if would_be > max_chars and window:
            _flush()
            # Build overlap prefix from the tail of the completed window.
            overlap_paras = _tail_overlap(window, overlap_chars)
            window = overlap_paras
            window_len = sum(len(p) for p in window) + max(0, 2 * (len(window) - 1))

        window.append(para)
        window_len += (2 if len(window) > 1 else 0) + para_len

    _flush()

    # Annotate continuation sub-chunks with an overlap marker so callers
    # can detect and optionally strip the prefix region.
    if len(sub_chunks) > 1 and overlap_chars > 0:
        result: list[str] = [sub_chunks[0]]
        for sub in sub_chunks[1:]:
            result.append("<!-- overlap -->\n\n" + sub)
        return result

    return sub_chunks


# ---------------------------------------------------------------------------
# Internal sub-split helpers
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs on double-newline boundaries, treating
    fenced code blocks as single atomic paragraphs.
    """
    raw = _PARA_SPLIT_RE.split(text)
    merged: list[str] = []
    fence_buf: list[str] = []
    inside_fence = False

    for block in raw:
        if _FENCE_RE.match(block):
            if not inside_fence:
                # Opening fence — start accumulating.
                inside_fence = True
                fence_buf = [block]
            else:
                # Closing fence — flush the atomic block.
                fence_buf.append(block)
                merged.append("\n\n".join(fence_buf))
                fence_buf = []
                inside_fence = False
        elif inside_fence:
            fence_buf.append(block)
        else:
            if block.strip():
                merged.append(block)

    # Handle unclosed fence (malformed markdown).
    if fence_buf:
        merged.append("\n\n".join(fence_buf))

    return merged


def _tail_overlap(paragraphs: list[str], overlap_chars: int) -> list[str]:
    """
    Return the tail of ``paragraphs`` whose total length fits within
    ``overlap_chars``.  Walks backwards to collect whole paragraphs.

    Returns an empty list if ``overlap_chars`` is 0.
    """
    if not overlap_chars:
        return []

    tail: list[str] = []
    budget = overlap_chars

    for para in reversed(paragraphs):
        if len(para) <= budget:
            tail.insert(0, para)
            budget -= len(para) + 2  # +2 for \n\n
        else:
            break

    return tail


# ---------------------------------------------------------------------------
# Chunk factory
# ---------------------------------------------------------------------------

def _make_chunk(
    page: DocPage,
    anchor: str,
    marker: SectionMarker | None,
    raw_kind: str,
    text: str,
    sub_index: int,
) -> Chunk:
    """
    Construct a ``Chunk`` from resolved segment data.

    ``marker`` may be ``None`` when the anchor from the markdown comment was
    not found in ``page.markers`` (e.g. the extractor skipped it due to a
    duplicate or malformed signature).  In that case we fall back to the
    ``raw_kind`` from the comment and leave symbol/params empty.
    """
    # Build a combined, de-duplicated keyword list:
    #   page-level keywords  +  this chunk's symbol components + params.
    keywords = _build_chunk_keywords(page.keywords, marker)

    # Derive metadata from the SectionMarker when available.
    kind       = marker.kind       if marker else ("heading" if raw_kind == "section" else "object")
    symbol     = marker.symbol     if marker else ""
    params     = marker.params     if marker else []
    source_url = marker.source_url if marker else ""

    # Deterministic chunk ID: anchor slug + sub-index + 6-char URL hash.
    url_hash    = _url_hash(page.url)
    anchor_slug = re.sub(r"[^a-zA-Z0-9]+", "_", anchor.lstrip("#")).strip("_")
    chunk_id    = f"{anchor_slug}__{sub_index}__{url_hash}"

    return Chunk(
        chunk_id=chunk_id,
        text=text,
        page_url=page.url,
        citation_url=page.url.rstrip("/") + anchor,
        anchor=anchor,
        page_title=page.title,
        section=page.section,
        kind=kind,
        symbol=symbol,
        keywords=keywords,
        params=params,
        source_url=source_url,
        is_continuation=sub_index > 0,
        sub_index=sub_index,
    )


def _build_chunk_keywords(
    page_keywords: Sequence[str],
    marker: SectionMarker | None,
) -> list[str]:
    """
    Build a de-duplicated keyword list for a chunk.

    Starts from the page-level keyword list (already contains module
    components and all symbol names) then appends any symbol-specific
    components and parameter names that aren't already present.
    """
    seen: set[str] = set(page_keywords)
    result: list[str] = list(page_keywords)

    if marker and marker.symbol:
        extras: list[str] = (
            [marker.symbol]
            + marker.symbol.split(".")
            + [marker.symbol.split(".")[-1]]   # short name
            + marker.params
        )
        for tok in extras:
            if tok and tok not in seen:
                result.append(tok)
                seen.add(tok)

    return result


def _url_hash(url: str) -> str:
    """Return a 6-character hex digest of the URL, used in chunk IDs."""
    return hashlib.md5(url.encode()).hexdigest()[:6]
