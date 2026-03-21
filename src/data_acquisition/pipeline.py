"""
pipeline.py
-----------
Orchestrates the full data-acquisition pipeline:
  discovery -> fetch -> clean -> extract -> convert -> save

Also owns the DocPage dataclass (the single data contract between all modules)
and the JSONL index manifest.

This module is intentionally import-friendly for notebooks: every public
function has clear parameters with defaults and returns plain Python objects.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from .cleaner import clean_html, extract_main_content, extract_title
from .converter import html_to_markdown
from .discovery import fetch_url_list
from .extractor import PageMarkers, SectionMarker, extract_markers
from .fetcher import RateLimitedFetcher

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DocPage:
    """
    One processed documentation page.

    Attributes
    ----------
    url:        Canonical source URL.
    title:      Human-readable page title.
    section:    Top-level section inferred from the URL path (e.g. "nn", "torch").
    markdown:   Full page content in Markdown, including ``<!-- section: … -->``
                and ``<!-- api: … -->`` chunk-split comments at every structural
                boundary.
    markers:    Structural metadata extracted from the HTML before conversion:
                headings, API symbols, parameter names, and the flat keyword
                list for BM25 search.  See ``extractor.PageMarkers``.
    char_count: len(markdown) — handy for quick quality checks.
    saved_path: Path relative to the output directory where the .md was written.
                Empty string if the page has not been saved yet.
    """

    url: str
    title: str
    section: str
    markdown: str
    markers: PageMarkers
    char_count: int = field(init=False)
    saved_path: str = ""

    def __post_init__(self) -> None:
        self.char_count = len(self.markdown)

    def to_index_record(self) -> dict:
        """
        Return a JSON-serialisable dict suitable for the JSONL index.

        The full ``markdown`` field is excluded to keep the index compact
        (the .md files on disk are the source of truth for content).

        The ``markers`` dataclass is serialised in full so the index can
        drive chunk splitting, BM25 indexing, and citation link construction
        without re-parsing the Markdown.

        Schema example (one line of the JSONL file)
        --------------------------------------------
        {
          "url": "https://pytorch.org/docs/stable/random.html",
          "title": "torch.random",
          "section": "random",
          "char_count": 4821,
          "saved_path": "random.md",
          "markers": {
            "module": "torch.random",
            "symbols": ["torch.random.fork_rng", "torch.random.manual_seed", ...],
            "keywords": ["torch", "random", "fork_rng", "devices", "enabled", ...],
            "sections": [
              {
                "anchor": "#module-torch.random",
                "label": "torch.random",
                "level": 1,
                "kind": "heading",
                "symbol": "",
                "params": [],
                "source_url": ""
              },
              {
                "anchor": "#torch.random.fork_rng",
                "label": "torch.random.fork_rng(devices=None, enabled=True, ...)",
                "level": 0,
                "kind": "function",
                "symbol": "torch.random.fork_rng",
                "params": ["devices", "enabled", "_caller", "_devices_kw", "device_type"],
                "source_url": "https://github.com/pytorch/pytorch/blob/v2.10.0/torch/random.py#L129"
              },
              ...
            ]
          }
        }
        """
        d = asdict(self)
        d.pop("markdown")
        return d

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        """Shortcut: fully-qualified API symbol names defined on this page."""
        return self.markers.symbols

    @property
    def keywords(self) -> list[str]:
        """Shortcut: BM25 keyword token list for this page."""
        return self.markers.keywords

    def citation_url(self, marker: SectionMarker) -> str:
        """
        Build a full citation URL for a specific section or API object.

        Parameters
        ----------
        marker:
            A ``SectionMarker`` from ``self.markers.sections``.

        Returns
        -------
        str
            Absolute URL pointing directly at the section, e.g.
            ``"https://pytorch.org/docs/stable/random.html#torch.random.fork_rng"``.
        """
        return self.url.rstrip("/") + marker.anchor


# ---------------------------------------------------------------------------
# Single-page processing
# ---------------------------------------------------------------------------

def process_page(url: str, fetcher: RateLimitedFetcher) -> DocPage | None:
    """
    Fetch, clean, extract markers from, and convert a single documentation page.

    The HTML tree is processed in this order:

    1. ``extract_title()``       — from the unmodified soup (must come first).
    2. ``extract_main_content()``— from the unmodified soup.
    3. ``clean_html()``          — mutates the article Tag in-place.
    4. ``extract_markers()``     — reads the cleaned Tag; does NOT mutate it.
    5. ``html_to_markdown()``    — reads the cleaned Tag; does NOT mutate it.

    Steps 4 and 5 both operate on the same post-clean Tag so there is no
    redundant parse; order between them is arbitrary.

    Parameters
    ----------
    url:
        Absolute URL of the page to process.
    fetcher:
        Shared RateLimitedFetcher instance (reuses connection pool).

    Returns
    -------
    DocPage or None
        None is returned (and a warning is logged) when the page yields
        less than 50 characters of content — usually a redirect or stub.
    """
    try:
        resp = fetcher.get(url)
    except RuntimeError as exc:
        log.error("Skipping %s: %s", url, exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # 1 & 2: Extract title and content from unmodified soup first.
    #        clean_html() must run on the extracted article — NOT on the
    #        full soup — because some page-level wrappers enclose the
    #        entire <body> and would delete the article if decomposed globally.
    title = extract_title(soup)
    content_el = extract_main_content(soup)

    # 3: Clean in-place (strips UI chrome, Pygments spans, comments, etc.)
    clean_html(content_el)

    # 4 & 5: Both operate on the same cleaned Tag — no redundant parse.
    markers = extract_markers(content_el)
    markdown = html_to_markdown(content_el)

    if len(markdown) < 50:
        log.warning("Very short content (%d chars) at %s — skipping", len(markdown), url)
        return None

    return DocPage(
        url=url,
        title=title,
        section=_infer_section(url),
        markdown=markdown,
        markers=markers,
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_page(page: DocPage, output_dir: Path) -> Path:
    """
    Write page.markdown to disk as a .md file.

    The path mirrors the URL structure:
      .../docs/stable/nn.html  ->  <output_dir>/nn.md
      .../docs/stable/torch/index.html  ->  <output_dir>/torch/index.md

    Sets page.saved_path (relative str) as a side-effect.

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    rel = _url_to_rel_path(page.url)
    dest = output_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(page.markdown, encoding="utf-8")
    page.saved_path = str(rel)
    return dest


def save_index(
    records: Iterable[dict],
    index_path: Path,
    append: bool = False,
) -> None:
    """Write (or append to) a JSONL metadata index."""
    mode = "a" if append else "w"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Index %s -> %s", "appended to" if append else "written", index_path)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    output_dir: str | Path = "pytorch_docs",
    max_pages: int | None = 200,
    requests_per_second: float = 2.0,
    on_page_saved: Callable[[DocPage], None] | None = None,
    resume: bool = True,           # ← NEW
) -> list[DocPage]:
    """
    End-to-end data acquisition pipeline.

    Parameters
    ----------
    output_dir:
        Root directory for .md files and the _index.jsonl manifest.
    max_pages:
        Maximum number of pages to fetch.  None = no limit.
    requests_per_second:
        Polite rate cap.
    on_page_saved:
        Optional callback fired after each page is saved — useful for
        notebook progress cells, e.g. ``on_page_saved=lambda p: display(p.title)``.
    resume:
        True  (default) – read _index.jsonl, skip already-processed URLs,
                          append new records to the index.
        False           – process all URLs, overwrite the index.

    Returns
    -------
    list[DocPage]
        All successfully processed pages.
    """
    output_dir = Path(output_dir)
    index_path = output_dir / "_index.jsonl"

    all_urls = fetch_url_list(max_pages=max_pages)

    if resume:
        processed_urls = _load_processed_urls(index_path)
        urls_to_fetch  = [u for u in all_urls if u not in processed_urls]
        if processed_urls:
            log.info(
                "Resume mode: %d/%d URLs already done, fetching %d new.",
                len(processed_urls), len(all_urls), len(urls_to_fetch),
            )
    else:
        urls_to_fetch = all_urls

    new_pages: list[DocPage] = []

    if urls_to_fetch:
        with RateLimitedFetcher(requests_per_second=requests_per_second) as fetcher:
            for url in tqdm(urls_to_fetch, desc="Acquiring docs", unit="page"):
                page = process_page(url, fetcher)
                if page is None:
                    continue
                save_page(page, output_dir)
                new_pages.append(page)
                if on_page_saved:
                    on_page_saved(page)

        save_index(
            (p.to_index_record() for p in new_pages),
            index_path,
            append=resume,
        )
        _log_summary(new_pages)
    else:
        log.info("No new URLs to fetch — all already processed.")

    return load_pages_from_disk(output_dir)


def _load_processed_urls(index_path: Path) -> set[str]:
    """
    Return the set of URLs already recorded in _index.jsonl.
    Returns an empty set when the file does not yet exist — so the first
    run and a resumed-with-no-prior-state run behave identically.
    """
    if not index_path.exists():
        return set()
    urls: set[str] = set()
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "url" in rec:
                    urls.add(rec["url"])
            except json.JSONDecodeError:
                pass
    return urls


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_summary(pages: list[DocPage]) -> dict:
    """
    Return a summary dict — useful in notebooks for quick inspection.

    Example
    -------
    >>> summary = build_summary(pages)
    >>> print(summary["total_pages"], summary["top_sections"])
    """
    sections: dict[str, int] = {}
    for p in pages:
        sections[p.section] = sections.get(p.section, 0) + 1

    total_chars = sum(p.char_count for p in pages)

    # Aggregate API symbol count across all pages
    total_symbols = sum(len(p.symbols) for p in pages)

    # Count pages that expose at least one API symbol (vs conceptual pages)
    api_pages = sum(1 for p in pages if p.symbols)

    return {
        "total_pages": len(pages),
        "total_chars": total_chars,
        "avg_chars": total_chars // max(len(pages), 1),
        "total_symbols": total_symbols,
        "api_pages": api_pages,
        "top_sections": dict(
            sorted(sections.items(), key=lambda x: -x[1])[:10]
        ),
    }


def _log_summary(pages: list[DocPage]) -> None:
    s = build_summary(pages)
    log.info(
        "Pipeline complete — %d pages (%d with API symbols), "
        "%s chars total, %s avg, %d symbols indexed",
        s["total_pages"],
        s["api_pages"],
        f"{s['total_chars']:,}",
        f"{s['avg_chars']:,}",
        s["total_symbols"],
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _infer_section(url: str) -> str:
    """Top-level section name from URL, e.g. 'nn' or 'torch'."""
    path = urlparse(url).path
    rel = re.sub(r"^/docs/stable/", "", path).strip("/")
    parts = rel.split("/")
    first = re.sub(r"\.[^.]+$", "", parts[0]) if parts else ""
    return first or "root"


def _url_to_rel_path(url: str) -> Path:
    """Convert a doc URL to a relative .md file path."""
    path = urlparse(url).path
    rel = re.sub(r"^/docs/stable/", "", path).strip("/")
    rel = re.sub(r"\.html?$", ".md", rel)
    return Path(rel or "index.md")


def load_pages_from_disk(output_dir: str | Path) -> list[DocPage]:
    """
    Reconstruct ``DocPage`` objects from a previously saved pipeline run.

    Reads the ``_index.jsonl`` manifest (written by ``save_index()``) and
    loads the corresponding ``.md`` files from disk to populate the
    ``markdown`` field.  All marker / keyword metadata is deserialized
    from the JSONL record without re-parsing any HTML.

    This is the canonical way to feed a saved pipeline run into the
    chunking + embedding notebooks without re-crawling the documentation.

    Parameters
    ----------
    output_dir:
        Root directory that was passed to ``run_pipeline()`` or
        ``save_page()`` — the directory that contains ``_index.jsonl``
        and all ``.md`` files.

    Returns
    -------
    list[DocPage]
        Pages in the order they appear in ``_index.jsonl``.
        Pages whose ``.md`` file is missing are logged and skipped.

    Raises
    ------
    FileNotFoundError
        When ``_index.jsonl`` does not exist in ``output_dir``.
    """
    import json
    from .extractor import PageMarkers, SectionMarker

    output_dir = Path(output_dir)
    index_path = output_dir / "_index.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found at {index_path}. "
            "Run run_pipeline() first to generate the data."
        )

    pages: list[DocPage] = []
    with index_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d in index: %s", lineno, exc)
                continue

            md_path = output_dir / rec["saved_path"]
            if not md_path.exists():
                log.warning("Markdown file missing, skipping: %s", md_path)
                continue

            markdown = md_path.read_text(encoding="utf-8")

            # Reconstruct the nested dataclasses from the serialised dict
            markers_dict = rec["markers"]
            sections = [
                SectionMarker(**s) for s in markers_dict["sections"]
            ]
            markers = PageMarkers(
                module=markers_dict["module"],
                symbols=markers_dict["symbols"],
                sections=sections,
                keywords=markers_dict["keywords"],
            )

            pages.append(
                DocPage(
                    url=rec["url"],
                    title=rec["title"],
                    section=rec["section"],
                    markdown=markdown,
                    markers=markers,
                    saved_path=rec["saved_path"],
                )
            )

    log.info("Loaded %d pages from %s", len(pages), output_dir)
    return pages
