"""
Data acquisition package
"""

from .discovery import (
    fetch_url_list,
    SITEMAP_URL,
    DEFAULT_SKIP_PATTERNS,
)

from .fetcher import RateLimitedFetcher

from .cleaner import (
    extract_title,
    clean_html,
    extract_main_content,
)

from .converter import html_to_markdown

from .pipeline import (
    DocPage,
    process_page,
    save_page,
    save_index,
    run_pipeline,
    build_summary,
)

from .extractor import (
    PageMarkers,
    SectionMarker,
    extract_markers,
)

from . import cleaner, converter, discovery, fetcher, extractor, pipeline

__all__ = [
    "fetch_url_list",
    "SITEMAP_URL",
    "load_pages_from_disk",
    "DEFAULT_SKIP_PATTERNS",
    "RateLimitedFetcher",
    "extract_title",
    "clean_html",
    "extract_main_content",
    "html_to_markdown",
    "DocPage",
    "process_page",
    "save_page",
    "save_index",
    "run_pipeline",
    "build_summary",
    "cleaner",
    "converter",
    "discovery",
    "fetcher",
    "pipeline",
    "extractor",
    "PageMarkers",
    "SectionMarker",
    "extract_markers",
]
