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

from . import cleaner, converter, discovery, fetcher, pipeline
