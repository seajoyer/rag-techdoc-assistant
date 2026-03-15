"""
discovery.py
------------
Discovers all PyTorch documentation page URLs from the official sitemap.
Keeps this concern isolated so it can be swapped for a local mirror or
a custom URL list without touching any other module.
"""

from __future__ import annotations

import logging
from typing import Generator

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (can be overridden at call-site)
# ---------------------------------------------------------------------------

SITEMAP_URL = "https://pytorch.org/docs/stable/sitemap.xml"

# URL substrings that indicate non-content pages (source viewer, indexes …)
DEFAULT_SKIP_PATTERNS: list[str] = [
    "_modules/",    # raw source-code viewer
    "genindex",     # alphabetical symbol index
    "py-modindex",  # module index
    "search",       # search page
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def iter_doc_urls(
    sitemap_url: str = SITEMAP_URL,
    skip_patterns: list[str] | None = None,
    session: requests.Session | None = None,
    timeout: int = 20,
) -> Generator[str, None, None]:
    """
    Yield every PyTorch documentation URL found in the sitemap,
    filtering out non-content pages.

    Parameters
    ----------
    sitemap_url:
        URL of the XML sitemap to parse.
    skip_patterns:
        URL substrings that mark pages to skip.
        Defaults to DEFAULT_SKIP_PATTERNS.
    session:
        Optional pre-configured requests.Session (useful for testing /
        injecting auth headers).
    timeout:
        HTTP request timeout in seconds.

    Yields
    ------
    str
        Absolute documentation page URLs, one at a time.
    """
    patterns = skip_patterns if skip_patterns is not None else DEFAULT_SKIP_PATTERNS
    sess = session or requests.Session()

    log.info("Fetching sitemap: %s", sitemap_url)
    resp = sess.get(sitemap_url, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "lxml-xml")
    total = skipped = 0

    for loc in soup.find_all("loc"):
        url = loc.text.strip()
        total += 1
        if _should_skip(url, patterns):
            skipped += 1
            continue
        yield url

    log.info("Sitemap: %d total URLs, %d skipped, %d yielded", total, skipped, total - skipped)


def fetch_url_list(
    sitemap_url: str = SITEMAP_URL,
    skip_patterns: list[str] | None = None,
    max_pages: int | None = None,
    session: requests.Session | None = None,
) -> list[str]:
    """
    Convenience wrapper that materialises iter_doc_urls into a list,
    optionally capped at max_pages. Better suited for notebook cells
    that want to inspect the list before processing.

    Parameters
    ----------
    max_pages:
        Maximum number of URLs to return.  None = return all.
    """
    urls = list(iter_doc_urls(sitemap_url=sitemap_url, skip_patterns=skip_patterns, session=session))
    if max_pages is not None:
        urls = urls[:max_pages]
    log.info("URL list ready: %d pages", len(urls))
    return urls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _should_skip(url: str, patterns: list[str]) -> bool:
    return any(pat in url for pat in patterns)
