"""
fetcher.py
----------
Thin HTTP layer: rate-limited, retry-aware GET requests.
All other modules receive plain `requests.Response` objects so they
remain independent of any HTTP transport detail.
"""

from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetcher class
# ---------------------------------------------------------------------------

class RateLimitedFetcher:
    """
    HTTP GET client with:
    - Configurable polite rate-limit (requests / second)
    - Exponential back-off retries
    - Shared session with custom User-Agent

    Parameters
    ----------
    requests_per_second:
        Maximum request throughput.  Keep ≤ 2 for public servers.
    max_retries:
        Number of attempts before raising RuntimeError.
    timeout:
        Per-request timeout in seconds.
    user_agent:
        Value sent in the User-Agent header.
    """

    def __init__(
        self,
        requests_per_second: float = 2.0,
        max_retries: int = 3,
        timeout: int = 20,
        user_agent: str = "RAG-research-bot/1.0 (personal project)",
    ) -> None:
        self.requests_per_second = requests_per_second
        self.max_retries = max_retries
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._last_request_at: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get(self, url: str) -> requests.Response:
        """
        Fetch *url*, respecting rate-limit and retrying on transient errors.

        Returns
        -------
        requests.Response
            A successful (2xx) response.

        Raises
        ------
        RuntimeError
            When all retry attempts are exhausted.
        """
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_slot()
            try:
                resp = self._session.get(url, timeout=self.timeout)
                self._last_request_at = time.monotonic()
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                log.warning(
                    "Attempt %d/%d failed for %s — %s",
                    attempt, self.max_retries, url, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)  # 2s, 4s, …

        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts")

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._session.close()

    # Context-manager support so notebook cells can use `with` blocks
    def __enter__(self) -> "RateLimitedFetcher":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _wait_for_slot(self) -> None:
        gap = 1.0 / self.requests_per_second
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < gap:
            time.sleep(gap - elapsed)
