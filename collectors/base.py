"""
WC2026 Predictor — Base Collector
Shared HTTP session, rate limiting, retry logic, and caching.
All source-specific collectors inherit from this.
"""

import time
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from functools import wraps

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import HEADERS, RAW_DIR, RATE_LIMITS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)


def _make_session(retries: int = 6, backoff: float = 4.0) -> requests.Session:
    """Requests session with automatic retry on 429/500/502/503/504."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,   # honour Retry-After from the server
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session


class BaseCollector:
    """
    Base class for all data source collectors.

    Provides:
    - Shared requests.Session with retry
    - Rate limiting per source
    - Disk-based response cache (avoids re-fetching during dev)
    - Structured logging
    """

    source_name: str = "base"

    def __init__(self, use_cache: bool = True):
        self.session   = _make_session()
        self.use_cache = use_cache
        self.log       = logging.getLogger(self.source_name)
        self._last_req = 0.0
        self._cache_dir = RAW_DIR / self.source_name
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Rate limiting ──────────────────────────────────────────────────────

    def _wait(self):
        """Block until the minimum gap since last request has elapsed."""
        gap = RATE_LIMITS.get(self.source_name, 2.0)
        elapsed = time.time() - self._last_req
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_req = time.time()

    # ── Disk cache ─────────────────────────────────────────────────────────

    def _cache_path(self, url: str, params: dict = None) -> Path:
        key = url + json.dumps(params or {}, sort_keys=True)
        h   = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{h}.json"

    def _from_cache(self, path: Path):
        if self.use_cache and path.exists():
            with open(path) as f:
                self.log.debug(f"Cache hit: {path.name}")
                return json.load(f)
        return None

    def _to_cache(self, path: Path, data):
        with open(path, "w") as f:
            json.dump(data, f)

    # ── Core GET ───────────────────────────────────────────────────────────

    def get_json(self, url: str, params: dict = None, headers: dict = None) -> dict:
        """Fetch JSON with caching, rate-limiting, and retry."""
        cpath  = self._cache_path(url, params)
        cached = self._from_cache(cpath)
        if cached is not None:
            return cached

        # Manual retry loop as a last resort if the urllib3 retry budget is
        # exhausted but we still receive 429s.
        for attempt in range(1, 4):
            self._wait()
            self.log.info(f"GET {url} params={params}")
            resp = self.session.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 30 * attempt))
                self.log.warning(f"429 rate-limited — sleeping {wait}s (attempt {attempt}/3)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            self._to_cache(cpath, data)
            return data

        # Final attempt after manual back-off
        resp.raise_for_status()
        data = resp.json()
        self._to_cache(cpath, data)
        return data

    def get_text(self, url: str, params: dict = None) -> str:
        """Fetch raw text (for CSV endpoints or HTML scraping)."""
        cpath = self._cache_path(url, params)
        if self.use_cache and cpath.exists():
            return cpath.read_text()

        self._wait()
        self.log.info(f"GET {url}")
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        cpath.write_text(resp.text)
        return resp.text

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def today() -> str:
        return datetime.utcnow().strftime("%Y-%m-%d")

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().isoformat()