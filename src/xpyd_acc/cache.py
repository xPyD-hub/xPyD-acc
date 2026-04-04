"""Response caching for batch comparison.

Content-addressable cache keyed by hash(endpoint_url + model + prompt + sampling_params).
Entries stored as JSON files with TTL metadata.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xpyd_acc.log import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = ".xpyd-acc-cache"
DEFAULT_TTL = 3600  # 1 hour


def _cache_key(
    url: str,
    model: str,
    prompt: str,
    sampling_params: dict[str, Any] | None = None,
) -> str:
    """Generate a deterministic cache key from request parameters."""
    parts = {
        "url": url,
        "model": model,
        "prompt": prompt,
        "sampling": sampling_params or {},
    }
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class CacheEntry:
    """A cached API response with metadata."""

    text: str
    logprobs: list[dict[str, Any]]
    request_id: str
    timestamp: float
    url: str
    model: str
    prompt_hash: str

    def is_expired(self, ttl: float) -> bool:
        return (time.time() - self.timestamp) > ttl

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "logprobs": self.logprobs,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "url": self.url,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        return cls(
            text=data["text"],
            logprobs=data["logprobs"],
            request_id=data["request_id"],
            timestamp=data["timestamp"],
            url=data["url"],
            model=data["model"],
            prompt_hash=data.get("prompt_hash", ""),
        )


@dataclass
class CacheStats:
    """Statistics about cache usage."""

    entry_count: int = 0
    total_size_bytes: int = 0
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class ResponseCache:
    """Content-addressable response cache backed by filesystem."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR, ttl: float = DEFAULT_TTL):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self._hits = 0
        self._misses = 0

    def _entry_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(
        self,
        url: str,
        model: str,
        prompt: str,
        sampling_params: dict[str, Any] | None = None,
    ) -> CacheEntry | None:
        """Look up a cached response. Returns None on miss or expired entry."""
        key = _cache_key(url, model, prompt, sampling_params)
        path = self._entry_path(key)
        if not path.exists():
            self._misses += 1
            logger.debug("Cache miss: %s", key[:12])
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entry = CacheEntry.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            self._misses += 1
            logger.debug("Cache corrupt, treating as miss: %s", key[:12])
            path.unlink(missing_ok=True)
            return None
        if entry.is_expired(self.ttl):
            self._misses += 1
            logger.debug("Cache expired: %s", key[:12])
            path.unlink(missing_ok=True)
            return None
        self._hits += 1
        logger.info("Cache hit: %s", key[:12])
        return entry

    def put(
        self,
        url: str,
        model: str,
        prompt: str,
        text: str,
        logprobs: list[dict[str, Any]],
        request_id: str,
        sampling_params: dict[str, Any] | None = None,
    ) -> None:
        """Store a response in the cache."""
        key = _cache_key(url, model, prompt, sampling_params)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        entry = CacheEntry(
            text=text,
            logprobs=logprobs,
            request_id=request_id,
            timestamp=time.time(),
            url=url,
            model=model,
            prompt_hash=key,
        )
        path = self._entry_path(key)
        path.write_text(json.dumps(entry.to_dict(), ensure_ascii=False), encoding="utf-8")
        logger.debug("Cache store: %s", key[:12])

    def clear(self) -> int:
        """Remove all cache entries. Returns number of entries removed."""
        if not self.cache_dir.exists():
            return 0
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        s = CacheStats(hits=self._hits, misses=self._misses)
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.json"):
                s.entry_count += 1
                s.total_size_bytes += f.stat().st_size
        return s
