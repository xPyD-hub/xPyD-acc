"""Tests for response caching."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from xpyd_acc.cache import (
    CacheEntry,
    CacheStats,
    ResponseCache,
    _cache_key,
)


def test_cache_key_deterministic():
    """Same inputs produce same key."""
    k1 = _cache_key("http://a", "m", "hello")
    k2 = _cache_key("http://a", "m", "hello")
    assert k1 == k2


def test_cache_key_varies_with_url():
    k1 = _cache_key("http://a", "m", "hello")
    k2 = _cache_key("http://b", "m", "hello")
    assert k1 != k2


def test_cache_key_varies_with_prompt():
    k1 = _cache_key("http://a", "m", "hello")
    k2 = _cache_key("http://a", "m", "world")
    assert k1 != k2


def test_cache_key_varies_with_sampling():
    k1 = _cache_key("http://a", "m", "hello", {"temperature": 0})
    k2 = _cache_key("http://a", "m", "hello", {"temperature": 1})
    assert k1 != k2


def test_cache_entry_not_expired():
    e = CacheEntry("txt", [], "rid", time.time(), "url", "m", "h")
    assert not e.is_expired(3600)


def test_cache_entry_expired():
    e = CacheEntry("txt", [], "rid", time.time() - 7200, "url", "m", "h")
    assert e.is_expired(3600)


def test_cache_entry_roundtrip():
    e = CacheEntry("txt", [{"token": "a", "logprob": -0.5}], "rid", 100.0, "url", "m", "h")
    d = e.to_dict()
    e2 = CacheEntry.from_dict(d)
    assert e2.text == e.text
    assert e2.logprobs == e.logprobs
    assert e2.request_id == e.request_id
    assert e2.timestamp == e.timestamp


def test_cache_miss(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=3600)
    result = cache.get("http://a", "m", "hello")
    assert result is None


def test_cache_put_and_hit(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=3600)
    cache.put("http://a", "m", "hello", "world", [{"token": "w"}], "r1")
    entry = cache.get("http://a", "m", "hello")
    assert entry is not None
    assert entry.text == "world"
    assert entry.logprobs == [{"token": "w"}]
    assert entry.request_id == "r1"


def test_cache_ttl_expiry(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=1)
    cache.put("http://a", "m", "hello", "world", [], "r1")
    # Manually expire by modifying the timestamp
    key = _cache_key("http://a", "m", "hello")
    path = cache._entry_path(key)
    data = json.loads(path.read_text())
    data["timestamp"] = time.time() - 100
    path.write_text(json.dumps(data))
    result = cache.get("http://a", "m", "hello")
    assert result is None


def test_cache_no_cache_bypass(tmp_path: Path):
    """Without a cache object, no caching happens (tested via miss counts)."""
    cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=3600)
    cache.get("http://a", "m", "hello")
    stats = cache.stats()
    assert stats.misses == 1
    assert stats.hits == 0


def test_cache_clear(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=3600)
    cache.put("http://a", "m", "p1", "t1", [], "r1")
    cache.put("http://a", "m", "p2", "t2", [], "r2")
    assert cache.stats().entry_count == 2
    count = cache.clear()
    assert count == 2
    assert cache.stats().entry_count == 0


def test_cache_clear_empty(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=3600)
    count = cache.clear()
    assert count == 0


def test_cache_stats(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=3600)
    cache.put("http://a", "m", "p1", "text", [], "r1")
    cache.get("http://a", "m", "p1")  # hit
    cache.get("http://a", "m", "p2")  # miss
    stats = cache.stats()
    assert stats.entry_count == 1
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.total_size_bytes > 0
    assert stats.hit_rate == pytest.approx(0.5)


def test_cache_stats_hit_rate_zero():
    stats = CacheStats()
    assert stats.hit_rate == 0.0


def test_cache_corrupt_entry(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=3600)
    cache.put("http://a", "m", "hello", "world", [], "r1")
    key = _cache_key("http://a", "m", "hello")
    path = cache._entry_path(key)
    path.write_text("not json")
    result = cache.get("http://a", "m", "hello")
    assert result is None
    assert not path.exists()  # corrupt entry cleaned up


def test_cache_with_sampling_params(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=3600)
    sp = {"temperature": 0, "seed": 42}
    cache.put("http://a", "m", "hello", "world", [], "r1", sp)
    # Same sampling params → hit
    entry = cache.get("http://a", "m", "hello", sp)
    assert entry is not None
    # Different sampling params → miss
    entry2 = cache.get("http://a", "m", "hello", {"temperature": 1})
    assert entry2 is None
