"""Tests for model fingerprinting (M55)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xpyd_acc.fingerprint import (
    DEFAULT_PROBES,
    Fingerprint,
    ProbeResult,
    _compute_hash,
    collect_fingerprint,
    compare_fingerprints,
)

# ---------------------------------------------------------------------------
# Unit tests for _compute_hash
# ---------------------------------------------------------------------------

def test_compute_hash_deterministic():
    probes = [ProbeResult(prompt="a", output="hello")]
    assert _compute_hash(probes) == _compute_hash(probes)


def test_compute_hash_different_outputs():
    a = [ProbeResult(prompt="a", output="hello")]
    b = [ProbeResult(prompt="a", output="world")]
    assert _compute_hash(a) != _compute_hash(b)


def test_compute_hash_length():
    probes = [ProbeResult(prompt="a", output="x")]
    assert len(_compute_hash(probes)) == 16


# ---------------------------------------------------------------------------
# Fingerprint dataclass tests
# ---------------------------------------------------------------------------

def _make_fp(endpoint: str, outputs: list[str]) -> Fingerprint:
    probes = [ProbeResult(prompt=f"p{i}", output=o) for i, o in enumerate(outputs)]
    return Fingerprint(
        endpoint=endpoint,
        model="test-model",
        hash=_compute_hash(probes),
        probes=probes,
        probe_count=len(probes),
    )


def test_fingerprint_matches_same():
    fp = _make_fp("http://a", ["x", "y"])
    fp2 = _make_fp("http://b", ["x", "y"])
    assert fp.matches(fp2)


def test_fingerprint_no_match():
    fp = _make_fp("http://a", ["x", "y"])
    fp2 = _make_fp("http://b", ["x", "z"])
    assert not fp.matches(fp2)


def test_fingerprint_diff_empty():
    fp = _make_fp("http://a", ["x", "y"])
    fp2 = _make_fp("http://b", ["x", "y"])
    assert fp.diff(fp2) == []


def test_fingerprint_diff_reports_changes():
    fp = _make_fp("http://a", ["x", "y", "z"])
    fp2 = _make_fp("http://b", ["x", "CHANGED", "z"])
    diffs = fp.diff(fp2)
    assert len(diffs) == 1
    assert diffs[0]["probe_index"] == 1
    assert diffs[0]["output_a"] == "y"
    assert diffs[0]["output_b"] == "CHANGED"


def test_fingerprint_to_dict():
    fp = _make_fp("http://a", ["x"])
    d = fp.to_dict()
    assert d["endpoint"] == "http://a"
    assert d["hash"] == fp.hash
    assert len(d["probes"]) == 1
    assert d["probes"][0]["output"] == "x"


# ---------------------------------------------------------------------------
# FingerprintComparison tests
# ---------------------------------------------------------------------------

def test_compare_fingerprints_match():
    fp1 = _make_fp("http://a", ["x", "y"])
    fp2 = _make_fp("http://b", ["x", "y"])
    cmp = compare_fingerprints(fp1, fp2)
    assert cmp.match is True
    assert cmp.differing_probes == []
    assert cmp.total_probes == 2


def test_compare_fingerprints_mismatch():
    fp1 = _make_fp("http://a", ["x", "y"])
    fp2 = _make_fp("http://b", ["x", "z"])
    cmp = compare_fingerprints(fp1, fp2)
    assert cmp.match is False
    assert len(cmp.differing_probes) == 1


def test_comparison_to_dict():
    fp1 = _make_fp("http://a", ["x"])
    fp2 = _make_fp("http://b", ["y"])
    cmp = compare_fingerprints(fp1, fp2)
    d = cmp.to_dict()
    assert "match" in d
    assert d["match"] is False
    assert d["total_probes"] == 1


# ---------------------------------------------------------------------------
# collect_fingerprint (mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collect_fingerprint_basic():
    """collect_fingerprint calls LogprobsCollector for each probe."""
    mock_token = MagicMock()
    mock_token.token = "4"
    mock_result = MagicMock()
    mock_result.tokens = [mock_token]
    mock_result.model = "test-model"

    with patch("xpyd_acc.logprobs.LogprobsCollector") as MockCollector:
        instance = MockCollector.return_value
        instance.collect = AsyncMock(return_value=mock_result)

        fp = await collect_fingerprint("http://test:8000", model="test-model")

    assert fp.endpoint == "http://test:8000"
    assert fp.probe_count == len(DEFAULT_PROBES)
    assert instance.collect.call_count == len(DEFAULT_PROBES)
    assert len(fp.hash) == 16


@pytest.mark.asyncio
async def test_collect_fingerprint_custom_probes():
    """Custom probes override the default set."""
    mock_token = MagicMock()
    mock_token.token = "ok"
    mock_result = MagicMock()
    mock_result.tokens = [mock_token]
    mock_result.model = "m"

    with patch("xpyd_acc.logprobs.LogprobsCollector") as MockCollector:
        instance = MockCollector.return_value
        instance.collect = AsyncMock(return_value=mock_result)

        fp = await collect_fingerprint("http://x", probes=["hello", "world"])

    assert fp.probe_count == 2
    assert instance.collect.call_count == 2


@pytest.mark.asyncio
async def test_collect_fingerprint_uses_temperature_zero():
    """Fingerprint collection uses temperature=0, seed=42."""
    mock_token = MagicMock()
    mock_token.token = "t"
    mock_result = MagicMock()
    mock_result.tokens = [mock_token]
    mock_result.model = "m"

    with patch("xpyd_acc.logprobs.LogprobsCollector") as MockCollector:
        instance = MockCollector.return_value
        instance.collect = AsyncMock(return_value=mock_result)

        await collect_fingerprint("http://x", probes=["hi"])

    call_kwargs = instance.collect.call_args
    sp = call_kwargs.kwargs.get("sampling_params") or call_kwargs[1].get("sampling_params")
    assert sp is not None
    assert sp.temperature == 0
    assert sp.seed == 42


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_probes_hash():
    assert _compute_hash([]) == _compute_hash([])


def test_fingerprint_diff_different_lengths():
    """diff handles mismatched probe counts (zips to shorter)."""
    fp1 = _make_fp("http://a", ["x", "y", "z"])
    fp2 = _make_fp("http://b", ["x", "y"])
    diffs = fp1.diff(fp2)
    assert diffs == []  # first two match, third ignored by zip


def test_fingerprint_json_roundtrip():
    fp = _make_fp("http://a", ["hello", "world"])
    d = fp.to_dict()
    s = json.dumps(d)
    loaded = json.loads(s)
    assert loaded["hash"] == fp.hash
    assert len(loaded["probes"]) == 2
