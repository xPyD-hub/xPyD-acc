"""Tests for KV cache comparison module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from xpyd_acc.kvcache import KVCacheComparator, KVCacheLoader


@pytest.fixture()
def tmp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _save_npz(path: Path, layers: dict[str, np.ndarray]) -> Path:
    """Save a dict of arrays as npz."""
    np.savez(str(path), **layers)
    return path


class TestKVCacheLoader:
    def test_load_valid(self, tmp_dir: Path) -> None:
        data = {"layer_0": np.ones((2, 4)), "layer_1": np.zeros((2, 4))}
        p = _save_npz(tmp_dir / "cache.npz", data)
        loaded = KVCacheLoader.load(p)
        assert set(loaded.keys()) == {"layer_0", "layer_1"}
        np.testing.assert_array_equal(loaded["layer_0"], data["layer_0"])

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            KVCacheLoader.load("/nonexistent/path.npz")


class TestKVCacheComparator:
    def test_identical_caches(self, tmp_dir: Path) -> None:
        data = {"layer_0": np.random.randn(4, 8), "layer_1": np.random.randn(4, 8)}
        comparator = KVCacheComparator()
        report = comparator.compare(data, data)
        assert report.match is True
        assert len(report.divergent_layers) == 0
        assert len(report.layers) == 2

    def test_divergent_caches(self) -> None:
        baseline = {"layer_0": np.zeros((4, 8))}
        target = {"layer_0": np.ones((4, 8))}
        comparator = KVCacheComparator()
        report = comparator.compare(baseline, target)
        assert report.match is False
        assert "layer_0" in report.divergent_layers
        assert report.layers[0].max_abs_diff == 1.0

    def test_within_threshold(self) -> None:
        baseline = {"layer_0": np.ones((4, 8))}
        target = {"layer_0": np.ones((4, 8)) + 1e-5}
        comparator = KVCacheComparator(max_abs_threshold=1e-3, cosine_threshold=0.999)
        report = comparator.compare(baseline, target)
        assert report.match is True

    def test_mismatched_layers(self) -> None:
        baseline = {"layer_0": np.ones((4, 8))}
        target = {"layer_1": np.ones((4, 8))}
        comparator = KVCacheComparator()
        report = comparator.compare(baseline, target)
        assert report.match is False
        assert len(report.divergent_layers) == 2  # both missing in the other

    def test_mismatched_shapes(self) -> None:
        baseline = {"layer_0": np.ones((4, 8))}
        target = {"layer_0": np.ones((4, 16))}
        comparator = KVCacheComparator()
        report = comparator.compare(baseline, target)
        assert report.match is False
        assert report.layers[0].divergent is True

    def test_empty_caches(self) -> None:
        comparator = KVCacheComparator()
        report = comparator.compare({}, {})
        assert report.match is True
        assert len(report.layers) == 0

    def test_zero_vectors(self) -> None:
        baseline = {"layer_0": np.zeros((4, 8))}
        target = {"layer_0": np.zeros((4, 8))}
        comparator = KVCacheComparator()
        report = comparator.compare(baseline, target)
        assert report.match is True
        assert report.layers[0].cosine_similarity == 1.0

    def test_format_report(self) -> None:
        baseline = {"layer_0": np.zeros((4, 8))}
        target = {"layer_0": np.ones((4, 8))}
        comparator = KVCacheComparator()
        report = comparator.compare(baseline, target, "base.npz", "target.npz")
        text = KVCacheComparator.format_report(report)
        assert "DIVERGENCE" in text
        assert "layer_0" in text

    def test_json_export(self) -> None:
        baseline = {"layer_0": np.ones((2, 2))}
        comparator = KVCacheComparator()
        report = comparator.compare(baseline, baseline)
        json_str = report.to_json()
        import json

        data = json.loads(json_str)
        assert data["match"] is True
        assert len(data["layers"]) == 1
