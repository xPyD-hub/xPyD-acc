"""Tests for capture_kv module — KV cache export from vLLM."""

from __future__ import annotations

import json

import numpy as np
import pytest

from xpyd_acc.capture_kv import (
    CaptureConfig,
    CapturePoint,
    CaptureResult,
    LayerCapture,
    capture_kv_mock,
    filter_layers,
    load_capture,
    reconstruct_tp_shards,
    save_capture,
)


class TestCaptureConfig:
    def test_valid_config(self):
        cfg = CaptureConfig(url="http://localhost:8000", prompt="hello", output_path="/tmp/out.npz")
        assert cfg.validate() == []

    def test_empty_url(self):
        cfg = CaptureConfig(url="", prompt="hello", output_path="/tmp/out.npz")
        assert "url is required" in cfg.validate()

    def test_empty_prompt(self):
        cfg = CaptureConfig(url="http://x", prompt="", output_path="/tmp/out.npz")
        assert "prompt is required" in cfg.validate()

    def test_negative_layer(self):
        cfg = CaptureConfig(
            url="http://x", prompt="hi", output_path="/tmp/out.npz", layers=[-1, 0]
        )
        errors = cfg.validate()
        assert any("-1" in e for e in errors)

    def test_invalid_tp_size(self):
        cfg = CaptureConfig(
            url="http://x", prompt="hi", output_path="/tmp/out.npz", tp_size=0
        )
        assert any("tp_size" in e for e in cfg.validate())

    def test_no_capture_points(self):
        cfg = CaptureConfig(
            url="http://x", prompt="hi", output_path="/tmp/out.npz", capture_points=[]
        )
        assert any("capture_point" in e for e in cfg.validate())


class TestCaptureResult:
    def test_success_property(self):
        cfg = CaptureConfig(url="http://x", prompt="hi", output_path="/tmp/out.npz")
        lc = LayerCapture(
            layer_index=0,
            capture_point=CapturePoint.AFTER_PREFILL,
            key=np.zeros((2, 4, 8)),
            value=np.zeros((2, 4, 8)),
        )
        result = CaptureResult(config=cfg, layers=[lc])
        assert result.success is True

    def test_failure_with_errors(self):
        cfg = CaptureConfig(url="http://x", prompt="hi", output_path="/tmp/out.npz")
        result = CaptureResult(config=cfg, errors=["connection failed"])
        assert result.success is False

    def test_failure_no_layers(self):
        cfg = CaptureConfig(url="http://x", prompt="hi", output_path="/tmp/out.npz")
        result = CaptureResult(config=cfg)
        assert result.success is False

    def test_to_dict(self):
        cfg = CaptureConfig(url="http://x", prompt="hi", output_path="/tmp/out.npz")
        lc = LayerCapture(
            layer_index=3,
            capture_point=CapturePoint.AFTER_PREFILL,
            key=np.zeros((2, 4, 8)),
            value=np.zeros((2, 4, 8)),
        )
        result = CaptureResult(config=cfg, layers=[lc], metadata={"model_layers": 32})
        d = result.to_dict()
        assert d["success"] is True
        assert d["num_layers"] == 1
        assert 3 in d["layer_indices"]
        assert "after_prefill" in d["capture_points"]


class TestReconstructTPShards:
    def test_single_shard(self):
        arr = np.ones((4, 10, 8))
        out = reconstruct_tp_shards([arr])
        np.testing.assert_array_equal(out, arr)

    def test_two_shards(self):
        s1 = np.ones((4, 10, 8))
        s2 = np.ones((4, 10, 8)) * 2
        out = reconstruct_tp_shards([s1, s2], axis=0)
        assert out.shape == (8, 10, 8)
        np.testing.assert_array_equal(out[:4], s1)
        np.testing.assert_array_equal(out[4:], s2)

    def test_empty_shards_raises(self):
        with pytest.raises(ValueError, match="empty"):
            reconstruct_tp_shards([])

    def test_shape_mismatch_raises(self):
        s1 = np.ones((4, 10, 8))
        s2 = np.ones((4, 12, 8))  # different seq_len
        with pytest.raises(ValueError, match="mismatch"):
            reconstruct_tp_shards([s1, s2], axis=0)

    def test_dim_mismatch_raises(self):
        s1 = np.ones((4, 10, 8))
        s2 = np.ones((4, 10))  # different number of dims
        with pytest.raises(ValueError, match="dims"):
            reconstruct_tp_shards([s1, s2])


class TestSaveAndLoad:
    def test_round_trip(self, tmp_path):
        cfg = CaptureConfig(url="http://x", prompt="hi world", output_path=str(tmp_path / "out"))
        lc = LayerCapture(
            layer_index=0,
            capture_point=CapturePoint.AFTER_PREFILL,
            key=np.random.randn(2, 4, 8).astype(np.float16),
            value=np.random.randn(2, 4, 8).astype(np.float16),
        )
        result = CaptureResult(config=cfg, layers=[lc], metadata={"mode": "test"})

        saved_path = save_capture(result, tmp_path / "out")
        assert saved_path.exists()

        data = load_capture(saved_path)
        assert "layer_0_after_prefill_key" in data
        assert "layer_0_after_prefill_value" in data
        assert "metadata" in data
        np.testing.assert_array_equal(data["layer_0_after_prefill_key"], lc.key)

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_capture("/tmp/nonexistent_capture_file.npz")


class TestFilterLayers:
    def test_filter_none_returns_all(self):
        layers = [
            LayerCapture(0, CapturePoint.AFTER_PREFILL, np.zeros(1), np.zeros(1)),
            LayerCapture(1, CapturePoint.AFTER_PREFILL, np.zeros(1), np.zeros(1)),
        ]
        assert filter_layers(layers, None) == layers

    def test_filter_specific(self):
        layers = [
            LayerCapture(0, CapturePoint.AFTER_PREFILL, np.zeros(1), np.zeros(1)),
            LayerCapture(1, CapturePoint.AFTER_PREFILL, np.zeros(1), np.zeros(1)),
            LayerCapture(2, CapturePoint.AFTER_PREFILL, np.zeros(1), np.zeros(1)),
        ]
        filtered = filter_layers(layers, [0, 2])
        assert len(filtered) == 2
        assert filtered[0].layer_index == 0
        assert filtered[1].layer_index == 2


class TestMockCapture:
    def test_basic_mock(self):
        cfg = CaptureConfig(
            url="http://localhost:8000",
            prompt="hello world test",
            output_path="/tmp/mock.npz",
            layers=[0, 1],
        )
        result = capture_kv_mock(cfg)
        assert result.success
        assert len(result.layers) == 2  # 2 layers * 1 capture point
        assert result.metadata["mode"] == "mock"

    def test_mock_multiple_capture_points(self):
        cfg = CaptureConfig(
            url="http://localhost:8000",
            prompt="hello world",
            output_path="/tmp/mock.npz",
            layers=[0],
            capture_points=[CapturePoint.AFTER_PREFILL, CapturePoint.DURING_DECODE],
        )
        result = capture_kv_mock(cfg)
        assert result.success
        assert len(result.layers) == 2  # 1 layer * 2 capture points

    def test_mock_invalid_config(self):
        cfg = CaptureConfig(url="", prompt="hi", output_path="/tmp/mock.npz")
        result = capture_kv_mock(cfg)
        assert not result.success
        assert len(result.errors) > 0

    def test_mock_all_layers(self):
        cfg = CaptureConfig(
            url="http://localhost:8000",
            prompt="hello world",
            output_path="/tmp/mock.npz",
        )
        result = capture_kv_mock(cfg)
        assert result.success
        assert len(result.layers) == 32  # all 32 layers

    def test_mock_out_of_range_layer_skipped(self):
        cfg = CaptureConfig(
            url="http://localhost:8000",
            prompt="hello",
            output_path="/tmp/mock.npz",
            layers=[0, 999],
        )
        result = capture_kv_mock(cfg)
        assert result.success
        assert len(result.layers) == 1  # layer 999 skipped


class TestCLIIntegration:
    def test_capture_kv_mock_cli(self, tmp_path):
        """Test CLI capture-kv --mock end-to-end."""
        import subprocess

        out_path = str(tmp_path / "capture")
        result = subprocess.run(
            [
                "python3", "-m", "xpyd_acc.cli",
                "capture-kv",
                "--url", "http://localhost:8000",
                "--prompt", "hello world",
                "--output", out_path,
                "--layers", "0,1",
                "--mock",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "KV cache saved" in result.stdout

    def test_capture_kv_json_export(self, tmp_path):
        out_path = str(tmp_path / "capture")
        json_path = str(tmp_path / "meta.json")
        import subprocess

        result = subprocess.run(
            [
                "python3", "-m", "xpyd_acc.cli",
                "capture-kv",
                "--url", "http://localhost:8000",
                "--prompt", "test prompt",
                "--output", out_path,
                "--mock",
                "--json", json_path,
                "--layers", "0",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        with open(json_path) as f:
            meta = json.load(f)
        assert meta["success"] is True
