"""Tests for inference_hooks module."""

from __future__ import annotations

import json

import numpy as np

from xpyd_acc.inference_hooks import (
    HookCapture,
    HookPoint,
    InferenceHook,
    MockInferenceHook,
    StageComparison,
    TraceResult,
    _cosine_sim,
    compare_captures,
    format_trace,
    run_trace,
)


class TestHookPoint:
    def test_enum_values(self):
        assert HookPoint.PREFILL.value == "prefill"
        assert HookPoint.KV_TRANSFER.value == "kv_transfer"
        assert HookPoint.DECODE_STEP.value == "decode_step"


class TestHookCapture:
    def test_to_dict_with_arrays(self):
        cap = HookCapture(
            hook_point=HookPoint.PREFILL,
            layer=0,
            hidden_states=np.zeros((4, 8), dtype=np.float32),
        )
        d = cap.to_dict()
        assert d["hook_point"] == "prefill"
        assert d["layer"] == 0
        assert d["hidden_states"]["shape"] == [4, 8]
        assert d["hidden_states"]["dtype"] == "float32"
        assert d["logits"] is None

    def test_to_dict_no_arrays(self):
        cap = HookCapture(hook_point=HookPoint.KV_TRANSFER, layer=2)
        d = cap.to_dict()
        assert d["hidden_states"] is None
        assert d["kv_cache"] is None

    def test_step_field(self):
        cap = HookCapture(hook_point=HookPoint.DECODE_STEP, layer=1, step=3)
        assert cap.step == 3
        assert cap.to_dict()["step"] == 3


class TestStageComparison:
    def test_to_dict(self):
        sc = StageComparison(
            hook_point=HookPoint.PREFILL,
            layer=0,
            max_abs_diff=1e-6,
            mean_abs_diff=5e-7,
            cosine_similarity=0.9999,
            field_name="hidden_states",
            diverged=False,
        )
        d = sc.to_dict()
        assert d["hook_point"] == "prefill"
        assert d["diverged"] is False

    def test_diverged_flag(self):
        sc = StageComparison(
            hook_point=HookPoint.PREFILL, layer=0, diverged=True
        )
        assert sc.diverged is True


class TestTraceResult:
    def test_to_dict_empty(self):
        tr = TraceResult(
            prompt="test",
            baseline_url="http://a",
            target_url="http://b",
            hooks=[HookPoint.PREFILL],
        )
        d = tr.to_dict()
        assert d["prompt"] == "test"
        assert d["hooks"] == ["prefill"]
        assert d["comparisons"] == []
        assert d["first_divergence"] is None
        assert d["overall_diverged"] is False

    def test_to_dict_with_comparisons(self):
        sc = StageComparison(
            hook_point=HookPoint.PREFILL, layer=0, diverged=True
        )
        tr = TraceResult(
            prompt="test",
            baseline_url="",
            target_url="",
            hooks=[HookPoint.PREFILL],
            comparisons=[sc],
            first_divergence=sc,
            overall_diverged=True,
        )
        d = tr.to_dict()
        assert d["overall_diverged"] is True
        assert d["first_divergence"]["diverged"] is True
        assert len(d["comparisons"]) == 1


class TestMockInferenceHook:
    def test_implements_protocol(self):
        hook = MockInferenceHook()
        assert isinstance(hook, InferenceHook)

    def test_prefill_returns_capture(self):
        hook = MockInferenceHook(num_layers=4, hidden_dim=16, seq_len=4)
        cap = hook.on_prefill(0)
        assert cap is not None
        assert cap.hook_point == HookPoint.PREFILL
        assert cap.layer == 0
        assert cap.hidden_states is not None
        assert cap.hidden_states.shape == (4, 16)
        assert cap.kv_cache is not None

    def test_prefill_out_of_range(self):
        hook = MockInferenceHook(num_layers=2)
        assert hook.on_prefill(5) is None

    def test_kv_transfer(self):
        hook = MockInferenceHook(num_layers=4, hidden_dim=16, seq_len=4)
        cap = hook.on_kv_transfer(1)
        assert cap is not None
        assert cap.hook_point == HookPoint.KV_TRANSFER
        assert cap.kv_cache is not None

    def test_decode_step(self):
        hook = MockInferenceHook(num_layers=4, hidden_dim=16, seq_len=4)
        cap = hook.on_decode_step(0, 0)
        assert cap is not None
        assert cap.hook_point == HookPoint.DECODE_STEP
        assert cap.step == 0
        assert cap.logits is not None

    def test_noise_scale_zero_is_deterministic(self):
        h1 = MockInferenceHook(seed=42, noise_scale=0.0)
        h2 = MockInferenceHook(seed=42, noise_scale=0.0)
        c1 = h1.on_prefill(0)
        c2 = h2.on_prefill(0)
        np.testing.assert_array_equal(c1.hidden_states, c2.hidden_states)

    def test_noise_scale_adds_variation(self):
        h1 = MockInferenceHook(seed=42, noise_scale=0.0)
        h2 = MockInferenceHook(seed=42, noise_scale=0.1)
        c1 = h1.on_prefill(0)
        c2 = h2.on_prefill(0)
        # They share the same base but noise differs
        assert not np.array_equal(c1.hidden_states, c2.hidden_states)


class TestCosineSim:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_sim(a, a) - 1.0) < 1e-10

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_sim(a, b)) < 1e-10

    def test_zero_vector(self):
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        assert _cosine_sim(a, b) == 0.0


class TestCompareCaptures:
    def test_identical_captures(self):
        hs = np.ones((4, 8), dtype=np.float32)
        c1 = HookCapture(hook_point=HookPoint.PREFILL, layer=0, hidden_states=hs)
        c2 = HookCapture(hook_point=HookPoint.PREFILL, layer=0, hidden_states=hs.copy())
        results = compare_captures(c1, c2)
        assert len(results) == 1
        assert not results[0].diverged
        assert results[0].max_abs_diff == 0.0

    def test_divergent_captures(self):
        hs1 = np.zeros((4, 8), dtype=np.float32)
        hs2 = np.ones((4, 8), dtype=np.float32)
        c1 = HookCapture(hook_point=HookPoint.PREFILL, layer=0, hidden_states=hs1)
        c2 = HookCapture(hook_point=HookPoint.PREFILL, layer=0, hidden_states=hs2)
        results = compare_captures(c1, c2, threshold=0.5)
        assert len(results) == 1
        assert results[0].diverged
        assert results[0].max_abs_diff == 1.0

    def test_shape_mismatch(self):
        c1 = HookCapture(
            hook_point=HookPoint.PREFILL, layer=0,
            hidden_states=np.zeros((4, 8), dtype=np.float32),
        )
        c2 = HookCapture(
            hook_point=HookPoint.PREFILL, layer=0,
            hidden_states=np.zeros((4, 16), dtype=np.float32),
        )
        results = compare_captures(c1, c2)
        assert len(results) == 1
        assert results[0].diverged
        assert results[0].max_abs_diff == float("inf")

    def test_multiple_fields(self):
        hs = np.ones((4, 8), dtype=np.float32)
        kv = np.ones((2, 4, 8), dtype=np.float32)
        c1 = HookCapture(hook_point=HookPoint.PREFILL, layer=0, hidden_states=hs, kv_cache=kv)
        c2 = HookCapture(
            hook_point=HookPoint.PREFILL, layer=0,
            hidden_states=hs.copy(), kv_cache=kv.copy(),
        )
        results = compare_captures(c1, c2)
        assert len(results) == 2  # hidden_states + kv_cache

    def test_none_fields_skipped(self):
        c1 = HookCapture(hook_point=HookPoint.PREFILL, layer=0)
        c2 = HookCapture(hook_point=HookPoint.PREFILL, layer=0)
        results = compare_captures(c1, c2)
        assert len(results) == 0


class TestRunTrace:
    def test_no_divergence(self):
        h1 = MockInferenceHook(seed=42, noise_scale=0.0)
        h2 = MockInferenceHook(seed=42, noise_scale=0.0)
        result = run_trace(h1, h2, prompt="test", num_layers=2, decode_steps=1)
        assert not result.overall_diverged
        assert result.first_divergence is None
        assert len(result.comparisons) > 0

    def test_with_divergence(self):
        h1 = MockInferenceHook(seed=42, noise_scale=0.0)
        h2 = MockInferenceHook(seed=42, noise_scale=1.0)
        result = run_trace(h1, h2, prompt="test", num_layers=2, threshold=0.01)
        assert result.overall_diverged
        assert result.first_divergence is not None

    def test_selective_hooks(self):
        h1 = MockInferenceHook(seed=42, noise_scale=0.0)
        h2 = MockInferenceHook(seed=42, noise_scale=0.0)
        result = run_trace(
            h1, h2, prompt="test",
            hooks=[HookPoint.PREFILL],
            num_layers=2,
        )
        # Only prefill comparisons
        for c in result.comparisons:
            assert c.hook_point == HookPoint.PREFILL

    def test_captures_stored(self):
        h1 = MockInferenceHook(seed=42, num_layers=2)
        h2 = MockInferenceHook(seed=42, num_layers=2)
        result = run_trace(h1, h2, prompt="test", num_layers=2, decode_steps=1)
        assert len(result.baseline_captures) > 0
        assert len(result.target_captures) > 0

    def test_urls_stored(self):
        h1 = MockInferenceHook()
        h2 = MockInferenceHook()
        result = run_trace(
            h1, h2, prompt="test",
            baseline_url="http://base", target_url="http://target",
        )
        assert result.baseline_url == "http://base"
        assert result.target_url == "http://target"


class TestFormatTrace:
    def test_format_match(self):
        h1 = MockInferenceHook(seed=42, noise_scale=0.0)
        h2 = MockInferenceHook(seed=42, noise_scale=0.0)
        result = run_trace(h1, h2, prompt="Hello world", num_layers=2)
        output = format_trace(result)
        assert "MATCH" in output
        assert "Hello world" in output
        assert "Stage Details:" in output

    def test_format_diverged(self):
        h1 = MockInferenceHook(seed=42, noise_scale=0.0)
        h2 = MockInferenceHook(seed=42, noise_scale=1.0)
        result = run_trace(h1, h2, prompt="test", num_layers=2, threshold=0.01)
        output = format_trace(result)
        assert "DIVERGED" in output
        assert "First divergence:" in output

    def test_format_with_errors(self):
        result = TraceResult(
            prompt="test",
            baseline_url="",
            target_url="",
            hooks=[],
            errors=["Connection failed"],
        )
        output = format_trace(result)
        assert "Connection failed" in output

    def test_format_long_prompt_truncated(self):
        long_prompt = "x" * 200
        result = TraceResult(
            prompt=long_prompt,
            baseline_url="",
            target_url="",
            hooks=[],
        )
        output = format_trace(result)
        assert "..." in output


class TestJsonExport:
    def test_trace_result_json_serializable(self):
        h1 = MockInferenceHook(seed=42, noise_scale=0.0, num_layers=2)
        h2 = MockInferenceHook(seed=42, noise_scale=0.1, num_layers=2)
        result = run_trace(h1, h2, prompt="test", num_layers=2)
        d = result.to_dict()
        # Should be JSON-serializable
        s = json.dumps(d)
        loaded = json.loads(s)
        assert loaded["prompt"] == "test"
        assert isinstance(loaded["comparisons"], list)


class TestCLIIntegration:
    def test_trace_parser_registered(self):
        """Verify trace subcommand is registered in CLI parsers."""
        import argparse

        from xpyd_acc.cli.parsers import register_all

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        register_all(sub)
        args = parser.parse_args([
            "trace", "--baseline", "http://a", "--target", "http://b",
            "--prompt", "test",
        ])
        assert args.command == "trace"
        assert args.baseline == "http://a"
