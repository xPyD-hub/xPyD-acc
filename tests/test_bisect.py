"""Tests for bisect module — auto-bisect divergence by context length."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from xpyd_acc.bisect import BisectResult, BisectStep, run_bisect


def _make_check_divergence_side_effect(threshold: int):
    """Return a side effect that diverges when prefix_length >= threshold."""

    async def _side_effect(
        baseline_url, target_url, prompt, model, prefix_length, **kwargs
    ):
        if prefix_length >= threshold:
            return True, 5  # diverges at token index 5
        return False, None

    return _side_effect


def _always_diverge():
    async def _side_effect(
        baseline_url, target_url, prompt, model, prefix_length, **kwargs
    ):
        return True, 0

    return _side_effect


def _never_diverge():
    async def _side_effect(
        baseline_url, target_url, prompt, model, prefix_length, **kwargs
    ):
        return False, None

    return _side_effect


@pytest.mark.asyncio
@patch("xpyd_acc.bisect._check_divergence")
async def test_bisect_finds_threshold(mock_check):
    """Binary search finds the correct divergence threshold."""
    # Diverges at length >= 50
    mock_check.side_effect = _make_check_divergence_side_effect(50)

    prompt = "x" * 100
    result = await run_bisect(
        "http://baseline", "http://target", prompt, "model",
        min_length=1, max_length=100,
    )

    assert result.threshold_length == 50
    assert not result.always_diverges
    assert not result.never_diverges
    assert result.total_iterations > 0
    assert len(result.steps) > 0


@pytest.mark.asyncio
@patch("xpyd_acc.bisect._check_divergence")
async def test_bisect_never_diverges(mock_check):
    """No divergence at any length."""
    mock_check.side_effect = _never_diverge()

    prompt = "x" * 100
    result = await run_bisect(
        "http://baseline", "http://target", prompt, "model",
    )

    assert result.threshold_length is None
    assert result.never_diverges
    assert not result.always_diverges
    assert result.total_iterations == 1  # only checked max


@pytest.mark.asyncio
@patch("xpyd_acc.bisect._check_divergence")
async def test_bisect_always_diverges(mock_check):
    """Divergence at all lengths."""
    mock_check.side_effect = _always_diverge()

    prompt = "x" * 100
    result = await run_bisect(
        "http://baseline", "http://target", prompt, "model",
    )

    assert result.threshold_length == 1
    assert result.always_diverges
    assert not result.never_diverges
    assert result.total_iterations == 2  # checked max and min


@pytest.mark.asyncio
@patch("xpyd_acc.bisect._check_divergence")
async def test_bisect_json_export(mock_check):
    """BisectResult serializes to valid JSON."""
    mock_check.side_effect = _make_check_divergence_side_effect(30)

    prompt = "x" * 100
    result = await run_bisect(
        "http://baseline", "http://target", prompt, "model",
    )

    json_str = result.to_json()
    data = json.loads(json_str)
    assert data["threshold_length"] == 30
    assert "steps" in data
    assert "total_iterations" in data


@pytest.mark.asyncio
@patch("xpyd_acc.bisect._check_divergence")
async def test_bisect_with_min_max_bounds(mock_check):
    """Respects min_length and max_length bounds."""
    mock_check.side_effect = _make_check_divergence_side_effect(40)

    prompt = "x" * 200
    result = await run_bisect(
        "http://baseline", "http://target", prompt, "model",
        min_length=20, max_length=80,
    )

    assert result.threshold_length == 40
    # All tested lengths should be within bounds
    for step in result.steps:
        assert 20 <= step.prefix_length <= 80


@pytest.mark.asyncio
@patch("xpyd_acc.bisect._check_divergence")
async def test_bisect_progress_callback(mock_check):
    """Progress callback is called for each step."""
    mock_check.side_effect = _make_check_divergence_side_effect(50)

    callbacks = []

    def on_progress(iteration, length, diverges):
        callbacks.append((iteration, length, diverges))

    prompt = "x" * 100
    await run_bisect(
        "http://baseline", "http://target", prompt, "model",
        progress_callback=on_progress,
    )

    assert len(callbacks) > 0
    # First callback should be for max length (100)
    assert callbacks[0][1] == 100
    assert callbacks[0][2] is True  # diverges at 100 >= 50


@pytest.mark.asyncio
@patch("xpyd_acc.bisect._check_divergence")
async def test_bisect_invalid_range(mock_check):
    """Returns never_diverges with 0 iterations if min > max."""
    prompt = "x" * 10
    result = await run_bisect(
        "http://baseline", "http://target", prompt, "model",
        min_length=50, max_length=10,
    )

    assert result.never_diverges
    assert result.total_iterations == 0
    mock_check.assert_not_called()


@pytest.mark.asyncio
@patch("xpyd_acc.bisect._check_divergence")
async def test_bisect_sampling_params_forwarded(mock_check):
    """Sampling parameters are forwarded to _check_divergence."""
    from xpyd_acc.sampling import SamplingParams

    mock_check.side_effect = _never_diverge()

    sp = SamplingParams(temperature=0.0, seed=42)
    prompt = "x" * 10
    await run_bisect(
        "http://baseline", "http://target", prompt, "model",
        sampling=sp,
    )

    # Check sampling was passed through
    call_kwargs = mock_check.call_args
    assert call_kwargs.kwargs.get("sampling") is sp


@pytest.mark.asyncio
@patch("xpyd_acc.bisect._check_divergence")
async def test_bisect_step_dataclass(mock_check):
    """BisectStep contains correct data."""
    mock_check.side_effect = _make_check_divergence_side_effect(50)

    prompt = "x" * 100
    result = await run_bisect(
        "http://baseline", "http://target", prompt, "model",
    )

    for step in result.steps:
        assert isinstance(step, BisectStep)
        assert step.iteration > 0
        assert step.prefix_length > 0
        assert isinstance(step.diverges, bool)
        if step.diverges:
            assert step.first_divergence_index is not None


def test_bisect_result_dataclass():
    """BisectResult fields and to_json work."""
    result = BisectResult(
        threshold_length=42,
        total_iterations=5,
        steps=[BisectStep(1, 100, True, 3)],
    )
    assert result.threshold_length == 42
    data = json.loads(result.to_json())
    assert data["threshold_length"] == 42
    assert len(data["steps"]) == 1
