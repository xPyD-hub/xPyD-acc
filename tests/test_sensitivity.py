"""Tests for prompt sensitivity analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xpyd_acc.sensitivity import (
    PerturbationResult,
    SensitivityResult,
    _classify,
    format_sensitivity,
    generate_perturbations,
    run_sensitivity,
)


class TestGeneratePerturbations:
    """Tests for generate_perturbations()."""

    def test_returns_requested_count(self) -> None:
        result = generate_perturbations("Hello world", count=5)
        assert len(result) == 5

    def test_no_duplicates(self) -> None:
        result = generate_perturbations("Hello world", count=5)
        assert len(set(result)) == len(result)

    def test_original_not_included(self) -> None:
        prompt = "Hello world"
        result = generate_perturbations(prompt, count=5)
        assert prompt not in result

    def test_each_perturbation_differs_from_original(self) -> None:
        prompt = "Hello world"
        result = generate_perturbations(prompt, count=5)
        for p in result:
            assert p != prompt

    def test_zero_count(self) -> None:
        result = generate_perturbations("Hello", count=0)
        assert result == []

    def test_single_perturbation(self) -> None:
        result = generate_perturbations("Test", count=1)
        assert len(result) == 1

    def test_large_count_caps_at_available(self) -> None:
        result = generate_perturbations("Hi", count=20)
        # Should return up to available unique perturbations
        assert len(result) <= 20
        assert len(result) > 0


class TestClassify:
    """Tests for _classify()."""

    def test_systematic_all_diverge(self) -> None:
        results = [PerturbationResult("p", True, 5, 10) for _ in range(3)]
        assert _classify(True, results) == "systematic"

    def test_robust_none_diverge(self) -> None:
        results = [PerturbationResult("p", False, None, 10) for _ in range(3)]
        assert _classify(False, results) == "robust"

    def test_sensitive_mixed(self) -> None:
        results = [
            PerturbationResult("p1", True, 5, 10),
            PerturbationResult("p2", False, None, 10),
        ]
        assert _classify(True, results) == "sensitive"

    def test_sensitive_original_ok_some_diverge(self) -> None:
        results = [
            PerturbationResult("p1", True, 3, 10),
            PerturbationResult("p2", False, None, 10),
        ]
        assert _classify(False, results) == "sensitive"

    def test_empty_perturbations_systematic(self) -> None:
        assert _classify(True, []) == "systematic"

    def test_empty_perturbations_robust(self) -> None:
        assert _classify(False, []) == "robust"


class TestSensitivityResult:
    """Tests for SensitivityResult dataclass."""

    def test_divergence_rate_all_diverge(self) -> None:
        pr = [PerturbationResult("p", True, 5, 10) for _ in range(4)]
        sr = SensitivityResult("test", True, 4, 4, "systematic", pr)
        assert sr.divergence_rate == 1.0

    def test_divergence_rate_none_diverge(self) -> None:
        pr = [PerturbationResult("p", False, None, 10) for _ in range(4)]
        sr = SensitivityResult("test", False, 4, 0, "robust", pr)
        assert sr.divergence_rate == 0.0

    def test_divergence_rate_mixed(self) -> None:
        pr = [
            PerturbationResult("p1", True, 5, 10),
            PerturbationResult("p2", False, None, 10),
        ]
        sr = SensitivityResult("test", True, 2, 1, "sensitive", pr)
        # 2 diverge (original + 1) out of 3 total
        assert abs(sr.divergence_rate - 2 / 3) < 1e-9

    def test_to_dict_round_trip(self) -> None:
        pr = [PerturbationResult("p1", True, 5, 10)]
        sr = SensitivityResult("test prompt", True, 1, 1, "systematic", pr)
        d = sr.to_dict()
        assert d["original_prompt"] == "test prompt"
        assert d["classification"] == "systematic"
        assert len(d["perturbation_results"]) == 1
        assert d["perturbation_results"][0]["diverges"] is True

    def test_to_json(self) -> None:
        sr = SensitivityResult("test", False, 0, 0, "robust", [])
        j = json.loads(sr.to_json())
        assert j["classification"] == "robust"


class TestFormatSensitivity:
    """Tests for format_sensitivity()."""

    def test_systematic_output(self) -> None:
        pr = [PerturbationResult("p", True, 3, 10)]
        sr = SensitivityResult("test", True, 1, 1, "systematic", pr)
        text = format_sensitivity(sr)
        assert "SYSTEMATIC" in text
        assert "DIVERGE" in text

    def test_robust_output(self) -> None:
        pr = [PerturbationResult("p", False, None, 10)]
        sr = SensitivityResult("test", False, 1, 0, "robust", pr)
        text = format_sensitivity(sr)
        assert "ROBUST" in text
        assert "MATCH" in text

    def test_sensitive_output(self) -> None:
        pr = [
            PerturbationResult("p1", True, 5, 10),
            PerturbationResult("p2", False, None, 10),
        ]
        sr = SensitivityResult("test", True, 2, 1, "sensitive", pr)
        text = format_sensitivity(sr)
        assert "SENSITIVE" in text


@dataclass
class _FakeLogprobsResult:
    """Minimal stand-in for LogprobsResult."""

    first_divergence_index: int | None
    baseline_tokens: list[str]


class TestRunSensitivity:
    """Tests for run_sensitivity() with mocked collector."""

    @pytest.mark.asyncio
    async def test_systematic_result(self) -> None:
        fake_result = _FakeLogprobsResult(first_divergence_index=5, baseline_tokens=["a"] * 10)
        with patch("xpyd_acc.sensitivity.LogprobsCollector") as MockCollector:
            instance = MockCollector.return_value
            instance.collect = AsyncMock(return_value=fake_result)
            result = await run_sensitivity(
                "http://base", "http://target", "Hello",
                perturbation_count=3,
            )
        assert result.classification == "systematic"
        assert result.original_diverges is True
        assert result.divergent_count == 3

    @pytest.mark.asyncio
    async def test_robust_result(self) -> None:
        fake_result = _FakeLogprobsResult(first_divergence_index=None, baseline_tokens=["a"] * 10)
        with patch("xpyd_acc.sensitivity.LogprobsCollector") as MockCollector:
            instance = MockCollector.return_value
            instance.collect = AsyncMock(return_value=fake_result)
            result = await run_sensitivity(
                "http://base", "http://target", "Hello",
                perturbation_count=3,
            )
        assert result.classification == "robust"
        assert result.original_diverges is False
        assert result.divergent_count == 0

    @pytest.mark.asyncio
    async def test_sensitive_result(self) -> None:
        diverge = _FakeLogprobsResult(first_divergence_index=3, baseline_tokens=["a"] * 10)
        match = _FakeLogprobsResult(first_divergence_index=None, baseline_tokens=["a"] * 10)
        with patch("xpyd_acc.sensitivity.LogprobsCollector") as MockCollector:
            instance = MockCollector.return_value
            # Original diverges, first perturbation matches, rest diverge
            instance.collect = AsyncMock(side_effect=[diverge, match, diverge, diverge])
            result = await run_sensitivity(
                "http://base", "http://target", "Hello",
                perturbation_count=3,
            )
        assert result.classification == "sensitive"

    @pytest.mark.asyncio
    async def test_callback_called(self) -> None:
        fake_result = _FakeLogprobsResult(first_divergence_index=None, baseline_tokens=["a"] * 5)
        callback = MagicMock()
        with patch("xpyd_acc.sensitivity.LogprobsCollector") as MockCollector:
            instance = MockCollector.return_value
            instance.collect = AsyncMock(return_value=fake_result)
            await run_sensitivity(
                "http://base", "http://target", "Hello",
                perturbation_count=2, on_result=callback,
            )
        assert callback.call_count == 2

    @pytest.mark.asyncio
    async def test_json_export(self, tmp_path: Path) -> None:
        fake_result = _FakeLogprobsResult(first_divergence_index=None, baseline_tokens=["a"] * 5)
        with patch("xpyd_acc.sensitivity.LogprobsCollector") as MockCollector:
            instance = MockCollector.return_value
            instance.collect = AsyncMock(return_value=fake_result)
            result = await run_sensitivity(
                "http://base", "http://target", "Hello",
                perturbation_count=2,
            )
        out = tmp_path / "result.json"
        out.write_text(result.to_json())
        loaded = json.loads(out.read_text())
        assert loaded["classification"] == "robust"


class TestSensitivityCLI:
    """Tests for sensitivity CLI subcommand."""

    def test_cli_subcommand_exists(self) -> None:
        from xpyd_acc.cli import main
        with pytest.raises(SystemExit) as exc:
            main(["sensitivity", "--help"])
        assert exc.value.code == 0

    def test_cli_requires_baseline(self) -> None:
        from xpyd_acc.cli import main
        with pytest.raises(SystemExit) as exc:
            main(["sensitivity", "--target", "http://t", "--prompt", "hi"])
        assert exc.value.code != 0

    def test_cli_requires_target(self) -> None:
        from xpyd_acc.cli import main
        with pytest.raises(SystemExit) as exc:
            main(["sensitivity", "--baseline", "http://b", "--prompt", "hi"])
        assert exc.value.code != 0

    def test_cli_requires_prompt(self) -> None:
        from xpyd_acc.cli import main
        with pytest.raises(SystemExit) as exc:
            main(["sensitivity", "--baseline", "http://b", "--target", "http://t"])
        assert exc.value.code != 0
