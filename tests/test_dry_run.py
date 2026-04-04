"""Tests for dry run mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_acc.dry_run import (
    DryRunResult,
    _estimate_tokens,
    format_dry_run,
    run_dry_run,
    validate_dataset,
)


def _make_dataset(tmp_path: Path, samples: list[dict]) -> Path:
    """Create a JSONL dataset file."""
    path = tmp_path / "test.jsonl"
    lines = [json.dumps(s) for s in samples]
    path.write_text("\n".join(lines))
    return path


class TestEstimateTokens:
    def test_basic(self):
        assert _estimate_tokens("hello world") >= 1

    def test_empty(self):
        assert _estimate_tokens("") == 1

    def test_long(self):
        text = "a" * 400
        assert _estimate_tokens(text) == 100


class TestValidateDataset:
    def test_missing_file(self):
        count, tokens, errors = validate_dataset("/nonexistent/file.jsonl")
        assert count == 0
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_valid_dataset(self, tmp_path):
        path = _make_dataset(tmp_path, [
            {"id": "1", "prompt": "What is 2+2?"},
            {"id": "2", "prompt": "What is 3+3?"},
        ])
        count, tokens, errors = validate_dataset(path)
        assert count == 2
        assert tokens > 0
        assert errors == []

    def test_empty_dataset(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        count, tokens, errors = validate_dataset(path)
        assert count == 0
        assert "empty" in errors[0].lower() or "empty" in str(errors).lower()


class TestDryRunResult:
    def test_to_json(self):
        result = DryRunResult(
            valid=True,
            sample_count=10,
            estimated_prompt_tokens=500,
            resolved_config={"model": "default"},
        )
        data = json.loads(result.to_json())
        assert data["valid"] is True
        assert data["sample_count"] == 10

    def test_valid_result(self):
        result = DryRunResult(
            valid=True,
            sample_count=5,
            estimated_prompt_tokens=200,
        )
        assert result.valid
        assert result.errors == []

    def test_invalid_result(self):
        result = DryRunResult(
            valid=False,
            sample_count=0,
            estimated_prompt_tokens=0,
            errors=["Dataset not found"],
        )
        assert not result.valid


class TestFormatDryRun:
    def test_pass(self):
        result = DryRunResult(
            valid=True,
            sample_count=10,
            estimated_prompt_tokens=500,
            resolved_config={"model": "test"},
            healthcheck_passed=True,
        )
        output = format_dry_run(result)
        assert "PASS" in output
        assert "10" in output

    def test_fail(self):
        result = DryRunResult(
            valid=False,
            sample_count=0,
            estimated_prompt_tokens=0,
            errors=["Dataset not found"],
        )
        output = format_dry_run(result)
        assert "FAIL" in output
        assert "Dataset not found" in output

    def test_warnings(self):
        result = DryRunResult(
            valid=True,
            sample_count=2000,
            estimated_prompt_tokens=50000,
            warnings=["Large dataset (2000 samples)"],
        )
        output = format_dry_run(result)
        assert "Large dataset" in output

    def test_skipped_healthcheck(self):
        result = DryRunResult(
            valid=True,
            sample_count=5,
            estimated_prompt_tokens=100,
            healthcheck_passed=None,
        )
        output = format_dry_run(result)
        assert "skipped" in output

    def test_template_shown(self):
        result = DryRunResult(
            valid=True,
            sample_count=5,
            estimated_prompt_tokens=100,
            template_name="gsm8k",
        )
        output = format_dry_run(result)
        assert "gsm8k" in output


@pytest.mark.asyncio
async def test_run_dry_run_valid(tmp_path):
    """Dry run with valid dataset, skipping healthcheck."""
    path = _make_dataset(tmp_path, [
        {"id": "1", "prompt": "Hello world"},
        {"id": "2", "prompt": "Goodbye world"},
    ])
    result = await run_dry_run(
        path,
        "http://localhost:8000",
        "http://localhost:8001",
        skip_healthcheck=True,
    )
    assert result.valid
    assert result.sample_count == 2
    assert result.estimated_prompt_tokens > 0
    assert result.healthcheck_passed is None
    assert result.errors == []


@pytest.mark.asyncio
async def test_run_dry_run_missing_dataset():
    """Dry run with missing dataset."""
    result = await run_dry_run(
        "/nonexistent/file.jsonl",
        "http://localhost:8000",
        "http://localhost:8001",
        skip_healthcheck=True,
    )
    assert not result.valid
    assert result.sample_count == 0
    assert any("not found" in e for e in result.errors)


@pytest.mark.asyncio
async def test_run_dry_run_resolved_config(tmp_path):
    """Resolved config reflects provided values."""
    path = _make_dataset(tmp_path, [{"id": "1", "prompt": "test"}])
    result = await run_dry_run(
        path,
        "http://base:8000",
        "http://target:8001",
        skip_healthcheck=True,
        model="gpt-4",
        max_tokens=128,
        concurrency=10,
    )
    assert result.resolved_config["model"] == "gpt-4"
    assert result.resolved_config["max_tokens"] == 128
    assert result.resolved_config["concurrency"] == 10


@pytest.mark.asyncio
async def test_run_dry_run_large_dataset_warning(tmp_path):
    """Large dataset triggers warning."""
    samples = [{"id": str(i), "prompt": f"sample {i}"} for i in range(1500)]
    path = _make_dataset(tmp_path, samples)
    result = await run_dry_run(
        path,
        "http://localhost:8000",
        "http://localhost:8001",
        skip_healthcheck=True,
    )
    assert result.valid
    assert any("Large dataset" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_run_dry_run_json_export(tmp_path):
    """Dry run result can be exported as JSON."""
    path = _make_dataset(tmp_path, [{"id": "1", "prompt": "test"}])
    result = await run_dry_run(
        path,
        "http://localhost:8000",
        "http://localhost:8001",
        skip_healthcheck=True,
    )
    data = json.loads(result.to_json())
    assert data["valid"] is True
    assert data["sample_count"] == 1
    assert "resolved_config" in data
