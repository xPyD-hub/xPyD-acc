"""Tests for multi-dataset CLI integration (M80)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from xpyd_acc.batch_compare import BatchReport, SampleResult
from xpyd_acc.multi_dataset import MultiDatasetReport, format_multi_dataset_report


def _make_report(total: int, divergent: int) -> BatchReport:
    """Create a minimal BatchReport for testing."""
    results = []
    for i in range(total):
        results.append(
            SampleResult(
                sample_id=f"s{i}",
                prompt=f"prompt {i}",
                baseline_output=f"output {i}",
                target_output=f"output {i}" if i >= divergent else f"diff {i}",
                exact_match=i >= divergent,
                first_divergence_index=0 if i < divergent else -1,
                logprob_gap=0.5 if i < divergent else 0.0,
                classification="likely_bug" if i < divergent else "match",
                baseline_logprob_at_divergence=-1.0 if i < divergent else 0.0,
                target_logprob_at_divergence=-1.5 if i < divergent else 0.0,
                context_length=10,
            )
        )
    return BatchReport(
        total_samples=total,
        match_samples=total - divergent,
        divergent_samples=divergent,
        divergence_rate=divergent / total if total else 0.0,
        results=results,
    )


class TestMultiDatasetReport:
    """Tests for MultiDatasetReport dataclass."""

    def test_aggregate_stats(self) -> None:
        r1 = _make_report(10, 2)
        r2 = _make_report(5, 1)
        mdr = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=["d1", "d2"],
            per_dataset={"d1": r1, "d2": r2},
        )
        assert mdr.total_samples == 15
        assert mdr.total_divergent == 3
        assert abs(mdr.overall_divergence_rate - 0.2) < 1e-9

    def test_empty_datasets(self) -> None:
        mdr = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=[],
            per_dataset={},
        )
        assert mdr.total_samples == 0
        assert mdr.overall_divergence_rate == 0.0

    def test_to_json_roundtrip(self) -> None:
        r1 = _make_report(3, 1)
        mdr = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=["ds1"],
            per_dataset={"ds1": r1},
        )
        data = json.loads(mdr.to_json())
        assert data["total_samples"] == 3
        assert data["total_divergent"] == 1
        assert "ds1" in data["per_dataset"]

    def test_to_markdown(self) -> None:
        r1 = _make_report(4, 2)
        mdr = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=["alpha"],
            per_dataset={"alpha": r1},
        )
        md = mdr.to_markdown()
        assert "Multi-Dataset" in md
        assert "alpha" in md
        assert "50.0%" in md

    def test_format_terminal(self) -> None:
        r1 = _make_report(10, 0)
        r2 = _make_report(10, 3)
        mdr = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=["good", "bad"],
            per_dataset={"good": r1, "bad": r2},
        )
        out = format_multi_dataset_report(mdr)
        assert "good" in out
        assert "bad" in out


class TestMultiDatasetCLIIntegration:
    """Tests for CLI batch.py multi-dataset path."""

    def test_dataset_flag_is_repeatable(self) -> None:
        """Verify argparse accepts multiple --dataset flags."""
        from xpyd_acc.cli.parsers import register_all

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        register_all(sub)
        args = parser.parse_args([
            "batch-compare",
            "--baseline", "http://base",
            "--target", "http://target",
            "--dataset", "a.jsonl",
            "--dataset", "b.jsonl",
            "--model", "m",
        ])
        assert args.dataset == ["a.jsonl", "b.jsonl"]

    def test_single_dataset_still_works(self) -> None:
        """Single --dataset should still produce a list with one element."""
        from xpyd_acc.cli.parsers import register_all

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        register_all(sub)
        args = parser.parse_args([
            "batch-compare",
            "--baseline", "http://base",
            "--target", "http://target",
            "--dataset", "only.jsonl",
            "--model", "m",
        ])
        assert args.dataset == ["only.jsonl"]

    @pytest.mark.asyncio
    async def test_multi_dataset_delegates(self, tmp_path: Path) -> None:
        """When 2+ datasets are given, _run_multi_dataset is called."""
        d1 = tmp_path / "d1.jsonl"
        d2 = tmp_path / "d2.jsonl"
        d1.write_text('{"prompt": "hello", "id": "s1"}\n')
        d2.write_text('{"prompt": "world", "id": "s2"}\n')

        with patch(
            "xpyd_acc.cli.batch._run_multi_dataset", new_callable=AsyncMock,
        ) as mock_fn:
            mock_fn.return_value = None
            from xpyd_acc.cli.batch import _run_batch_compare

            args = argparse.Namespace(
                dataset=[str(d1), str(d2)],
                baseline="http://base",
                target=["http://target"],
                snapshot=None,
                rerun=None,
                dry_run=False,
                _config=None,
            )
            await _run_batch_compare(args)
            mock_fn.assert_called_once()

    def test_json_export_multi_dataset(self, tmp_path: Path) -> None:
        """JSON export includes per-dataset breakdown."""
        r1 = _make_report(5, 1)
        r2 = _make_report(5, 2)
        mdr = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=["gsm8k", "mmlu"],
            per_dataset={"gsm8k": r1, "mmlu": r2},
        )
        out = tmp_path / "out.json"
        out.write_text(mdr.to_json())
        data = json.loads(out.read_text())
        assert "gsm8k" in data["per_dataset"]
        assert "mmlu" in data["per_dataset"]
        assert data["per_dataset"]["gsm8k"]["total_samples"] == 5

    def test_markdown_export_multi_dataset(self) -> None:
        """Markdown includes per-dataset sections."""
        r1 = _make_report(3, 1)
        mdr = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=["test_ds"],
            per_dataset={"test_ds": r1},
        )
        md = mdr.to_markdown()
        assert "## Dataset: test_ds" in md
        assert "Per-Dataset Summary" in md
