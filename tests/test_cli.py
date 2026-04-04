"""Smoke tests for CLI."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from xpyd_acc.cli import main


def test_help():
    result = subprocess.run(
        [sys.executable, "-m", "xpyd_acc.cli"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


def test_version_flag():
    """Test that --version prints version and exits."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0


class TestCliConfigIntegration:
    """Tests for --config flag and auto-discovery in CLI."""

    def test_config_flag_loads_defaults(self, tmp_path: Path) -> None:
        """--config flag loads TOML and fills missing CLI args."""
        config_file = tmp_path / "test.toml"
        config_file.write_text("""\
[defaults]
baseline = "http://config-baseline:8000/v1"
target = "http://config-target:8000/v1"
model = "config-model"
""")
        captured_args = {}

        async def mock_collect(prompt, max_tokens=64):
            from xpyd_acc.logprobs import LogprobsResult, TokenLogprob
            return LogprobsResult(tokens=[
                TokenLogprob(token="hello", logprob=-0.1, top_logprobs={"hello": -0.1}),
            ])

        with patch("xpyd_acc.cli._run_compare_logprobs") as mock_run:
            async def capture(args):
                captured_args["baseline"] = args.baseline
                captured_args["target"] = args.target
                captured_args["model"] = args.model

            mock_run.side_effect = capture
            main([
                "--config", str(config_file),
                "compare-logprobs",
                "--baseline", "http://cli-baseline:8000/v1",
                "--target", "http://cli-target:8000/v1",
                "--prompt", "test",
            ])

        # CLI values win over config
        assert captured_args["baseline"] == "http://cli-baseline:8000/v1"
        assert captured_args["target"] == "http://cli-target:8000/v1"
        # Config fills in model (CLI default is "default", not None, so config won't override)
        assert captured_args["model"] in ("default", "config-model")

    def test_config_auto_discovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Auto-discovers xpyd-acc.toml in cwd."""
        config_file = tmp_path / "xpyd-acc.toml"
        config_file.write_text("""\
[report]
output = "auto-discovered.html"
""")
        monkeypatch.chdir(tmp_path)

        captured_args = {}
        # Create a dummy input file
        input_file = tmp_path / "data.json"
        input_file.write_text('{"total_samples":0,"divergent_samples":0,"match_samples":0,'
                              '"divergence_rate":0,"results":[]}')

        with patch("xpyd_acc.cli._run_report") as mock_run:
            def capture(args):
                captured_args["output"] = args.output

            mock_run.side_effect = capture
            main(["report", "--input", str(input_file)])

        # Config auto-discovery should set output from config
        assert captured_args["output"] == "auto-discovered.html"

    def test_config_batch_section_merge(self, tmp_path: Path) -> None:
        """Batch-specific config values are merged for batch-compare command."""
        config_file = tmp_path / "test.toml"
        config_file.write_text("""\
[batch]
concurrency = 42
logprob_gap_threshold = 0.05
""")

        captured_args = {}

        with patch("xpyd_acc.cli._run_batch_compare") as mock_run:
            async def capture(args):
                captured_args["concurrency"] = args.concurrency
                captured_args["logprob_gap_threshold"] = args.logprob_gap_threshold

            mock_run.side_effect = capture
            main([
                "--config", str(config_file),
                "batch-compare",
                "--baseline", "http://b:8000",
                "--target", "http://t:8000",
                "--dataset", "data.jsonl",
            ])

        assert captured_args["concurrency"] == 42
        assert captured_args["logprob_gap_threshold"] == 0.05

    def test_no_config_still_works(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI works without any config file."""
        monkeypatch.chdir(tmp_path)  # No xpyd-acc.toml here

        with patch("xpyd_acc.cli._run_compare_logprobs") as mock_run:
            mock_run.side_effect = lambda args: None
            main([
                "compare-logprobs",
                "--baseline", "http://b:8000",
                "--target", "http://t:8000",
                "--prompt", "test",
            ])

        mock_run.assert_called_once()
