"""Tests for TOML configuration file support."""

from __future__ import annotations

from pathlib import Path

import pytest

from xpyd_acc.config import (
    AppConfig,
    BatchConfig,
    DefaultsConfig,
    KVConfig,
    ReportConfig,
    discover_config,
    load_config,
    merge_cli_args,
)


class TestLoadConfig:
    def test_full_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text("""\
[defaults]
baseline = "http://agg:8000/v1"
target = "http://pd:8000/v1"
model = "llama-7b"
api_key = "sk-test"
max_tokens = 128

[batch]
concurrency = 10
logprob_gap_threshold = 0.05
dataset = "data/gsm8k.jsonl"
csv = "results.csv"

[kv]
max_abs_threshold = 0.001
cosine_threshold = 0.9999

[report]
output = "reports/latest.html"
""")
        config = load_config(config_file)
        assert config.defaults.baseline == "http://agg:8000/v1"
        assert config.defaults.target == "http://pd:8000/v1"
        assert config.defaults.model == "llama-7b"
        assert config.defaults.api_key == "sk-test"
        assert config.defaults.max_tokens == 128
        assert config.batch.concurrency == 10
        assert config.batch.logprob_gap_threshold == 0.05
        assert config.batch.dataset == "data/gsm8k.jsonl"
        assert config.batch.csv == "results.csv"
        assert config.kv.max_abs_threshold == 0.001
        assert config.kv.cosine_threshold == 0.9999
        assert config.report.output == "reports/latest.html"

    def test_partial_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text("""\
[defaults]
baseline = "http://agg:8000"
""")
        config = load_config(config_file)
        assert config.defaults.baseline == "http://agg:8000"
        assert config.defaults.model == "default"  # default value
        assert config.batch.concurrency == 5  # default value

    def test_empty_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text("")
        config = load_config(config_file)
        assert config.defaults.baseline is None
        assert config.batch.concurrency == 5

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.toml")

    def test_invalid_toml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad.toml"
        config_file.write_text("this is not valid toml [[[")
        with pytest.raises(Exception):  # noqa: B017
            load_config(config_file)

    def test_unknown_keys_ignored(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text("""\
[defaults]
baseline = "http://agg:8000"
unknown_key = "should be ignored"
""")
        config = load_config(config_file)
        assert config.defaults.baseline == "http://agg:8000"


class TestDiscoverConfig:
    def test_discovers_in_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_file = tmp_path / "xpyd-acc.toml"
        config_file.write_text("""\
[defaults]
model = "discovered-model"
""")
        monkeypatch.chdir(tmp_path)
        config = discover_config()
        assert config is not None
        assert config.defaults.model == "discovered-model"

    def test_no_config_in_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        config = discover_config()
        assert config is None


class TestMergeCliArgs:
    def test_cli_overrides_config(self) -> None:
        config = AppConfig(
            defaults=DefaultsConfig(baseline="http://config:8000", model="config-model"),
        )
        args = {"baseline": "http://cli:8000", "model": None, "target": None}
        merged = merge_cli_args(config, args, "compare-logprobs")
        assert merged["baseline"] == "http://cli:8000"  # CLI wins
        assert merged["model"] == "config-model"  # config fills None

    def test_config_fills_none(self) -> None:
        config = AppConfig(
            defaults=DefaultsConfig(
                baseline="http://config:8000",
                target="http://target:8000",
            ),
        )
        args = {"baseline": None, "target": None, "model": None}
        merged = merge_cli_args(config, args, "compare-logprobs")
        assert merged["baseline"] == "http://config:8000"
        assert merged["target"] == "http://target:8000"

    def test_batch_specific_merge(self) -> None:
        config = AppConfig(
            batch=BatchConfig(concurrency=20, dataset="data.jsonl"),
        )
        args = {"concurrency": None, "dataset": None, "csv": None, "baseline": None,
                "target": None, "model": None, "api_key": None, "max_tokens": None}
        merged = merge_cli_args(config, args, "batch-compare")
        assert merged["concurrency"] == 20
        assert merged["dataset"] == "data.jsonl"

    def test_kv_specific_merge(self) -> None:
        config = AppConfig(
            kv=KVConfig(max_abs_threshold=0.01, cosine_threshold=0.99),
        )
        args = {"max_abs_threshold": None, "cosine_threshold": None,
                "baseline": None, "target": None}
        merged = merge_cli_args(config, args, "check-kv")
        assert merged["max_abs_threshold"] == 0.01
        assert merged["cosine_threshold"] == 0.99

    def test_report_specific_merge(self) -> None:
        config = AppConfig(
            report=ReportConfig(output="custom.html"),
        )
        args = {"output": None, "input": "data.json"}
        merged = merge_cli_args(config, args, "report")
        assert merged["output"] == "custom.html"

    def test_no_config_passthrough(self) -> None:
        config = AppConfig()
        args = {"baseline": "http://x:8000", "model": "m"}
        merged = merge_cli_args(config, args, "compare-logprobs")
        assert merged["baseline"] == "http://x:8000"
        assert merged["model"] == "m"
