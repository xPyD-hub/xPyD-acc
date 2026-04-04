"""TOML configuration file support for xpyd-acc.

Loads settings from a TOML file and merges with CLI arguments.
CLI flags always override config file values.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AUTO_CONFIG_NAME = "xpyd-acc.toml"


@dataclass
class DefaultsConfig:
    """Default settings shared across subcommands."""

    baseline: str | None = None
    target: str | None = None
    model: str = "default"
    api_key: str = "no-key"
    max_tokens: int = 64
    retries: int = 3
    retry_delay: float = 1.0
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None


@dataclass
class BatchConfig:
    """Batch comparison settings."""

    concurrency: int = 5
    logprob_gap_threshold: float = 0.1
    dataset: str | None = None
    csv: str | None = None


@dataclass
class KVConfig:
    """KV cache comparison settings."""

    max_abs_threshold: float = 1e-3
    cosine_threshold: float = 0.999


@dataclass
class ReportConfig:
    """Report generation settings."""

    output: str = "report.html"


@dataclass
class MatchingConfig:
    """Tolerance-based matching settings."""

    normalize_whitespace: bool = False
    ignore_case: bool = False
    numeric_tolerance: float | None = None


@dataclass
class AppConfig:
    """Top-level application configuration."""

    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    kv: KVConfig = field(default_factory=KVConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)


def _parse_section(data: dict[str, Any], cls: type) -> Any:
    """Parse a dict into a dataclass, ignoring unknown keys."""
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


def load_config(path: str | Path) -> AppConfig:
    """Load configuration from a TOML file.

    Args:
        path: Path to the TOML configuration file.

    Returns:
        Parsed AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("rb") as f:
        raw = tomllib.load(f)

    defaults = _parse_section(raw.get("defaults", {}), DefaultsConfig)
    batch = _parse_section(raw.get("batch", {}), BatchConfig)
    kv = _parse_section(raw.get("kv", {}), KVConfig)
    report = _parse_section(raw.get("report", {}), ReportConfig)
    matching = _parse_section(raw.get("matching", {}), MatchingConfig)

    return AppConfig(defaults=defaults, batch=batch, kv=kv, report=report, matching=matching)


def discover_config() -> AppConfig | None:
    """Auto-discover xpyd-acc.toml in the current directory.

    Returns:
        AppConfig if found, None otherwise.
    """
    candidate = Path.cwd() / AUTO_CONFIG_NAME
    if candidate.exists():
        return load_config(candidate)
    return None


def merge_cli_args(config: AppConfig, args: dict[str, Any], command: str) -> dict[str, Any]:
    """Merge config file values with CLI arguments.

    CLI arguments (non-None values) take precedence over config values.

    Args:
        config: Loaded AppConfig.
        args: CLI argument dict (from argparse Namespace).
        command: The subcommand name.

    Returns:
        Merged argument dict ready for use.
    """
    merged = dict(args)

    # Apply defaults
    defaults_map = {
        "baseline": config.defaults.baseline,
        "target": config.defaults.target,
        "model": config.defaults.model,
        "api_key": config.defaults.api_key,
        "max_tokens": config.defaults.max_tokens,
        "retries": config.defaults.retries,
        "retry_delay": config.defaults.retry_delay,
        "temperature": config.defaults.temperature,
        "top_p": config.defaults.top_p,
        "seed": config.defaults.seed,
    }

    for key, config_val in defaults_map.items():
        if key in merged and merged[key] is None and config_val is not None:
            merged[key] = config_val

    # Apply command-specific config
    if command == "batch-compare":
        batch_map = {
            "concurrency": config.batch.concurrency,
            "logprob_gap_threshold": config.batch.logprob_gap_threshold,
            "dataset": config.batch.dataset,
            "csv": config.batch.csv,
        }
        for key, config_val in batch_map.items():
            if key in merged and merged[key] is None and config_val is not None:
                merged[key] = config_val

    elif command in ("check-kv", "diagnose"):
        kv_map: dict[str, Any] = {}
        if command == "check-kv":
            kv_map = {
                "max_abs_threshold": config.kv.max_abs_threshold,
                "cosine_threshold": config.kv.cosine_threshold,
            }
        else:
            kv_map = {
                "kv_max_abs_threshold": config.kv.max_abs_threshold,
                "kv_cosine_threshold": config.kv.cosine_threshold,
            }
        for key, config_val in kv_map.items():
            if key in merged and merged[key] is None and config_val is not None:
                merged[key] = config_val

    elif command == "report":
        if "output" in merged and merged["output"] is None:
            merged["output"] = config.report.output

    # Apply matching config for commands that support tolerance
    if command in ("batch-compare", "compare-streaming"):
        matching_map: dict[str, Any] = {
            "normalize_whitespace": config.matching.normalize_whitespace,
            "ignore_case": config.matching.ignore_case,
            "numeric_tolerance": config.matching.numeric_tolerance,
        }
        for key, config_val in matching_map.items():
            if key in merged and (merged[key] is None or merged[key] is False):
                if config_val is not None and config_val is not False:
                    merged[key] = config_val

    return merged
