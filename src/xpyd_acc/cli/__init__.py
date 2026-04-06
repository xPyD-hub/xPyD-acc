"""CLI entry point — parser setup, config loading, and command dispatch."""

from __future__ import annotations

import argparse
import asyncio

from . import parsers
from ._common import _get_version
from ._common import _resolve_fail_threshold as _resolve_fail_threshold

# Re-export all handler functions at module level so tests can patch them
# via "xpyd_acc.cli._run_compare_logprobs" etc.
from .analysis import (
    _run_entropy,
    _run_file_compare,
    _run_fingerprint,
    _run_latency_regression,
    _run_length_bias,
    _run_sensitivity,
    _run_watch,
    handle_capture_kv,
    handle_heatmap,
    handle_root_cause,
    handle_token_diff,
)
from .batch import _run_batch_compare
from .benchmark import (
    _run_benchmark,
    _run_bisect,
    _run_concurrency_sweep,
    _run_reproducibility,
)
from .compare import _run_compare_logprobs, _run_compare_output, _run_compare_streaming
from .config_cmd import _run_completion, _run_config, _run_init, _run_profiles
from .data import _run_cache, _run_dataset_stats, _run_history, _run_serve, _run_snapshot
from .diagnose import _run_check_kv, _run_detect, _run_diagnose, _run_healthcheck
from .report import (
    _run_ab_test,
    _run_aggregate,
    _run_annotate,
    _run_auto_threshold,
    _run_cluster,
    _run_diff,
    _run_explain,
    _run_filter,
    _run_grafana_dashboard,
    _run_prometheus,
    _run_regression,
    _run_repl,
    _run_report,
    _run_summary,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="xpyd-acc",
        description="PD disaggregation accuracy diagnostic tool",
    )
    parser.add_argument(
        "-V", "--version", action="version",
        version=f"%(prog)s {_get_version()}",
    )
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    verbosity_group.add_argument(
        "-q", "--quiet", action="store_true", default=False,
        help="Quiet mode (ERROR level only)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to TOML config file (auto-discovers xpyd-acc.toml in cwd if not set)",
    )
    sub = parser.add_subparsers(dest="command")

    parsers.register_all(sub)

    args = parser.parse_args(argv)

    # Setup logging from verbosity flags
    from xpyd_acc.log import setup_logging
    verbosity = -1 if args.quiet else args.verbose
    setup_logging(verbosity)

    if not args.command:
        parser.print_help()
        return

    # Load config file
    from xpyd_acc.config import AppConfig, discover_config, load_config, merge_cli_args

    config: AppConfig | None = None
    if args.config:
        config = load_config(args.config)
    else:
        config = discover_config()

    if config is not None:
        args_dict = vars(args)
        merged = merge_cli_args(config, args_dict, args.command)
        for key, val in merged.items():
            setattr(args, key, val)

    # Apply named profile if --profile was specified
    if hasattr(args, "profile") and args.profile is not None:
        from xpyd_acc.profiles import apply_profile, parse_profiles, resolve_profile

        user_profiles = parse_profiles(config.profiles_raw) if config is not None else None
        try:
            profile = resolve_profile(args.profile, user_profiles)
        except KeyError as exc:
            parser.error(str(exc))
        apply_profile(vars(args), profile)

    # Early-return commands (don't need env/defaults processing)
    # Use module-level names so patches work
    _early = {
        "cluster": lambda: _run_cluster(args),
        "summary": lambda: _run_summary(args),
        "annotate": lambda: _run_annotate(args),
        "explain": lambda: _run_explain(args),
        "reproducibility": lambda: _run_reproducibility(args),
        "fingerprint": lambda: _run_fingerprint(args),
        "root-cause": lambda: handle_root_cause(args),
        "token-diff": lambda: handle_token_diff(args),
        "heatmap": lambda: handle_heatmap(args),
        "capture-kv": lambda: handle_capture_kv(args),
        "filter": lambda: _run_filter(args),
        "serve": lambda: _run_serve(args),
        "grafana-dashboard": lambda: _run_grafana_dashboard(args),
        "prometheus": lambda: _run_prometheus(args),
        "init": lambda: _run_init(args),
        "config": lambda: _run_config(args),
        "profiles": lambda: _run_profiles(config),
        "completion": lambda: _run_completion(args, parser),
        "auto-threshold": lambda: _run_auto_threshold(args),
        "repl": lambda: _run_repl(args),
        "latency-regression": lambda: _run_latency_regression(args),
        "compare-files": lambda: _run_file_compare(args),
    }

    if args.command in _early:
        _early[args.command]()
        return

    # Apply environment variable defaults (priority: CLI > env > config > defaults)
    from xpyd_acc.env import get_env_defaults

    env = get_env_defaults()
    _ENV_MAPPING: dict[str, str | None] = {
        "api_key": env.api_key,
        "baseline": env.baseline_url,
        "target": env.target_url,
        "model": env.model,
    }
    for key, env_val in _ENV_MAPPING.items():
        if env_val is not None and hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, env_val)

    # Apply numeric env defaults separately (typed)
    if env.temperature is not None and hasattr(args, "temperature") and args.temperature is None:
        args.temperature = env.temperature
    if env.top_p is not None and hasattr(args, "top_p") and args.top_p is None:
        args.top_p = env.top_p
    if env.seed is not None and hasattr(args, "seed") and args.seed is None:
        args.seed = env.seed
    if env.timeout is not None and hasattr(args, "timeout") and args.timeout is None:
        args.timeout = env.timeout
    if env.rate_limit is not None and hasattr(args, "rate_limit") and args.rate_limit is None:
        args.rate_limit = env.rate_limit
    if env.max_tokens is not None and hasattr(args, "max_tokens") and args.max_tokens is None:
        args.max_tokens = env.max_tokens
    if env.concurrency is not None and hasattr(args, "concurrency") and args.concurrency is None:
        args.concurrency = env.concurrency

    # Apply hardcoded defaults for any remaining None values
    _FINAL_DEFAULTS: dict[str, object] = {
        "model": "default",
        "max_tokens": 64,
        "api_key": "no-key",
        "concurrency": 5,
        "logprob_gap_threshold": 0.1,
        "output": "report.html",
        "retries": 3,
        "retry_delay": 1.0,
        "max_abs_threshold": 1e-3,
        "cosine_threshold": 0.999,
        "kv_max_abs_threshold": 1e-3,
        "kv_cosine_threshold": 0.999,
        "timeout": 120.0,
    }
    for key, default in _FINAL_DEFAULTS.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, default)

    # Stash config on args for subcommands that need it
    args._config = config

    # Dispatch — references are to module-level names (patchable by tests)
    _dispatch: dict[str, object] = {
        "batch-compare": lambda: asyncio.run(_run_batch_compare(args)),
        "compare-output": lambda: _run_compare_output(args),
        "compare-logprobs": lambda: asyncio.run(_run_compare_logprobs(args)),
        "compare-streaming": lambda: asyncio.run(_run_compare_streaming(args)),
        "healthcheck": lambda: asyncio.run(_run_healthcheck(args)),
        "detect": lambda: asyncio.run(_run_detect(args)),
        "check-kv": lambda: _run_check_kv(args),
        "diagnose": lambda: asyncio.run(_run_diagnose(args)),
        "report": lambda: _run_report(args),
        "regression": lambda: _run_regression(args),
        "diff": lambda: _run_diff(args),
        "aggregate": lambda: _run_aggregate(args),
        "watch": lambda: _run_watch(args),
        "snapshot": lambda: asyncio.run(_run_snapshot(args)),
        "cache": lambda: _run_cache(args),
        "history": lambda: _run_history(args),
        "benchmark": lambda: asyncio.run(_run_benchmark(args)),
        "bisect": lambda: asyncio.run(_run_bisect(args)),
        "dataset-stats": lambda: _run_dataset_stats(args),
        "ab-test": lambda: _run_ab_test(args),
        "concurrency-sweep": lambda: asyncio.run(_run_concurrency_sweep(args)),
        "entropy": lambda: _run_entropy(args),
        "length-bias": lambda: _run_length_bias(args),
        "sensitivity": lambda: asyncio.run(_run_sensitivity(args)),
    }

    handler = _dispatch.get(args.command)
    if handler:
        handler()
    else:
        print(f"xpyd-acc {args.command} — not yet implemented")
