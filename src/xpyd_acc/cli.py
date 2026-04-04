"""CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys


def _get_version() -> str:
    """Return the package version."""
    try:
        from importlib.metadata import version
        return version("xpyd-acc")
    except Exception:
        return "unknown"


def _add_sampling_args(parser: argparse.ArgumentParser) -> None:
    """Add --temperature, --top-p, --seed flags to a subcommand parser."""
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Named profile/preset to activate (e.g., greedy, stochastic)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (0 for greedy/deterministic)",
    )
    parser.add_argument(
        "--top-p", type=float, default=None, dest="top_p",
        help="Nucleus sampling top-p value",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible generation",
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

    # compare-logprobs
    lp = sub.add_parser("compare-logprobs", help="Compare logprobs between two endpoints")
    lp.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    lp.add_argument("--target", required=True, help="Target endpoint URL")
    lp.add_argument("--prompt", required=True, help="Prompt to send")
    lp.add_argument("--model", default=None, help="Model name (default: default)")
    lp.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    lp.add_argument("--api-key", default=None, help="API key for both endpoints")
    lp.add_argument("--retries", type=int, default=None, help="Max retry attempts (default: 3)")
    lp.add_argument(
        "--retry-delay", type=float, default=None,
        help="Base retry delay in seconds (default: 1.0)",
    )
    _add_sampling_args(lp)

    diag = sub.add_parser("diagnose", help="Run full diagnostic pipeline")
    diag.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    diag.add_argument("--target", required=True, help="Target endpoint URL")
    diag.add_argument("--prompt", required=True, help="Prompt to send")
    diag.add_argument("--model", default=None, help="Model name (default: default)")
    diag.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    diag.add_argument("--api-key", default=None, help="API key for endpoints")
    diag.add_argument("--retries", type=int, default=None, help="Max retry attempts (default: 3)")
    diag.add_argument(
        "--retry-delay", type=float, default=None,
        help="Base retry delay in seconds (default: 1.0)",
    )
    diag.add_argument("--kv-baseline", default=None, help="Path to baseline KV cache (.npz)")
    diag.add_argument("--kv-target", default=None, help="Path to target KV cache (.npz)")
    diag.add_argument(
        "--kv-max-abs-threshold", type=float, default=None,
        help="KV cache max absolute diff threshold (default: 1e-3)",
    )
    diag.add_argument(
        "--kv-cosine-threshold", type=float, default=None,
        help="KV cache cosine similarity threshold (default: 0.999)",
    )
    diag.add_argument("--json", action="store_true", help="Output report as JSON")
    _add_sampling_args(diag)

    oc = sub.add_parser("compare-output", help="Compare text outputs from two endpoints")
    oc_input = oc.add_mutually_exclusive_group(required=True)
    oc_input.add_argument("--baseline-text", help="Baseline output text (inline)")
    oc_input.add_argument("--baseline-file", help="Path to file with baseline output")
    oc.add_argument("--target-text", help="Target output text (inline)")
    oc.add_argument("--target-file", help="Path to file with target output")

    bc = sub.add_parser("batch-compare", help="Run batch dataset comparison")
    bc.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    bc.add_argument("--target", required=True, help="Target endpoint URL")
    bc.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    bc.add_argument("--model", default=None, help="Model name (default: default)")
    bc.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    bc.add_argument("--api-key", default=None, help="API key for endpoints")
    bc.add_argument(
        "--concurrency", type=int, default=None,
        help="Max concurrent requests (default: 5)",
    )
    bc.add_argument(
        "--logprob-gap-threshold", type=float, default=None,
        help="Logprob gap threshold for bug vs uncertainty (default: 0.1)",
    )
    bc.add_argument("--csv", default=None, help="Path to export CSV results")
    bc.add_argument("--json", default=None, dest="json_path", help="Path to export JSON results")
    bc.add_argument("--markdown", default=None, help="Path to export Markdown report")
    bc.add_argument("--retries", type=int, default=None, help="Max retry attempts (default: 3)")
    bc.add_argument(
        "--retry-delay", type=float, default=None,
        help="Base retry delay in seconds (default: 1.0)",
    )
    bc.add_argument(
        "--no-progress", action="store_true", default=False,
        help="Disable progress bar during batch comparison",
    )
    bc.add_argument(
        "--skip-healthcheck", action="store_true", default=False,
        help="Skip pre-flight endpoint health check",
    )
    bc.add_argument(
        "--template", default=None,
        help="Prompt template: built-in name (gsm8k, mmlu, etc.) or path to YAML/TOML file",
    )
    bc.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Validate setup without sending API requests",
    )
    bc.add_argument(
        "--normalize-whitespace", action="store_true", default=False,
        help="Collapse and strip whitespace before comparison",
    )
    bc.add_argument(
        "--ignore-case", action="store_true", default=False,
        help="Case-insensitive text matching",
    )
    bc.add_argument(
        "--numeric-tolerance", type=float, default=None,
        help="Treat numbers within tolerance as equal",
    )
    bc.add_argument(
        "--rerun", default=None, metavar="REPORT_JSON",
        help="Rerun only divergent samples from a previous JSON report",
    )
    bc.add_argument(
        "--rerun-merge", action="store_true", default=False,
        help="Merge rerun results back into the original report file",
    )
    bc.add_argument(
        "--timeout", type=float, default=None,
        help="HTTP request timeout in seconds (default: 120.0)",
    )
    _add_sampling_args(bc)

    rp = sub.add_parser("report", help="Generate HTML report from batch comparison JSON")
    rp.add_argument("--input", required=True, help="Path to batch results JSON file")
    rp.add_argument("--output", default=None, help="Output HTML file path (default: report.html)")

    cs = sub.add_parser("compare-streaming", help="Compare SSE streaming outputs")
    cs.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    cs.add_argument("--target", required=True, help="Target endpoint URL")
    cs.add_argument("--prompt", required=True, help="Prompt to send")
    cs.add_argument("--model", default=None, help="Model name (default: default)")
    cs.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    cs.add_argument("--api-key", default=None, help="API key for both endpoints")
    cs.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    cs.add_argument(
        "--skip-healthcheck", action="store_true", default=False,
        help="Skip pre-flight endpoint health check",
    )
    cs.add_argument(
        "--timing", action="store_true", default=False,
        help="Enable token timing analysis (TTFT, inter-token latency)",
    )
    cs.add_argument(
        "--normalize-whitespace", action="store_true", default=False,
        help="Collapse and strip whitespace before comparison",
    )
    cs.add_argument(
        "--ignore-case", action="store_true", default=False,
        help="Case-insensitive text matching",
    )
    cs.add_argument(
        "--numeric-tolerance", type=float, default=None,
        help="Treat numbers within tolerance as equal",
    )
    _add_sampling_args(cs)

    hc = sub.add_parser("healthcheck", help="Check endpoint health")
    hc.add_argument("url", nargs="+", help="Endpoint URL(s) to check")
    hc.add_argument("--api-key", default=None, help="API key for endpoints")
    hc.add_argument("--timeout", type=float, default=10.0, help="Timeout per endpoint in seconds")

    det = sub.add_parser("detect", help="Detect xPyD endpoint type")
    det.add_argument("url", help="Endpoint URL to probe")
    det.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")

    reg = sub.add_parser("regression", help="Detect regressions between two batch runs")
    reg.add_argument("--baseline", required=True, help="Path to baseline batch result JSON")
    reg.add_argument("--current", required=True, help="Path to current batch result JSON")
    reg.add_argument(
        "--json", dest="json_path", default=None,
        help="Export regression report as JSON",
    )

    kv = sub.add_parser("check-kv", help="Check KV cache numerical accuracy")
    kv.add_argument("--baseline", required=True, help="Path to baseline KV cache (.npz)")
    kv.add_argument("--target", required=True, help="Path to target KV cache (.npz)")
    kv.add_argument(
        "--max-abs-threshold", type=float, default=None,
        help="Max absolute diff threshold for divergence (default: 1e-3)",
    )
    kv.add_argument(
        "--cosine-threshold", type=float, default=None,
        help="Cosine similarity threshold for divergence (default: 0.999)",
    )
    kv.add_argument("--json", action="store_true", help="Output report as JSON")

    # aggregate
    agg = sub.add_parser("aggregate", help="Aggregate multiple batch run reports")
    agg.add_argument(
        "--reports", nargs="+", required=True,
        help="Paths to batch comparison JSON report files",
    )
    agg.add_argument(
        "--json", default=None, dest="json_path",
        help="Export aggregated report as JSON to this path",
    )

    # watch
    wp = sub.add_parser("watch", help="Continuous divergence monitoring")
    wp.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    wp.add_argument("--target", required=True, help="Target endpoint URL")
    wp.add_argument("--prompt", required=True, help="Prompt to compare")
    wp.add_argument("--model", default=None, help="Model name")
    wp.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    wp.add_argument("--api-key", default=None, help="API key for endpoints")
    wp.add_argument(
        "--interval", type=float, default=60.0,
        help="Seconds between iterations (default: 60)",
    )
    wp.add_argument(
        "--max-iterations", type=int, default=None,
        help="Stop after N iterations (default: unlimited)",
    )
    wp.add_argument(
        "--alert-threshold", type=int, default=None,
        help="Exit code 1 after N consecutive failures",
    )
    wp.add_argument("--log", default=None, dest="log_path", help="JSON log file path")
    wp.add_argument("--retries", type=int, default=None, help="Max retry attempts")
    wp.add_argument("--retry-delay", type=float, default=None, help="Base retry delay (seconds)")
    wp.add_argument("--skip-healthcheck", action="store_true", help="Skip pre-flight healthcheck")
    _add_sampling_args(wp)

    sub.add_parser("profiles", help="List available named profiles")

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

    # Handle 'profiles' subcommand
    if args.command == "profiles":
        _run_profiles(config)
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

    if args.command == "batch-compare":
        asyncio.run(_run_batch_compare(args))
    elif args.command == "compare-output":
        _run_compare_output(args)
    elif args.command == "compare-logprobs":
        asyncio.run(_run_compare_logprobs(args))
    elif args.command == "compare-streaming":
        asyncio.run(_run_compare_streaming(args))
    elif args.command == "healthcheck":
        asyncio.run(_run_healthcheck(args))
    elif args.command == "detect":
        asyncio.run(_run_detect(args))
    elif args.command == "check-kv":
        _run_check_kv(args)
    elif args.command == "diagnose":
        asyncio.run(_run_diagnose(args))
    elif args.command == "report":
        _run_report(args)
    elif args.command == "regression":
        _run_regression(args)
    elif args.command == "aggregate":
        _run_aggregate(args)
    elif args.command == "watch":
        _run_watch(args)
    else:
        print(f"xpyd-acc {args.command} — not yet implemented")


async def _preflight_healthcheck(
    urls: list[str], api_key: str = "no-key", timeout: float = 10.0,
) -> None:
    """Run pre-flight health check; exit if any endpoint is unhealthy."""
    from xpyd_acc.healthcheck import check_endpoints, format_healthcheck

    results = await check_endpoints(urls, api_key=api_key, timeout=timeout)
    print(format_healthcheck(results))
    if not all(r.healthy for r in results):
        print("\nAborting: unhealthy endpoint(s). Use --skip-healthcheck to bypass.")
        sys.exit(1)
    print()


def _run_profiles(config: object) -> None:
    """List all available named profiles."""
    from xpyd_acc.profiles import list_profiles, parse_profiles

    user_profiles = parse_profiles(config.profiles_raw) if config is not None else None
    all_profiles = list_profiles(user_profiles)

    if not all_profiles:
        print("No profiles available.")
        return

    print("Available profiles:\n")
    for name in sorted(all_profiles):
        profile = all_profiles[name]
        settings = profile.to_dict()
        if settings:
            parts = [f"{k}={v}" for k, v in settings.items()]
            print(f"  {name}: {', '.join(parts)}")
        else:
            print(f"  {name}: (empty)")


async def _run_healthcheck(args: argparse.Namespace) -> None:
    """Run standalone endpoint health check."""
    from xpyd_acc.healthcheck import check_endpoints, format_healthcheck

    results = await check_endpoints(
        args.url, api_key=args.api_key or "no-key", timeout=args.timeout,
    )
    print(format_healthcheck(results))
    if not all(r.healthy for r in results):
        sys.exit(1)


async def _run_batch_compare(args: argparse.Namespace) -> None:
    """Run batch dataset comparison."""
    # Handle rerun mode
    if getattr(args, "rerun", None):
        await _run_rerun(args)
        return

    # Handle dry run mode
    if getattr(args, "dry_run", False):
        from xpyd_acc.dry_run import format_dry_run, run_dry_run

        result = await run_dry_run(
            args.dataset,
            args.baseline,
            args.target,
            template=args.template,
            skip_healthcheck=args.skip_healthcheck,
            model=args.model,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
            concurrency=args.concurrency,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
        print(format_dry_run(result))
        if getattr(args, "json_path", None):
            from pathlib import Path

            Path(args.json_path).write_text(result.to_json())
            print(f"\nDry run report exported to {args.json_path}")
        sys.exit(0 if result.valid else 1)

    from xpyd_acc.batch_compare import (
        export_csv,
        export_markdown,
        format_report,
        load_dataset,
        run_batch,
    )
    from xpyd_acc.sampling import SamplingParams

    sampling = SamplingParams.from_args(args)
    samples = load_dataset(args.dataset)

    # Apply template if specified
    if args.template:
        from xpyd_acc.templates import resolve_template

        template = resolve_template(args.template)
        print(f"Using template: {template.name}")
        for sample in samples:
            variables = {"prompt": sample.prompt, **sample.metadata}
            sample.prompt = template.render(variables)

    print(f"Loaded {len(samples)} samples from {args.dataset}")

    if not args.skip_healthcheck:
        await _preflight_healthcheck([args.baseline, args.target], api_key=args.api_key)

    # Set up Rich progress bar unless disabled or non-TTY
    use_progress = not args.no_progress and sys.stderr.isatty()
    progress_ctx = None
    task_id = None

    if use_progress:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        progress_ctx = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )

    def on_progress(completed: int, total: int) -> None:
        if progress_ctx is not None and task_id is not None:
            progress_ctx.update(task_id, completed=completed)

    if progress_ctx is not None:
        progress_ctx.start()
        task_id = progress_ctx.add_task("Comparing samples", total=len(samples))

    try:
        from xpyd_acc.output_compare import MatchConfig

        match_config = MatchConfig(
            normalize_whitespace=args.normalize_whitespace,
            ignore_case=args.ignore_case,
            numeric_tolerance=args.numeric_tolerance,
        )
        # Only pass config if any tolerance is enabled
        effective_match = match_config if (
            match_config.normalize_whitespace
            or match_config.ignore_case
            or match_config.numeric_tolerance is not None
        ) else None

        report = await run_batch(
            samples,
            args.baseline,
            args.target,
            model=args.model,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
            logprob_gap_threshold=args.logprob_gap_threshold,
            concurrency=args.concurrency,
            retries=args.retries,
            retry_delay=args.retry_delay,
            on_progress=on_progress if use_progress else None,
            match_config=effective_match,
            sampling_params=sampling,
            timeout=args.timeout,
        )
    finally:
        if progress_ctx is not None:
            progress_ctx.stop()

    print()
    print(format_report(report))

    if args.csv:
        export_csv(report, args.csv)
        print(f"\nCSV exported to {args.csv}")

    if args.json_path:
        from pathlib import Path
        Path(args.json_path).write_text(report.to_json())
        print(f"\nJSON exported to {args.json_path}")

    if args.markdown:
        export_markdown(report, args.markdown)
        print(f"\nMarkdown exported to {args.markdown}")

    if report.divergent_samples > 0:
        sys.exit(1)


async def _run_rerun(args: argparse.Namespace) -> None:
    """Run selective sample rerun from a previous report."""
    from pathlib import Path

    from xpyd_acc.batch_compare import (
        export_csv,
        export_markdown,
        format_report,
        run_batch,
    )
    from xpyd_acc.rerun import load_divergent_samples, merge_rerun_results
    from xpyd_acc.sampling import SamplingParams

    sampling = SamplingParams.from_args(args)

    plan = load_divergent_samples(args.rerun)
    print(
        f"Rerun: {plan.divergent_count} divergent samples "
        f"out of {plan.total_in_report} total"
    )

    if not args.skip_healthcheck:
        await _preflight_healthcheck([args.baseline, args.target], api_key=args.api_key)

    # Apply template if specified
    if args.template:
        from xpyd_acc.templates import resolve_template

        template = resolve_template(args.template)
        print(f"Using template: {template.name}")
        for sample in plan.divergent_samples:
            variables = {"prompt": sample.prompt, **sample.metadata}
            sample.prompt = template.render(variables)

    # Set up Rich progress bar unless disabled or non-TTY
    use_progress = not args.no_progress and sys.stderr.isatty()
    progress_ctx = None
    task_id = None

    if use_progress:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        progress_ctx = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )

    def on_progress(completed: int, total: int) -> None:
        if progress_ctx is not None and task_id is not None:
            progress_ctx.update(task_id, completed=completed)

    if progress_ctx is not None:
        progress_ctx.start()
        task_id = progress_ctx.add_task("Rerunning samples", total=len(plan.divergent_samples))

    try:
        from xpyd_acc.output_compare import MatchConfig

        match_config = MatchConfig(
            normalize_whitespace=args.normalize_whitespace,
            ignore_case=args.ignore_case,
            numeric_tolerance=args.numeric_tolerance,
        )
        effective_match = match_config if (
            match_config.normalize_whitespace
            or match_config.ignore_case
            or match_config.numeric_tolerance is not None
        ) else None

        report = await run_batch(
            plan.divergent_samples,
            args.baseline,
            args.target,
            model=args.model,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
            logprob_gap_threshold=args.logprob_gap_threshold,
            concurrency=args.concurrency,
            retries=args.retries,
            retry_delay=args.retry_delay,
            on_progress=on_progress if use_progress else None,
            match_config=effective_match,
            sampling_params=sampling,
            timeout=args.timeout,
        )
    finally:
        if progress_ctx is not None:
            progress_ctx.stop()

    # Handle merge mode
    if args.rerun_merge:
        report = merge_rerun_results(args.rerun, report)
        # Overwrite the original report
        Path(args.rerun).write_text(report.to_json())
        print(f"\nMerged results written back to {args.rerun}")

    print()
    print(format_report(report))

    if args.csv:
        export_csv(report, args.csv)
        print(f"\nCSV exported to {args.csv}")

    if args.json_path:
        Path(args.json_path).write_text(report.to_json())
        print(f"\nJSON exported to {args.json_path}")

    if args.markdown:
        export_markdown(report, args.markdown)
        print(f"\nMarkdown exported to {args.markdown}")

    if report.divergent_samples > 0:
        sys.exit(1)


def _run_compare_output(args: argparse.Namespace) -> None:
    """Run text output comparison."""
    from pathlib import Path

    from xpyd_acc.output_compare import OutputComparator

    if args.baseline_text is not None:
        baseline = args.baseline_text
    else:
        baseline = Path(args.baseline_file).read_text()

    if args.target_text is not None:
        target = args.target_text
    elif args.target_file is not None:
        target = Path(args.target_file).read_text()
    else:
        print("Error: provide --target-text or --target-file", file=sys.stderr)
        sys.exit(1)

    comparator = OutputComparator()
    report = comparator.compare(baseline, target)
    print(OutputComparator.format_report(report))

    if not report.exact_match:
        sys.exit(1)


async def _run_compare_logprobs(args: argparse.Namespace) -> None:
    """Run logprobs comparison between two endpoints."""
    from xpyd_acc.logprobs import LogprobsCollector, LogprobsComparator
    from xpyd_acc.sampling import SamplingParams

    sampling = SamplingParams.from_args(args)
    baseline_collector = LogprobsCollector(args.baseline, api_key=args.api_key, model=args.model)
    target_collector = LogprobsCollector(args.target, api_key=args.api_key, model=args.model)

    print(f"Collecting logprobs from baseline: {args.baseline}")
    baseline_result = await baseline_collector.collect(
        args.prompt, max_tokens=args.max_tokens,
        retries=args.retries, retry_delay=args.retry_delay,
        sampling_params=sampling,
    )

    print(f"Collecting logprobs from target: {args.target}")
    target_result = await target_collector.collect(
        args.prompt, max_tokens=args.max_tokens,
        retries=args.retries, retry_delay=args.retry_delay,
        sampling_params=sampling,
    )

    comparator = LogprobsComparator()
    report = comparator.compare(baseline_result, target_result)
    print()
    print(comparator.format_report(report))

    if not report.match:
        sys.exit(1)


async def _run_diagnose(args: argparse.Namespace) -> None:
    """Run the full diagnostic pipeline."""
    from xpyd_acc.diagnose import DiagnosticPipeline, format_rich_report
    from xpyd_acc.sampling import SamplingParams

    sampling = SamplingParams.from_args(args)
    pipeline = DiagnosticPipeline(
        baseline_url=args.baseline,
        target_url=args.target,
        prompt=args.prompt,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        kv_baseline_path=args.kv_baseline,
        kv_target_path=args.kv_target,
        kv_max_abs_threshold=args.kv_max_abs_threshold,
        kv_cosine_threshold=args.kv_cosine_threshold,
        sampling_params=sampling,
    )

    report = await pipeline.run()

    if args.json:
        print(report.to_json())
    else:
        print(format_rich_report(report))

    if not report.overall_pass:
        sys.exit(1)


async def _run_compare_streaming(args: argparse.Namespace) -> None:
    """Run streaming comparison between two endpoints."""
    import time as time_mod

    from xpyd_acc.sampling import SamplingParams
    from xpyd_acc.streaming import (
        StreamingCollector,
        StreamingComparator,
        collect_stream,
        format_streaming_report,
    )

    sampling = SamplingParams.from_args(args)
    baseline = StreamingCollector(args.baseline, api_key=args.api_key, model=args.model)
    target = StreamingCollector(args.target, api_key=args.api_key, model=args.model)

    if not args.skip_healthcheck:
        await _preflight_healthcheck([args.baseline, args.target], api_key=args.api_key)

    print(f"Streaming from baseline: {args.baseline}")
    print(f"Streaming from target:   {args.target}")

    if args.timing:
        from xpyd_acc.output_compare import MatchConfig as _MC

        _stream_match = _MC(
            normalize_whitespace=args.normalize_whitespace,
            ignore_case=args.ignore_case,
            numeric_tolerance=args.numeric_tolerance,
        )
        _eff_match = _stream_match if (
            _stream_match.normalize_whitespace
            or _stream_match.ignore_case
            or _stream_match.numeric_tolerance is not None
        ) else None

        # Collect with timing info
        baseline_start = time_mod.monotonic()
        baseline_tokens = await collect_stream(
            baseline, args.prompt, max_tokens=args.max_tokens, timeout=args.timeout,
            sampling_params=sampling,
        )
        baseline_elapsed = time_mod.monotonic() - baseline_start

        target_start = time_mod.monotonic()
        target_tokens = await collect_stream(
            target, args.prompt, max_tokens=args.max_tokens, timeout=args.timeout,
            sampling_params=sampling,
        )
        target_elapsed = time_mod.monotonic() - target_start

        from xpyd_acc.streaming import compare_token_lists
        from xpyd_acc.timing import compare_timing, compute_timing_stats, format_timing_report

        stream_report = compare_token_lists(
            baseline_tokens, target_tokens,
            elapsed=baseline_elapsed + target_elapsed,
            match_config=_eff_match,
        )
        print()
        print(format_streaming_report(stream_report))

        baseline_stats = compute_timing_stats(baseline_tokens, baseline_start)
        target_stats = compute_timing_stats(target_tokens, target_start)
        timing_report = compare_timing(baseline_stats, target_stats)
        print()
        print(format_timing_report(timing_report))
    else:
        from xpyd_acc.output_compare import MatchConfig as _MC2

        _stream_match2 = _MC2(
            normalize_whitespace=args.normalize_whitespace,
            ignore_case=args.ignore_case,
            numeric_tolerance=args.numeric_tolerance,
        )
        _eff_match2 = _stream_match2 if (
            _stream_match2.normalize_whitespace
            or _stream_match2.ignore_case
            or _stream_match2.numeric_tolerance is not None
        ) else None

        comparator = StreamingComparator()
        report = await comparator.compare(
            baseline, target, args.prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            match_config=_eff_match2,
            sampling_params=sampling,
        )
        print()
        print(format_streaming_report(report))

        if not report.match:
            sys.exit(1)


async def _run_detect(args: argparse.Namespace) -> None:
    """Run endpoint type detection."""
    from xpyd_acc.ecosystem import detect_endpoint_type, format_detect_report

    info = await detect_endpoint_type(args.url, timeout=args.timeout)
    print(format_detect_report(info))


def _run_check_kv(args: argparse.Namespace) -> None:
    """Run KV cache comparison between two npz dumps."""
    from xpyd_acc.kvcache import KVCacheComparator, KVCacheLoader

    baseline = KVCacheLoader.load(args.baseline)
    target = KVCacheLoader.load(args.target)

    comparator = KVCacheComparator(
        max_abs_threshold=args.max_abs_threshold,
        cosine_threshold=args.cosine_threshold,
    )
    report = comparator.compare(
        baseline, target,
        baseline_path=args.baseline,
        target_path=args.target,
    )

    if args.json:
        print(report.to_json())
    else:
        print(KVCacheComparator.format_report(report))

    if not report.match:
        sys.exit(1)


def _run_report(args: argparse.Namespace) -> None:
    """Generate HTML report from batch results JSON."""
    import json as json_mod
    from pathlib import Path

    from xpyd_acc.batch_compare import BatchReport, SampleResult
    from xpyd_acc.report import write_html_report

    data = json_mod.loads(Path(args.input).read_text())

    results = []
    for r in data["results"]:
        results.append(SampleResult(
            sample_id=r["sample_id"],
            prompt=r["prompt"],
            baseline_output=r["baseline_output"],
            target_output=r["target_output"],
            exact_match=r["exact_match"],
            first_divergence_index=r.get("first_divergence_index"),
            baseline_logprob_at_divergence=r.get("baseline_logprob_at_divergence"),
            target_logprob_at_divergence=r.get("target_logprob_at_divergence"),
            logprob_gap=r.get("logprob_gap"),
            classification=r.get("classification", "unknown"),
            context_length=r.get("context_length", 0),
        ))

    report = BatchReport(
        total_samples=data["total_samples"],
        divergent_samples=data["divergent_samples"],
        match_samples=data["match_samples"],
        divergence_rate=data["divergence_rate"],
        results=results,
        divergence_index_mean=data.get("divergence_index_mean"),
        divergence_index_median=data.get("divergence_index_median"),
        logprob_gap_mean=data.get("logprob_gap_mean"),
        logprob_gap_median=data.get("logprob_gap_median"),
        likely_bugs=data.get("likely_bugs", 0),
        likely_uncertainty=data.get("likely_uncertainty", 0),
        unknown_classification=data.get("unknown_classification", 0),
        divergence_by_context_length=data.get("divergence_by_context_length", {}),
    )

    write_html_report(report, args.output)
    print(f"HTML report written to {args.output}")


def _run_regression(args: argparse.Namespace) -> None:
    """Run regression detection between two batch result JSONs."""
    from xpyd_acc.regression import compare_runs, format_regression_report

    report = compare_runs(args.baseline, args.current)
    print(format_regression_report(report))

    if getattr(args, "json_path", None):
        from pathlib import Path

        Path(args.json_path).write_text(report.to_json())
        print(f"\nRegression report exported to {args.json_path}")

    sys.exit(1 if report.has_regressions else 0)


def _run_aggregate(args: argparse.Namespace) -> None:
    """Aggregate multiple batch run reports."""
    from pathlib import Path

    from xpyd_acc.aggregate import (
        aggregate_reports,
        format_aggregated_report,
        load_batch_report_from_json,
    )

    reports = [load_batch_report_from_json(p) for p in args.reports]
    agg_report = aggregate_reports(reports)
    print(format_aggregated_report(agg_report))

    if getattr(args, "json_path", None):
        Path(args.json_path).write_text(agg_report.to_json())
        print(f"\nAggregated report exported to {args.json_path}")



def _run_watch(args: argparse.Namespace) -> None:
    """Run continuous watch mode."""
    from xpyd_acc.config import load_config
    from xpyd_acc.env import apply_env_defaults
    from xpyd_acc.profiles import apply_profile
    from xpyd_acc.sampling import SamplingParams
    from xpyd_acc.watch import run_watch

    config = load_config(getattr(args, "config", None))
    apply_env_defaults(args, config)
    apply_profile(args, config)

    sampling = SamplingParams.from_args(args)
    model = args.model or config.get("defaults", {}).get("model", "default")
    max_tokens = args.max_tokens or config.get("defaults", {}).get("max_tokens", 64)
    api_key = args.api_key or config.get("defaults", {}).get("api_key")
    defaults = config.get("defaults", {})
    retries = args.retries if args.retries is not None else defaults.get("retries", 3)
    retry_delay = (
        args.retry_delay if args.retry_delay is not None
        else defaults.get("retry_delay", 1.0)
    )

    if not getattr(args, "skip_healthcheck", False):
        asyncio.run(_preflight_healthcheck(
            [args.baseline, args.target],
            api_key=api_key or "no-key",
        ))

    summary = asyncio.run(run_watch(
        baseline_url=args.baseline,
        target_url=args.target,
        prompt=args.prompt,
        model=model,
        max_tokens=max_tokens,
        api_key=api_key,
        interval=args.interval,
        max_iterations=args.max_iterations,
        alert_threshold=args.alert_threshold,
        log_path=args.log_path,
        retries=retries,
        retry_delay=retry_delay,
        sampling_params=sampling,
        no_live=False,
    ))

    if args.alert_threshold and summary.consecutive_failures_at_end >= args.alert_threshold:
        sys.exit(1)
