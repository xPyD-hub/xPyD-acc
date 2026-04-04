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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="xpyd-acc",
        description="PD disaggregation accuracy diagnostic tool",
    )
    parser.add_argument(
        "-V", "--version", action="version",
        version=f"%(prog)s {_get_version()}",
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

    args = parser.parse_args(argv)
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

    baseline_collector = LogprobsCollector(args.baseline, api_key=args.api_key, model=args.model)
    target_collector = LogprobsCollector(args.target, api_key=args.api_key, model=args.model)

    print(f"Collecting logprobs from baseline: {args.baseline}")
    baseline_result = await baseline_collector.collect(
        args.prompt, max_tokens=args.max_tokens,
        retries=args.retries, retry_delay=args.retry_delay,
    )

    print(f"Collecting logprobs from target: {args.target}")
    target_result = await target_collector.collect(
        args.prompt, max_tokens=args.max_tokens,
        retries=args.retries, retry_delay=args.retry_delay,
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

    from xpyd_acc.streaming import (
        StreamingCollector,
        StreamingComparator,
        collect_stream,
        format_streaming_report,
    )

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
        )
        baseline_elapsed = time_mod.monotonic() - baseline_start

        target_start = time_mod.monotonic()
        target_tokens = await collect_stream(
            target, args.prompt, max_tokens=args.max_tokens, timeout=args.timeout,
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
