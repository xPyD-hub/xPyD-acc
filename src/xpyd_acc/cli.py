"""CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="xpyd-acc",
        description="PD disaggregation accuracy diagnostic tool",
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

    diag = sub.add_parser("diagnose", help="Run full diagnostic pipeline")
    diag.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    diag.add_argument("--target", required=True, help="Target endpoint URL")
    diag.add_argument("--prompt", required=True, help="Prompt to send")
    diag.add_argument("--model", default=None, help="Model name (default: default)")
    diag.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    diag.add_argument("--api-key", default=None, help="API key for endpoints")
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

    rp = sub.add_parser("report", help="Generate HTML report from batch comparison JSON")
    rp.add_argument("--input", required=True, help="Path to batch results JSON file")
    rp.add_argument("--output", default=None, help="Output HTML file path (default: report.html)")

    det = sub.add_parser("detect", help="Detect xPyD endpoint type")
    det.add_argument("url", help="Endpoint URL to probe")
    det.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")

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

    # Apply hardcoded defaults for any remaining None values
    _FINAL_DEFAULTS: dict[str, object] = {
        "model": "default",
        "max_tokens": 64,
        "api_key": "no-key",
        "concurrency": 5,
        "logprob_gap_threshold": 0.1,
        "output": "report.html",
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
    elif args.command == "detect":
        asyncio.run(_run_detect(args))
    elif args.command == "check-kv":
        _run_check_kv(args)
    elif args.command == "diagnose":
        asyncio.run(_run_diagnose(args))
    elif args.command == "report":
        _run_report(args)
    else:
        print(f"xpyd-acc {args.command} — not yet implemented")


async def _run_batch_compare(args: argparse.Namespace) -> None:
    """Run batch dataset comparison."""
    from xpyd_acc.batch_compare import export_csv, format_report, load_dataset, run_batch

    samples = load_dataset(args.dataset)
    print(f"Loaded {len(samples)} samples from {args.dataset}")

    report = await run_batch(
        samples,
        args.baseline,
        args.target,
        model=args.model,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
        logprob_gap_threshold=args.logprob_gap_threshold,
        concurrency=args.concurrency,
    )

    print()
    print(format_report(report))

    if args.csv:
        export_csv(report, args.csv)
        print(f"\nCSV exported to {args.csv}")

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
    baseline_result = await baseline_collector.collect(args.prompt, max_tokens=args.max_tokens)

    print(f"Collecting logprobs from target: {args.target}")
    target_result = await target_collector.collect(args.prompt, max_tokens=args.max_tokens)

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
