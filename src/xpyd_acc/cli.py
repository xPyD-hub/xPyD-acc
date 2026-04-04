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
    sub = parser.add_subparsers(dest="command")

    # compare-logprobs
    lp = sub.add_parser("compare-logprobs", help="Compare logprobs between two endpoints")
    lp.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    lp.add_argument("--target", required=True, help="Target endpoint URL")
    lp.add_argument("--prompt", required=True, help="Prompt to send")
    lp.add_argument("--model", default="default", help="Model name")
    lp.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    lp.add_argument("--api-key", default="no-key", help="API key for both endpoints")

    diag = sub.add_parser("diagnose", help="Run full diagnostic pipeline")
    diag.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    diag.add_argument("--target", required=True, help="Target endpoint URL")
    diag.add_argument("--prompt", required=True, help="Prompt to send")
    diag.add_argument("--model", default="default", help="Model name")
    diag.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    diag.add_argument("--api-key", default="no-key", help="API key for endpoints")
    diag.add_argument("--kv-baseline", default=None, help="Path to baseline KV cache (.npz)")
    diag.add_argument("--kv-target", default=None, help="Path to target KV cache (.npz)")
    diag.add_argument(
        "--kv-max-abs-threshold", type=float, default=1e-3,
        help="KV cache max absolute diff threshold (default: 1e-3)",
    )
    diag.add_argument(
        "--kv-cosine-threshold", type=float, default=0.999,
        help="KV cache cosine similarity threshold (default: 0.999)",
    )
    diag.add_argument("--json", action="store_true", help="Output report as JSON")

    kv = sub.add_parser("check-kv", help="Check KV cache numerical accuracy")
    kv.add_argument("--baseline", required=True, help="Path to baseline KV cache (.npz)")
    kv.add_argument("--target", required=True, help="Path to target KV cache (.npz)")
    kv.add_argument(
        "--max-abs-threshold", type=float, default=1e-3,
        help="Max absolute diff threshold for divergence (default: 1e-3)",
    )
    kv.add_argument(
        "--cosine-threshold", type=float, default=0.999,
        help="Cosine similarity threshold for divergence (default: 0.999)",
    )
    kv.add_argument("--json", action="store_true", help="Output report as JSON")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return

    if args.command == "compare-logprobs":
        asyncio.run(_run_compare_logprobs(args))
    elif args.command == "check-kv":
        _run_check_kv(args)
    elif args.command == "diagnose":
        asyncio.run(_run_diagnose(args))
    else:
        print(f"xpyd-acc {args.command} — not yet implemented")


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
