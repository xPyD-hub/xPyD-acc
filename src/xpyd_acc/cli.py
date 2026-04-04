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

    sub.add_parser("diagnose", help="Run full diagnostic pipeline")
    sub.add_parser("check-kv", help="Check KV cache numerical accuracy")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return

    if args.command == "compare-logprobs":
        asyncio.run(_run_compare_logprobs(args))
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
