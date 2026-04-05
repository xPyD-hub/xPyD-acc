"""CLI handlers for diagnostic subcommands."""

from __future__ import annotations

import argparse
import sys


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


async def _run_healthcheck(args: argparse.Namespace) -> None:
    """Run standalone endpoint health check."""
    from xpyd_acc.healthcheck import check_endpoints, format_healthcheck

    results = await check_endpoints(
        args.url, api_key=args.api_key or "no-key", timeout=args.timeout,
    )
    print(format_healthcheck(results))
    if not all(r.healthy for r in results):
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
