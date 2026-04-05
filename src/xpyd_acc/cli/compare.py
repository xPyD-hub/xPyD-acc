"""CLI handlers for comparison subcommands (non-batch)."""

from __future__ import annotations

import argparse
import sys

from ._common import _preflight_healthcheck


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
    from xpyd_acc.distribution import (
        TokenDistribution,
        compare_distributions,
        format_distribution_report,
    )
    from xpyd_acc.logprobs import LogprobsCollector, LogprobsComparator
    from xpyd_acc.sampling import SamplingParams

    sampling = SamplingParams.from_args(args)
    top_k = getattr(args, "top_k", None) or 5
    kl_threshold = getattr(args, "kl_threshold", None)
    if kl_threshold is None:
        kl_threshold = 0.1

    baseline_collector = LogprobsCollector(args.baseline, api_key=args.api_key, model=args.model)
    target_collector = LogprobsCollector(args.target, api_key=args.api_key, model=args.model)

    print(f"Collecting logprobs from baseline: {args.baseline}")
    baseline_result = await baseline_collector.collect(
        args.prompt, max_tokens=args.max_tokens,
        retries=args.retries, retry_delay=args.retry_delay,
        sampling_params=sampling,
        top_k=top_k,
    )

    print(f"Collecting logprobs from target: {args.target}")
    target_result = await target_collector.collect(
        args.prompt, max_tokens=args.max_tokens,
        retries=args.retries, retry_delay=args.retry_delay,
        sampling_params=sampling,
        top_k=top_k,
    )

    comparator = LogprobsComparator()
    report = comparator.compare(baseline_result, target_result)
    print()
    print(comparator.format_report(report))

    # Distribution analysis when top_k > 1
    if top_k > 1:
        baseline_dists = [
            TokenDistribution(index=t.index, tokens=t.top_logprobs)
            for t in baseline_result.tokens
        ]
        target_dists = [
            TokenDistribution(index=t.index, tokens=t.top_logprobs)
            for t in target_result.tokens
        ]
        dist_report = compare_distributions(
            baseline_dists, target_dists,
            kl_threshold=kl_threshold,
            baseline_endpoint=args.baseline,
            target_endpoint=args.target,
        )
        print()
        print(format_distribution_report(dist_report))

    if not report.match:
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
