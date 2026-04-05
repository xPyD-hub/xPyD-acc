"""CLI handlers for benchmark, reproducibility, concurrency-sweep, bisect."""

from __future__ import annotations

import argparse


async def _run_benchmark(args: argparse.Namespace) -> None:
    """Run endpoint latency benchmark."""
    from xpyd_acc.benchmark import run_benchmark
    from xpyd_acc.sampling import SamplingParams

    sp = SamplingParams(
        temperature=getattr(args, "temperature", None),
        top_p=getattr(args, "top_p", None),
        seed=getattr(args, "seed", None),
    )

    await run_benchmark(
        url=args.url,
        prompt=args.prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
        requests=args.requests,
        concurrency=args.concurrency,
        sampling_params=sp,
        json_path=args.json_path,
    )


def _run_reproducibility(args: argparse.Namespace) -> None:
    """Handle the 'reproducibility' subcommand (M62)."""
    import asyncio
    from pathlib import Path

    from xpyd_acc.reproducibility import format_reproducibility, run_reproducibility

    api_key = args.api_key or "no-key"

    async def _go() -> None:
        report = await run_reproducibility(
            url=args.url,
            baseline_url=args.baseline,
            target_url=args.target,
            prompt=args.prompt,
            model=args.model,
            runs=args.runs,
            api_key=api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            retries=args.retries,
            retry_delay=args.retry_delay,
            timeout=args.timeout,
        )

        print(format_reproducibility(report))

        if args.repro_json:
            Path(args.repro_json).write_text(report.to_json())
            print(f"\nExported to {args.repro_json}")

        if args.threshold is not None:
            results = [
                r for r in [report.single, report.baseline, report.target]
                if r is not None
            ]
            for r in results:
                if r.majority_fraction < args.threshold:
                    print(
                        f"\n❌ FAIL: {r.url} majority fraction "
                        f"{r.majority_fraction:.2%} < threshold {args.threshold:.2%}"
                    )
                    raise SystemExit(1)
            print(f"\n✅ PASS: all endpoints above threshold {args.threshold:.2%}")

    asyncio.run(_go())


async def _run_concurrency_sweep(args: argparse.Namespace) -> None:
    """Run concurrency sweep analysis."""
    from pathlib import Path

    from xpyd_acc.concurrency_sweep import format_sweep, run_sweep

    levels = [int(x.strip()) for x in args.levels.split(",")]

    result = await run_sweep(
        baseline_url=args.baseline,
        target_url=args.target,
        dataset_path=args.dataset,
        levels=levels,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        template_path=args.template,
        retries=args.retries,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
        skip_validation=args.skip_validation,
    )

    print(format_sweep(result))

    if args.json_path:
        Path(args.json_path).write_text(result.to_json())
        print(f"\nExported to {args.json_path}")

    if result.any_divergence:
        raise SystemExit(1)


async def _run_bisect(args: argparse.Namespace) -> None:
    """Run bisect to find minimum divergence context length."""
    import pathlib

    from xpyd_acc.bisect import run_bisect
    from xpyd_acc.sampling import SamplingParams

    sp = SamplingParams(
        temperature=getattr(args, "temperature", None),
        top_p=getattr(args, "top_p", None),
        seed=getattr(args, "seed", None),
    )

    def progress(iteration: int, length: int, diverges: bool) -> None:
        status = "❌ DIVERGES" if diverges else "✅ MATCH"
        print(f"  Step {iteration}: length={length} → {status}")

    print(f"🔍 Bisecting divergence over prompt length ({len(args.prompt)} chars)...")
    result = await run_bisect(
        baseline_url=args.baseline,
        target_url=args.target,
        prompt=args.prompt,
        model=args.model,
        min_length=getattr(args, "min_length", None),
        max_length=getattr(args, "max_length", None),
        api_key=getattr(args, "api_key", None),
        sampling=sp,
        retries=args.retries or 3,
        retry_delay=args.retry_delay or 1.0,
        timeout=args.timeout or 120.0,
        progress_callback=progress,
    )

    print()
    if result.never_diverges:
        print("✅ No divergence found at any tested length.")
    elif result.always_diverges:
        print(f"❌ Divergence at all tested lengths (minimum tested: {result.threshold_length}).")
    else:
        print(f"🎯 Divergence threshold: {result.threshold_length} characters.")
    print(f"   Total iterations: {result.total_iterations}")

    json_path = getattr(args, "json", None)
    if json_path:
        pathlib.Path(json_path).write_text(result.to_json())
        print(f"   Exported to {json_path}")
