"""CLI handlers for watch, entropy, length-bias, sensitivity, fingerprint."""

from __future__ import annotations

import argparse
import asyncio
import sys

from ._common import _preflight_healthcheck


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


def _run_entropy(args: argparse.Namespace) -> None:
    """Run entropy analysis on logprob files."""
    import json
    from pathlib import Path

    from xpyd_acc.entropy import (
        entropy_at_divergence,
        entropy_stats,
        format_entropy_comparison,
        format_entropy_stats,
        load_logprobs_file,
        sequence_entropy,
    )

    baseline_lp = load_logprobs_file(args.baseline_logprobs)
    bl_entropies = sequence_entropy(baseline_lp)
    bl_stats = entropy_stats(bl_entropies)

    output: dict = {"baseline_stats": bl_stats.to_dict()}
    print("Baseline " + format_entropy_stats(bl_stats))

    if args.target_logprobs:
        target_lp = load_logprobs_file(args.target_logprobs)
        tg_entropies = sequence_entropy(target_lp)
        tg_stats = entropy_stats(tg_entropies)
        output["target_stats"] = tg_stats.to_dict()
        print("\nTarget " + format_entropy_stats(tg_stats))

        if args.divergence_index is not None:
            comp = entropy_at_divergence(
                baseline_lp, target_lp, args.divergence_index,
                context_window=args.context_window,
            )
            output["comparison"] = comp.to_dict()
            print("\n" + format_entropy_comparison(comp))

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(output, indent=2))
        print(f"\nExported to {args.json_path}")


def _run_length_bias(args: argparse.Namespace) -> None:
    """Run output length bias analysis on a batch report."""
    import json
    from pathlib import Path

    from xpyd_acc.length_bias import (
        analyze_length_bias,
        format_length_bias,
        load_report_file,
    )

    report = load_report_file(args.report)
    result = analyze_length_bias(report, alpha=args.alpha)
    print(format_length_bias(result))

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nExported to {args.json_path}")

    if result.classification != "no_bias":
        raise SystemExit(1)


async def _run_sensitivity(args: argparse.Namespace) -> None:
    """Run prompt sensitivity analysis."""
    from pathlib import Path

    from xpyd_acc.sensitivity import format_sensitivity, run_sensitivity

    result = await run_sensitivity(
        baseline_url=args.baseline,
        target_url=args.target,
        prompt=args.prompt,
        model=args.model or "default",
        max_tokens=getattr(args, "max_tokens", None) or 64,
        api_key=getattr(args, "api_key", None),
        perturbation_count=args.perturbations,
        retries=getattr(args, "retries", None) or 3,
        retry_delay=getattr(args, "retry_delay", None) or 1.0,
        temperature=getattr(args, "temperature", None),
        top_p=getattr(args, "top_p", None),
        seed=getattr(args, "seed", None),
    )

    print(format_sensitivity(result))

    if args.json_path:
        Path(args.json_path).write_text(result.to_json())
        print(f"\nExported to {args.json_path}")

    if result.classification == "systematic":
        raise SystemExit(1)


def _run_fingerprint(args: argparse.Namespace) -> None:
    """Handle the 'fingerprint' subcommand (M55)."""
    import json as _json
    from pathlib import Path

    from xpyd_acc.fingerprint import collect_fingerprint, compare_fingerprints

    api_key = args.api_key or "no-key"

    async def _go():
        fp_baseline = await collect_fingerprint(
            args.baseline,
            api_key=api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
        print(f"Baseline fingerprint: {fp_baseline.hash}  ({fp_baseline.endpoint})")

        result: dict = fp_baseline.to_dict()

        if args.target:
            fp_target = await collect_fingerprint(
                args.target,
                api_key=api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
                retry_delay=args.retry_delay,
            )
            print(f"Target fingerprint:   {fp_target.hash}  ({fp_target.endpoint})")

            cmp = compare_fingerprints(fp_baseline, fp_target)
            if cmp.match:
                print("\n✅ Fingerprints MATCH — endpoints produce identical probe outputs")
            else:
                ndiff = len(cmp.differing_probes)
                print(
                    f"\n❌ Fingerprints DIFFER — "
                    f"{ndiff}/{cmp.total_probes} probes diverged"
                )
                for d in cmp.differing_probes:
                    print(f"  Probe {d['probe_index']}: {d['prompt'][:40]}")
                    print(f"    baseline: {d['output_a'][:60]}")
                    print(f"    target:   {d['output_b'][:60]}")
            result = {
                "baseline": fp_baseline.to_dict(),
                "target": fp_target.to_dict(),
                "comparison": cmp.to_dict(),
            }
            if not cmp.match:
                raise SystemExit(1)

        if args.fp_json:
            Path(args.fp_json).write_text(_json.dumps(result, indent=2))
            print(f"\nExported to {args.fp_json}")

    asyncio.run(_go())
