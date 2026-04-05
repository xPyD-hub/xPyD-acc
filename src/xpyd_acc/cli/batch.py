"""CLI handlers for batch comparison, rerun, and snapshot subcommands."""

from __future__ import annotations

import argparse
import sys

from ._common import (
    _apply_cost_to_report,
    _check_truncation_threshold,
    _maybe_apply_confidence,
    _maybe_send_webhook,
    _preflight_healthcheck,
    _resolve_fail_threshold,
)


async def _run_batch_compare(args: argparse.Namespace) -> None:
    """Run batch dataset comparison."""
    snapshot_path = getattr(args, "snapshot", None)
    has_baseline = args.baseline is not None
    if snapshot_path and has_baseline:
        print("Error: --baseline and --snapshot are mutually exclusive")
        sys.exit(2)
    if not snapshot_path and not has_baseline:
        print("Error: one of --baseline or --snapshot is required")
        sys.exit(2)

    if snapshot_path:
        from .data import _run_batch_with_snapshot
        await _run_batch_with_snapshot(args)
        return

    if getattr(args, "rerun", None):
        await _run_rerun(args)
        return

    target_urls: list[str] = args.target if isinstance(args.target, list) else [args.target]

    if getattr(args, "dry_run", False):
        from xpyd_acc.dry_run import format_dry_run, run_dry_run

        result = await run_dry_run(
            args.dataset, args.baseline, target_urls[0],
            template=args.template, skip_healthcheck=args.skip_healthcheck,
            model=args.model, max_tokens=args.max_tokens, api_key=args.api_key,
            concurrency=args.concurrency, retries=args.retries, retry_delay=args.retry_delay,
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
        run_multi_batch,
    )
    from xpyd_acc.sampling import SamplingParams

    sampling = SamplingParams.from_args(args)
    samples = load_dataset(args.dataset)

    if args.template:
        from xpyd_acc.templates import resolve_template
        template = resolve_template(args.template)
        print(f"Using template: {template.name}")
        for sample in samples:
            variables = {"prompt": sample.prompt, **sample.metadata}
            sample.prompt = template.render(variables)

    print(f"Loaded {len(samples)} samples from {args.dataset}")

    if not args.skip_healthcheck:
        await _preflight_healthcheck([args.baseline, *target_urls], api_key=args.api_key)

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
            TextColumn("[bold blue]{task.description}"), BarColumn(),
            MofNCompleteColumn(), TextColumn("•"),
            TimeElapsedColumn(), TextColumn("•"), TimeRemainingColumn(),
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
        effective_match = match_config if (
            match_config.normalize_whitespace or match_config.ignore_case
            or match_config.numeric_tolerance is not None
        ) else None

        normalizer_specs = getattr(args, "normalizers", None) or []
        resolved_normalizers = None
        if normalizer_specs:
            from xpyd_acc.normalizers import resolve_normalizers
            resolved_normalizers = resolve_normalizers(normalizer_specs)

        batch_cache = None
        if not getattr(args, "no_cache", False):
            from xpyd_acc.cache import DEFAULT_CACHE_DIR, DEFAULT_TTL, ResponseCache
            cache_dir = getattr(args, "cache_dir", None) or DEFAULT_CACHE_DIR
            cache_ttl = getattr(args, "cache_ttl", None) or DEFAULT_TTL
            batch_cache = ResponseCache(cache_dir=cache_dir, ttl=cache_ttl)

        from xpyd_acc.headers import parse_env_headers, parse_header_args, resolve_headers
        cli_hdrs = parse_header_args(getattr(args, "headers", None))
        env_hdrs = parse_env_headers()
        cfg_hdrs = args._config.get("defaults", {}).get("headers") if args._config else None
        custom_headers = resolve_headers(
            cli_headers=cli_hdrs, env_headers=env_hdrs, config_headers=cfg_hdrs,
        ) or None

        is_multi = len(target_urls) > 1

        if is_multi:
            multi_report = await run_multi_batch(
                samples, args.baseline, target_urls,
                model=args.model, max_tokens=args.max_tokens, api_key=args.api_key,
                logprob_gap_threshold=args.logprob_gap_threshold,
                concurrency=args.concurrency, retries=args.retries,
                retry_delay=args.retry_delay,
                on_progress=on_progress if use_progress else None,
                match_config=effective_match, sampling_params=sampling,
                timeout=args.timeout,
                skip_validation=getattr(args, "skip_validation", False),
                custom_headers=custom_headers,
            )
            report = None
        else:
            multi_report = None
            from xpyd_acc.rate_limit import RateLimiter
            _rl = RateLimiter(getattr(args, "rate_limit", None))
            report = await run_batch(
                samples, args.baseline, target_urls[0],
                model=args.model, max_tokens=args.max_tokens, api_key=args.api_key,
                logprob_gap_threshold=args.logprob_gap_threshold,
                concurrency=args.concurrency, retries=args.retries,
                retry_delay=args.retry_delay,
                on_progress=on_progress if use_progress else None,
                match_config=effective_match, sampling_params=sampling,
                timeout=args.timeout,
                deduplicate=getattr(args, "deduplicate", False),
                enable_request_ids=not getattr(args, "no_request_id", False),
                cache=batch_cache, rate_limiter=_rl,
                normalizers=resolved_normalizers,
                skip_validation=getattr(args, "skip_validation", False),
                checkpoint_path=getattr(args, "checkpoint", None),
                checkpoint_clear=getattr(args, "checkpoint_clear", False),
                custom_headers=custom_headers,
            )
    finally:
        if progress_ctx is not None:
            progress_ctx.stop()

    if batch_cache is not None:
        cs = batch_cache.stats()
        if cs.hits + cs.misses > 0:
            print(f"\nCache: {cs.hits} hits, {cs.misses} misses ({cs.hit_rate:.0%} hit rate)")

    _apply_cost_to_report(args, report if report is not None else multi_report)

    if multi_report is not None:
        _print_multi_report(args, multi_report, target_urls, format_report, export_csv)
    else:
        await _print_single_report(args, report, format_report, export_csv, export_markdown)

def _print_multi_report(args, multi_report, target_urls, format_report, export_csv):
    """Print and export multi-target batch report."""
    for url in target_urls:
        print(f"\n--- Target: {url} ---")
        print(format_report(multi_report.per_target[url]))

    if len(target_urls) > 1:
        print("\n--- Cross-Target Agreement Matrix ---")
        header = f"{'':>30s}"
        for u in target_urls:
            header += f" {u[-20:]:>20s}"
        print(header)
        for u1 in target_urls:
            row = f"{u1[-30:]:>30s}"
            for u2 in target_urls:
                val = multi_report.agreement_matrix[u1][u2]
                row += f" {val:>19.1%}"
            print(row)

    if args.json_path:
        from pathlib import Path
        Path(args.json_path).write_text(multi_report.to_json())
        print(f"\nJSON exported to {args.json_path}")

    if args.markdown:
        from pathlib import Path
        Path(args.markdown).write_text(multi_report.to_markdown())
        print(f"\nMarkdown exported to {args.markdown}")

    if args.csv:
        first_report = multi_report.per_target[target_urls[0]]
        export_csv(first_report, args.csv)
        print(f"\nCSV exported to {args.csv} (first target)")

    if args.junit:
        from pathlib import Path

        from xpyd_acc.junit import multi_report_to_junit
        Path(args.junit).write_text(multi_report_to_junit(multi_report))
        print(f"\nJUnit XML exported to {args.junit}")

    fail_threshold = _resolve_fail_threshold(args, getattr(args, "_config", None))
    worst_rate = max(r.divergence_rate for r in multi_report.per_target.values())
    if fail_threshold is not None:
        if worst_rate > fail_threshold:
            print(
                f"\n✗ FAIL: worst divergence rate {worst_rate:.1%}"
                f" exceeds threshold {fail_threshold:.1%}",
            )
            sys.exit(1)
        else:
            print(
                f"\n✓ PASS: worst divergence rate {worst_rate:.1%}"
                f" within threshold {fail_threshold:.1%}",
            )
    elif any(r.divergent_samples > 0 for r in multi_report.per_target.values()):
        sys.exit(1)

async def _print_single_report(args, report, format_report, export_csv, export_markdown):
    """Print and export single-target batch report."""
    _maybe_apply_confidence(args, report)

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

    if args.junit:
        from pathlib import Path

        from xpyd_acc.junit import report_to_junit
        Path(args.junit).write_text(report_to_junit(report))
        print(f"\nJUnit XML exported to {args.junit}")

    await _maybe_send_webhook(args, report)
    _check_truncation_threshold(args, report)

    fail_threshold = _resolve_fail_threshold(args, getattr(args, "_config", None))
    if fail_threshold is not None:
        if getattr(args, "confidence", False) and report.divergence_ci_lower is not None:
            check_val = report.divergence_ci_lower
            label = f"CI lower bound {check_val:.1%}"
        else:
            check_val = report.divergence_rate
            label = f"divergence rate {check_val:.1%}"

        if check_val > fail_threshold:
            print(f"\n✗ FAIL: {label} exceeds threshold {fail_threshold:.1%}")
            sys.exit(1)
        else:
            print(f"\n✓ PASS: {label} within threshold {fail_threshold:.1%}")
    elif report.divergent_samples > 0:
        sys.exit(1)

async def _run_rerun(args: argparse.Namespace) -> None:
    """Run selective sample rerun from a previous report."""
    from pathlib import Path

    from xpyd_acc.batch_compare import export_csv, export_markdown, format_report, run_batch
    from xpyd_acc.rerun import load_divergent_samples, merge_rerun_results
    from xpyd_acc.sampling import SamplingParams

    sampling = SamplingParams.from_args(args)
    plan = load_divergent_samples(args.rerun)
    print(
        f"Rerun: {plan.divergent_count} divergent samples "
        f"out of {plan.total_in_report} total"
    )

    rerun_target = args.target[0] if isinstance(args.target, list) else args.target

    if not args.skip_healthcheck:
        await _preflight_healthcheck([args.baseline, rerun_target], api_key=args.api_key)

    if args.template:
        from xpyd_acc.templates import resolve_template
        template = resolve_template(args.template)
        print(f"Using template: {template.name}")
        for sample in plan.divergent_samples:
            variables = {"prompt": sample.prompt, **sample.metadata}
            sample.prompt = template.render(variables)

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
            TextColumn("[bold blue]{task.description}"), BarColumn(),
            MofNCompleteColumn(), TextColumn("•"),
            TimeElapsedColumn(), TextColumn("•"), TimeRemainingColumn(),
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
            match_config.normalize_whitespace or match_config.ignore_case
            or match_config.numeric_tolerance is not None
        ) else None

        report = await run_batch(
            plan.divergent_samples, args.baseline, rerun_target,
            model=args.model, max_tokens=args.max_tokens, api_key=args.api_key,
            logprob_gap_threshold=args.logprob_gap_threshold,
            concurrency=args.concurrency, retries=args.retries,
            retry_delay=args.retry_delay,
            on_progress=on_progress if use_progress else None,
            match_config=effective_match, sampling_params=sampling,
            timeout=args.timeout,
            deduplicate=getattr(args, "deduplicate", False),
            enable_request_ids=not getattr(args, "no_request_id", False),
            skip_validation=getattr(args, "skip_validation", False),
        )
    finally:
        if progress_ctx is not None:
            progress_ctx.stop()

    if args.rerun_merge:
        report = merge_rerun_results(args.rerun, report)
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

    fail_threshold = _resolve_fail_threshold(args, getattr(args, "_config", None))
    if fail_threshold is not None:
        if report.divergence_rate > fail_threshold:
            print(
                f"\n✗ FAIL: divergence rate {report.divergence_rate:.1%}"
                f" exceeds threshold {fail_threshold:.1%}",
            )
            sys.exit(1)
        else:
            print(
                f"\n✓ PASS: divergence rate {report.divergence_rate:.1%}"
                f" within threshold {fail_threshold:.1%}",
            )
    elif report.divergent_samples > 0:
        sys.exit(1)

