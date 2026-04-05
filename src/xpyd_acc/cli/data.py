"""CLI handlers for snapshot, cache, history, dataset-stats."""

from __future__ import annotations

import argparse
import sys

from ._common import _preflight_healthcheck


async def _run_batch_with_snapshot(args: argparse.Namespace) -> None:
    """Run batch comparison using a saved snapshot as baseline."""
    import asyncio as _asyncio

    from xpyd_acc.batch_compare import (
        DatasetSample,
        SampleResult,
        _collect_output,
        _find_first_divergence,
        _tokenize,
        classify_divergence,
        compute_report,
        export_csv,
        export_markdown,
        format_report,
        load_dataset,
    )
    from xpyd_acc.output_compare import MatchConfig, normalized_match
    from xpyd_acc.sampling import SamplingParams
    from xpyd_acc.snapshot import load_snapshot, validate_snapshot_dataset

    sampling = SamplingParams.from_args(args)
    snapshot = load_snapshot(args.snapshot)
    samples = load_dataset(args.dataset)
    validate_snapshot_dataset(snapshot, samples)

    if args.template:
        from xpyd_acc.templates import resolve_template
        template = resolve_template(args.template)
        print(f"Using template: {template.name}")
        for sample in samples:
            variables = {"prompt": sample.prompt, **sample.metadata}
            sample.prompt = template.render(variables)

    print(f"Using snapshot from {snapshot.captured_at} ({len(snapshot.samples)} samples)")
    print(f"Target: {args.target}")

    if not args.skip_healthcheck:
        await _preflight_healthcheck([args.target], api_key=args.api_key)

    snap_by_id = {s.sample_id: s for s in snapshot.samples}

    match_config = MatchConfig(
        normalize_whitespace=args.normalize_whitespace,
        ignore_case=args.ignore_case,
        numeric_tolerance=args.numeric_tolerance,
    )
    effective_match = match_config if (
        match_config.normalize_whitespace or match_config.ignore_case
        or match_config.numeric_tolerance is not None
    ) else None

    semaphore = _asyncio.Semaphore(args.concurrency or 5)
    completed = 0
    total = len(samples)

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

    def on_progress_update(done: int, _total: int) -> None:
        if progress_ctx is not None and task_id is not None:
            progress_ctx.update(task_id, completed=done)

    async def process_one(sample: DatasetSample) -> SampleResult:
        nonlocal completed
        snap_sample = snap_by_id[sample.id]
        baseline_text = snap_sample.output
        baseline_lp = snap_sample.logprobs

        async with semaphore:
            target_text, target_lp, _rid = await _collect_output(
                args.target, sample.prompt,
                model=args.model or snapshot.model,
                max_tokens=args.max_tokens or 64,
                api_key=args.api_key or "no-key",
                retries=args.retries or 3,
                retry_delay=args.retry_delay or 1.0,
                sampling_params=sampling,
                timeout=args.timeout or 120.0,
            )

        b_tokens = _tokenize(baseline_text)
        t_tokens = _tokenize(target_text)
        exact = normalized_match(baseline_text, target_text, effective_match)
        div_idx = _find_first_divergence(b_tokens, t_tokens)

        b_lp_at_div = None
        t_lp_at_div = None
        gap = None

        if div_idx is not None and div_idx < len(target_lp):
            lp_entry = target_lp[div_idx]
            t_lp_at_div = lp_entry.get("logprob")
            top_lps = lp_entry.get("top_logprobs", [])
            if len(top_lps) >= 2:
                gap = abs(top_lps[0].get("logprob", 0) - top_lps[1].get("logprob", 0))
        if div_idx is not None and div_idx < len(baseline_lp):
            b_lp_at_div = baseline_lp[div_idx].get("logprob")

        threshold = args.logprob_gap_threshold or 0.1
        classification = "match" if exact else classify_divergence(gap, threshold=threshold)
        ctx_len = len(_tokenize(sample.prompt))

        completed += 1
        on_progress_update(completed, total)

        return SampleResult(
            sample_id=sample.id, prompt=sample.prompt,
            baseline_output=baseline_text, target_output=target_text,
            exact_match=exact, first_divergence_index=div_idx,
            baseline_logprob_at_divergence=b_lp_at_div,
            target_logprob_at_divergence=t_lp_at_div,
            logprob_gap=gap, classification=classification,
            context_length=ctx_len,
        )

    if progress_ctx is not None:
        progress_ctx.start()
        task_id = progress_ctx.add_task("Comparing samples", total=total)

    try:
        tasks = [process_one(s) for s in samples]
        results = list(await _asyncio.gather(*tasks))
    finally:
        if progress_ctx is not None:
            progress_ctx.stop()

    report = compute_report(results, logprob_gap_threshold=args.logprob_gap_threshold or 0.1)

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


async def _run_snapshot(args: argparse.Namespace) -> None:
    """Run snapshot capture."""
    from xpyd_acc.batch_compare import load_dataset
    from xpyd_acc.config import load_config
    from xpyd_acc.sampling import SamplingParams
    from xpyd_acc.snapshot import capture_snapshot, save_snapshot

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {}) if cfg else {}

    sampling = SamplingParams.from_args(args)
    samples = load_dataset(args.dataset)

    if args.template:
        from xpyd_acc.templates import resolve_template
        template = resolve_template(args.template)
        print(f"Using template: {template.name}")
        for sample in samples:
            variables = {"prompt": sample.prompt, **sample.metadata}
            sample.prompt = template.render(variables)

    print(f"Capturing snapshot: {len(samples)} samples from {args.baseline}")

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
        task_id = progress_ctx.add_task("Capturing snapshot", total=len(samples))

    try:
        snap = await capture_snapshot(
            samples, args.baseline,
            model=args.model or defaults.get("model", "default"),
            max_tokens=args.max_tokens or defaults.get("max_tokens", 64),
            api_key=args.api_key or defaults.get("api_key", "no-key"),
            concurrency=args.concurrency or defaults.get("concurrency", 5),
            retries=args.retries or defaults.get("retries", 3),
            retry_delay=args.retry_delay or defaults.get("retry_delay", 1.0),
            sampling_params=sampling,
            timeout=args.timeout or defaults.get("timeout", 120.0),
            on_progress=on_progress if use_progress else None,
        )
    finally:
        if progress_ctx is not None:
            progress_ctx.stop()

    save_snapshot(snap, args.output)
    print(f"\nSnapshot saved to {args.output} ({len(snap.samples)} samples)")


def _run_cache(args: argparse.Namespace) -> None:
    """Manage response cache."""
    from xpyd_acc.cache import DEFAULT_CACHE_DIR, DEFAULT_TTL, ResponseCache

    cache_dir = getattr(args, "cache_dir", None) or DEFAULT_CACHE_DIR
    cache = ResponseCache(cache_dir=cache_dir, ttl=DEFAULT_TTL)

    action = getattr(args, "cache_action", None)
    if action == "clear":
        count = cache.clear()
        print(f"Cleared {count} cached entries from {cache_dir}")
    elif action == "stats":
        stats = cache.stats()
        print(f"Cache directory: {cache_dir}")
        print(f"Entries: {stats.entry_count}")
        print(f"Total size: {stats.total_size_bytes:,} bytes")
    else:
        print("Usage: xpyd-acc cache {clear|stats}")


def _run_history(args: argparse.Namespace) -> None:
    """Handle history subcommands."""
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    from xpyd_acc.history import HistoryStore

    history_dir = Path(args.history_dir) if getattr(args, "history_dir", None) else None
    store = HistoryStore(history_dir=history_dir)

    if args.history_action == "save":
        entry = store.save(report_path=args.report, tag=args.tag)
        print(f"Saved: {entry.entry_id} | rate={entry.divergence_rate:.4f} | "
              f"samples={entry.sample_count} | tag={entry.tag or '(none)'}")

    elif args.history_action == "list":
        entries = store.list_entries()
        if not entries:
            print("No history entries found.")
            return
        console = Console()
        table = Table(title="History Entries")
        table.add_column("ID", style="dim")
        table.add_column("Timestamp")
        table.add_column("Tag")
        table.add_column("Divergence Rate", justify="right")
        table.add_column("Samples", justify="right")
        for e in entries:
            table.add_row(
                e.entry_id, e.timestamp[:19], e.tag or "-",
                f"{e.divergence_rate:.4f}", str(e.sample_count),
            )
        console.print(table)

    elif args.history_action == "trend":
        trend_data = store.trend(last_n=args.last)
        if not trend_data:
            print("No history entries for trend analysis.")
            return
        console = Console()
        table = Table(title="Divergence Rate Trend")
        table.add_column("Timestamp")
        table.add_column("Tag")
        table.add_column("Rate", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Samples", justify="right")
        for t in trend_data:
            delta_str = f"{t['delta']:+.4f}" if t['delta'] != 0 else "-"
            style = "red" if t['delta'] > 0 else ("green" if t['delta'] < 0 else "")
            table.add_row(
                t["timestamp"][:19], t["tag"] or "-",
                f"{t['divergence_rate']:.4f}", delta_str,
                str(t["sample_count"]), style=style,
            )
        console.print(table)

        if args.fail_on_regression and store.has_regression(last_n=args.last):
            print("\nFAIL: Divergence rate increased in the latest run.")
            sys.exit(1)
        elif args.fail_on_regression:
            print("\nPASS: No regression detected.")

    elif args.history_action == "purge":
        removed = store.purge(
            older_than_days=args.older_than,
            keep_last=args.keep_last,
            dry_run=args.dry_run,
        )
        action_label = "Would remove" if args.dry_run else "Removed"
        if not removed:
            print("Nothing to purge.")
        else:
            console = Console()
            table = Table(title=f"{action_label} {len(removed)} entries")
            table.add_column("ID", style="dim")
            table.add_column("Timestamp")
            table.add_column("Tag")
            table.add_column("Rate", justify="right")
            for e in removed:
                table.add_row(
                    e.entry_id, e.timestamp[:19], e.tag or "-",
                    f"{e.divergence_rate:.4f}",
                )
            console.print(table)

    else:
        print("Usage: xpyd-acc history {save|list|trend|purge}")


def _run_dataset_stats(args: argparse.Namespace) -> None:
    """Run dataset statistics analysis."""
    from xpyd_acc.batch_compare import load_dataset
    from xpyd_acc.dataset_stats import compute_stats, print_stats
    from xpyd_acc.templates import load_template

    samples = load_dataset(args.dataset)
    template = load_template(args.template) if args.template else None
    report = compute_stats(samples, template)
    print_stats(report)
    if args.json_path:
        report.to_json(args.json_path)
        print(f"\nExported to {args.json_path}")
