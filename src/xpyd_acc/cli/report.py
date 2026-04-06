"""CLI handlers for report, diff, regression, summary, filter,
explain, annotate, cluster, aggregate.
"""

from __future__ import annotations

import argparse
import json
import sys


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


def _run_diff(args: argparse.Namespace) -> None:
    """Run side-by-side diff of two batch reports."""
    from xpyd_acc.diff import diff_reports, format_diff_report

    result = diff_reports(args.old, args.new_report)
    print(format_diff_report(result))

    if getattr(args, "json_path", None):
        from pathlib import Path
        Path(args.json_path).write_text(result.to_json())
        print(f"\nDiff result exported to {args.json_path}")

    sys.exit(1 if result.regressions > 0 else 0)


def _run_summary(args: argparse.Namespace) -> None:
    """Run the summary subcommand."""
    from pathlib import Path

    from xpyd_acc.summary import load_and_summarize

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"Error: report file not found: {report_path}", file=sys.stderr)
        sys.exit(1)
    try:
        output = load_and_summarize(report_path, args.summary_format)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error: failed to read report: {exc}", file=sys.stderr)
        sys.exit(1)
    print(output)


def _run_filter(args: argparse.Namespace) -> None:
    """Filter samples from a batch report."""
    from xpyd_acc.annotate import AnnotationStore
    from xpyd_acc.filter import FilterConfig, filter_samples, load_report, save_report

    report = load_report(args.input)
    config = FilterConfig(
        classification=args.classification,
        divergent_only=args.divergent_only,
        matched_only=args.matched_only,
        min_logprob_gap=args.min_logprob_gap,
        max_logprob_gap=args.max_logprob_gap,
        min_context_length=args.min_context_length,
        max_context_length=args.max_context_length,
        search=args.search,
    )
    filtered = filter_samples(report, config)

    ann_label = getattr(args, "annotation_label", None)
    annotated_only = getattr(args, "annotated", False)
    unannotated_only = getattr(args, "unannotated", False)

    if ann_label or annotated_only or unannotated_only:
        store = AnnotationStore.load(args.input)
        results = filtered.get("results", [])
        kept: list[dict] = []
        for r in results:
            sid = r.get("sample_id", "")
            ann = store.get(sid)
            has_ann = ann is not None and not ann.is_empty()

            if annotated_only and not has_ann:
                continue
            if unannotated_only and has_ann:
                continue
            if ann_label:
                if ann is None or ann_label not in ann.labels:
                    continue
            kept.append(r)

        total = len(kept)
        divergent = sum(1 for r in kept if not r.get("exact_match", True))
        filtered["results"] = kept
        filtered["total_samples"] = total
        filtered["divergent_samples"] = divergent
        filtered["match_samples"] = total - divergent
        filtered["divergence_rate"] = divergent / total if total else 0.0

    save_report(filtered, args.output)

    total = filtered["total_samples"]
    divergent = filtered["divergent_samples"]
    rate = filtered["divergence_rate"]
    print(f"\nFiltered report: {total} samples, {divergent} divergent ({rate:.1%})")
    print(f"Saved to {args.output}")


def _run_explain(args: argparse.Namespace) -> None:
    """Handle the 'explain' subcommand."""
    from xpyd_acc.explain import format_explain, load_and_explain

    try:
        result = load_and_explain(args.report, args.sample)
    except FileNotFoundError:
        print(f"Error: report file not found: {args.report}")
        raise SystemExit(1)
    except KeyError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    if args.explain_json:
        from pathlib import Path
        Path(args.explain_json).write_text(result.to_json())
        print(f"Exported to {args.explain_json}")
    else:
        print(format_explain(result))


def _run_annotate(args: argparse.Namespace) -> None:
    """Handle the 'annotate' subcommand."""
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    from xpyd_acc.annotate import AnnotationStore

    console = Console()
    report_path = Path(args.report)
    if not report_path.exists():
        console.print(f"[red]Report not found:[/red] {report_path}")
        raise SystemExit(1)

    store = AnnotationStore.load(report_path)

    if args.list_annotations:
        ids = store.list_annotated_ids()
        if not ids:
            console.print("No annotations found.")
            return
        table = Table(title="Annotations")
        table.add_column("Sample ID")
        table.add_column("Labels")
        table.add_column("Note")
        for sid in ids:
            ann = store.get(sid)
            if ann is None:
                continue
            table.add_row(sid, ", ".join(ann.labels), ann.note or "")
        console.print(table)
        return

    if not args.sample:
        console.print("[red]--sample is required for add/clear operations[/red]")
        raise SystemExit(1)

    if args.clear:
        removed = store.clear(args.sample)
        store.save(report_path)
        if removed:
            console.print(f"Cleared annotations for sample [bold]{args.sample}[/bold]")
        else:
            console.print(f"No annotations found for sample [bold]{args.sample}[/bold]")
        return

    if args.note is None and args.label is None:
        console.print("[red]Provide --note and/or --label[/red]")
        raise SystemExit(1)

    if args.note is not None:
        store.set_note(args.sample, args.note)
    if args.label is not None:
        store.add_label(args.sample, args.label)
    store.save(report_path)
    console.print(f"Annotated sample [bold]{args.sample}[/bold]")


def _run_cluster(args: argparse.Namespace) -> None:
    """Run the cluster subcommand."""
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    from xpyd_acc.cluster import cluster_divergences

    console = Console()
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {input_path}")
        raise SystemExit(1)

    with open(input_path) as f:
        report = json.load(f)

    result = cluster_divergences(report, k=args.clusters)

    if result.total_divergent == 0:
        console.print("[green]No divergent samples to cluster.[/green]")
        return

    console.print("\n[bold]Divergence Pattern Clustering[/bold]")
    console.print(f"  Total divergent samples: {result.total_divergent}")
    console.print(f"  Excluded matched samples: {result.excluded_matched}")
    console.print(f"  Clusters (K): {result.k}")
    if result.silhouette_score is not None:
        console.print(f"  Silhouette score: {result.silhouette_score:.3f}")

    table = Table(title="Clusters")
    table.add_column("ID", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Avg Div Index", style="yellow")
    table.add_column("Avg Logprob Gap", style="yellow")
    table.add_column("Avg Context Len", style="yellow")
    table.add_column("Representative", style="magenta")

    for c in result.clusters:
        table.add_row(
            str(c.cluster_id), str(c.size),
            f"{c.avg_divergence_index:.1f}", f"{c.avg_logprob_gap:.4f}",
            f"{c.avg_context_length:.0f}", c.representative_sample_id,
        )

    console.print(table)

    if args.cluster_json:
        result.to_json(args.cluster_json)
        console.print(f"\n  Exported to {args.cluster_json}")


def _run_aggregate(args: argparse.Namespace) -> None:
    """Aggregate multiple batch run reports."""
    from pathlib import Path

    from xpyd_acc.aggregate import (
        aggregate_reports,
        format_aggregated_report,
        load_batch_report_from_json,
    )

    reports = [load_batch_report_from_json(p) for p in args.reports]
    agg_report = aggregate_reports(reports)
    print(format_aggregated_report(agg_report))

    if getattr(args, "json_path", None):
        Path(args.json_path).write_text(agg_report.to_json())
        print(f"\nAggregated report exported to {args.json_path}")


def _run_ab_test(args: argparse.Namespace) -> None:
    """Run A/B test comparing divergence rates from two batch reports."""
    import json as json_mod
    from pathlib import Path

    from xpyd_acc.ab_test import format_ab_test, run_ab_test

    report_a = json_mod.loads(Path(args.report_a).read_text(encoding="utf-8"))
    report_b = json_mod.loads(Path(args.report_b).read_text(encoding="utf-8"))

    result = run_ab_test(
        report_a_total=report_a["total_samples"],
        report_a_divergent=report_a["divergent_samples"],
        report_b_total=report_b["total_samples"],
        report_b_divergent=report_b["divergent_samples"],
        alpha=args.alpha,
    )
    print(format_ab_test(result))

    if getattr(args, "json_path", None):
        result.save_json(args.json_path)
        print(f"\nA/B test result exported to {args.json_path}")

    sys.exit(1 if result.significant else 0)


def _run_prometheus(args: argparse.Namespace) -> None:
    """Export batch report as Prometheus text exposition format."""
    from pathlib import Path

    from xpyd_acc.batch_compare import load_report
    from xpyd_acc.prometheus import push_to_gateway, to_prometheus

    report = load_report(args.report)
    metrics = to_prometheus(report, model=args.model, dataset=args.dataset)

    if args.output:
        Path(args.output).write_text(metrics, encoding="utf-8")
        print(f"Prometheus metrics written to {args.output}")
    else:
        print(metrics)

    if args.push_gateway:
        push_to_gateway(metrics, args.push_gateway, job=args.job)
        print(f"Metrics pushed to {args.push_gateway}")


def _run_grafana_dashboard(args: argparse.Namespace) -> None:
    """Generate Grafana dashboard JSON template."""
    from pathlib import Path

    from xpyd_acc.batch_compare import load_report
    from xpyd_acc.grafana import generate_dashboard

    report = None
    if args.report:
        report = load_report(args.report)

    dashboard = generate_dashboard(
        report,
        title=args.title,
        datasource=args.datasource,
    )

    Path(args.output).write_text(dashboard.to_json(), encoding="utf-8")
    print(f"Grafana dashboard written to {args.output}")


def _run_auto_threshold(args: argparse.Namespace) -> None:
    """Analyze historical reports and recommend thresholds."""
    import json
    from pathlib import Path

    from xpyd_acc.auto_threshold import (
        analyze_thresholds,
        format_recommendations,
        load_reports,
    )

    reports = load_reports(args.reports)
    rec = analyze_thresholds(reports, percentile_level=args.percentile)

    print(format_recommendations(rec))

    if args.json:
        Path(args.json).write_text(
            json.dumps(rec.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"\nRecommendations exported to {args.json}")
