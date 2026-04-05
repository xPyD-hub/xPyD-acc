"""CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


def _get_version() -> str:
    """Return the package version."""
    try:
        from importlib.metadata import version
        return version("xpyd-acc")
    except Exception:
        return "unknown"


def _add_sampling_args(parser: argparse.ArgumentParser) -> None:
    """Add --temperature, --top-p, --seed flags to a subcommand parser."""
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Named profile/preset to activate (e.g., greedy, stochastic)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (0 for greedy/deterministic)",
    )
    parser.add_argument(
        "--top-p", type=float, default=None, dest="top_p",
        help="Nucleus sampling top-p value",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible generation",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="xpyd-acc",
        description="PD disaggregation accuracy diagnostic tool",
    )
    parser.add_argument(
        "-V", "--version", action="version",
        version=f"%(prog)s {_get_version()}",
    )
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    verbosity_group.add_argument(
        "-q", "--quiet", action="store_true", default=False,
        help="Quiet mode (ERROR level only)",
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
    lp.add_argument("--retries", type=int, default=None, help="Max retry attempts (default: 3)")
    lp.add_argument(
        "--retry-delay", type=float, default=None,
        help="Base retry delay in seconds (default: 1.0)",
    )
    _add_sampling_args(lp)
    lp.add_argument(
        "--top-k", type=int, default=None, dest="top_k",
        help="Number of top logprobs to collect per position (default: 5)",
    )
    lp.add_argument(
        "--kl-threshold", type=float, default=None, dest="kl_threshold",
        help="KL divergence threshold for flagging positions (default: 0.1)",
    )

    diag = sub.add_parser("diagnose", help="Run full diagnostic pipeline")
    diag.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    diag.add_argument("--target", required=True, help="Target endpoint URL")
    diag.add_argument("--prompt", required=True, help="Prompt to send")
    diag.add_argument("--model", default=None, help="Model name (default: default)")
    diag.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    diag.add_argument("--api-key", default=None, help="API key for endpoints")
    diag.add_argument("--retries", type=int, default=None, help="Max retry attempts (default: 3)")
    diag.add_argument(
        "--retry-delay", type=float, default=None,
        help="Base retry delay in seconds (default: 1.0)",
    )
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
    _add_sampling_args(diag)

    oc = sub.add_parser("compare-output", help="Compare text outputs from two endpoints")
    oc_input = oc.add_mutually_exclusive_group(required=True)
    oc_input.add_argument("--baseline-text", help="Baseline output text (inline)")
    oc_input.add_argument("--baseline-file", help="Path to file with baseline output")
    oc.add_argument("--target-text", help="Target output text (inline)")
    oc.add_argument("--target-file", help="Path to file with target output")

    bc = sub.add_parser("batch-compare", help="Run batch dataset comparison")
    bc.add_argument("--baseline", default=None, help="Baseline endpoint URL")
    bc.add_argument(
        "--target", required=True, action="append",
        help="Target endpoint URL (can be specified multiple times)",
    )
    bc.add_argument(
        "--snapshot", default=None, metavar="SNAPSHOT_JSON",
        help="Use saved snapshot as baseline (mutually exclusive with --baseline)",
    )
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
    bc.add_argument("--json", default=None, dest="json_path", help="Path to export JSON results")
    bc.add_argument("--markdown", default=None, help="Path to export Markdown report")
    bc.add_argument("--junit", default=None, help="Path to export JUnit XML results")
    bc.add_argument("--retries", type=int, default=None, help="Max retry attempts (default: 3)")
    bc.add_argument(
        "--retry-delay", type=float, default=None,
        help="Base retry delay in seconds (default: 1.0)",
    )
    bc.add_argument(
        "--no-progress", action="store_true", default=False,
        help="Disable progress bar during batch comparison",
    )
    bc.add_argument(
        "--skip-healthcheck", action="store_true", default=False,
        help="Skip pre-flight endpoint health check",
    )
    bc.add_argument(
        "--skip-validation", action="store_true", default=False,
        help="Skip response schema validation",
    )
    bc.add_argument(
        "--template", default=None,
        help="Prompt template: built-in name (gsm8k, mmlu, etc.) or path to YAML/TOML file",
    )
    bc.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Validate setup without sending API requests",
    )
    bc.add_argument(
        "--normalize-whitespace", action="store_true", default=False,
        help="Collapse and strip whitespace before comparison",
    )
    bc.add_argument(
        "--ignore-case", action="store_true", default=False,
        help="Case-insensitive text matching",
    )
    bc.add_argument(
        "--numeric-tolerance", type=float, default=None,
        help="Treat numbers within tolerance as equal",
    )
    bc.add_argument(
        "--rerun", default=None, metavar="REPORT_JSON",
        help="Rerun only divergent samples from a previous JSON report",
    )
    bc.add_argument(
        "--rerun-merge", action="store_true", default=False,
        help="Merge rerun results back into the original report file",
    )
    bc.add_argument(
        "--timeout", type=float, default=None,
        help="HTTP request timeout in seconds (default: 120.0)",
    )
    bc.add_argument(
        "--fail-threshold", type=float, default=None,
        help="Fail (exit 1) if divergence rate exceeds this threshold (0.0–1.0)",
    )
    bc.add_argument(
        "--confidence", action="store_true", default=False,
        help="Compute Wilson score confidence interval for divergence rate",
    )
    bc.add_argument(
        "--confidence-level", type=float, default=0.95,
        help="Confidence level for CI (default: 0.95)",
    )
    _add_sampling_args(bc)
    bc.add_argument(
        "--no-request-id", action="store_true", default=False,
        help="Disable X-Request-ID headers on API requests",
    )
    bc.add_argument(
        "--deduplicate", action="store_true", default=False,
        help="Send each unique prompt only once per endpoint, reuse results for duplicates",
    )
    bc.add_argument(
        "--normalizer", action="append", default=None, dest="normalizers",
        help="Output normalizer (repeatable). Built-in: strip_thinking_tags, "
             "normalize_json, normalize_numbers. Or module:function for custom.",
    )
    bc.add_argument(
        "--rate-limit", type=float, default=None,
        help="Max requests per second to each endpoint",
    )
    bc.add_argument(
        "--cache-dir", default=None,
        help="Directory for response cache (default: .xpyd-acc-cache)",
    )
    bc.add_argument(
        "--no-cache", action="store_true", default=False,
        help="Disable response caching entirely",
    )
    bc.add_argument(
        "--warn-truncated", type=float, default=None,
        help="Warn (exit 2) if truncated sample ratio exceeds this threshold (0.0–1.0)",
    )
    bc.add_argument(
        "--cache-ttl", type=float, default=None,
        help="Cache entry TTL in seconds (default: 3600)",
    )
    bc.add_argument(
        "--webhook", default=None, metavar="URL",
        help="Webhook URL to POST divergence alerts to",
    )
    bc.add_argument(
        "--webhook-header", action="append", default=None, dest="webhook_headers",
        help="Custom webhook header as 'Key: Value' (repeatable)",
    )
    bc.add_argument(
        "--webhook-always", action="store_true", default=False,
        help="Send webhook on every run, not just on divergence",
    )
    bc.add_argument(
        "--input-price", type=float, default=None,
        help="Input token price per 1M tokens (USD) for cost estimation",
    )
    bc.add_argument(
        "--output-price", type=float, default=None,
        help="Output token price per 1M tokens (USD) for cost estimation",
    )
    bc.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Checkpoint file path for resumable runs (saves progress, resumes on restart)",
    )
    bc.add_argument(
        "--checkpoint-clear", action="store_true", default=False,
        help="Delete existing checkpoint file before starting (fresh run)",
    )

    # Cache management subcommand
    cache_cmd = sub.add_parser("cache", help="Manage response cache")
    cache_sub = cache_cmd.add_subparsers(dest="cache_action")
    cache_clear = cache_sub.add_parser("clear", help="Remove all cached responses")
    cache_clear.add_argument(
        "--cache-dir", default=None,
        help="Cache directory (default: .xpyd-acc-cache)",
    )
    cache_stats = cache_sub.add_parser("stats", help="Show cache statistics")
    cache_stats.add_argument(
        "--cache-dir", default=None,
        help="Cache directory (default: .xpyd-acc-cache)",
    )

    rp = sub.add_parser("report", help="Generate HTML report from batch comparison JSON")
    rp.add_argument("--input", required=True, help="Path to batch results JSON file")
    rp.add_argument("--output", default=None, help="Output HTML file path (default: report.html)")

    cs = sub.add_parser("compare-streaming", help="Compare SSE streaming outputs")
    cs.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    cs.add_argument("--target", required=True, help="Target endpoint URL")
    cs.add_argument("--prompt", required=True, help="Prompt to send")
    cs.add_argument("--model", default=None, help="Model name (default: default)")
    cs.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    cs.add_argument("--api-key", default=None, help="API key for both endpoints")
    cs.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    cs.add_argument(
        "--skip-healthcheck", action="store_true", default=False,
        help="Skip pre-flight endpoint health check",
    )
    cs.add_argument(
        "--timing", action="store_true", default=False,
        help="Enable token timing analysis (TTFT, inter-token latency)",
    )
    cs.add_argument(
        "--normalize-whitespace", action="store_true", default=False,
        help="Collapse and strip whitespace before comparison",
    )
    cs.add_argument(
        "--ignore-case", action="store_true", default=False,
        help="Case-insensitive text matching",
    )
    cs.add_argument(
        "--numeric-tolerance", type=float, default=None,
        help="Treat numbers within tolerance as equal",
    )
    _add_sampling_args(cs)

    hc = sub.add_parser("healthcheck", help="Check endpoint health")
    hc.add_argument("url", nargs="+", help="Endpoint URL(s) to check")
    hc.add_argument("--api-key", default=None, help="API key for endpoints")
    hc.add_argument("--timeout", type=float, default=10.0, help="Timeout per endpoint in seconds")

    det = sub.add_parser("detect", help="Detect xPyD endpoint type")
    det.add_argument("url", help="Endpoint URL to probe")
    det.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")

    reg = sub.add_parser("regression", help="Detect regressions between two batch runs")
    reg.add_argument("--baseline", required=True, help="Path to baseline batch result JSON")
    reg.add_argument("--current", required=True, help="Path to current batch result JSON")
    reg.add_argument(
        "--json", dest="json_path", default=None,
        help="Export regression report as JSON",
    )

    # diff
    diff_p = sub.add_parser("diff", help="Side-by-side comparison of two batch reports")
    diff_p.add_argument("--old", required=True, help="Path to old batch report JSON")
    diff_p.add_argument(
        "--new", required=True, dest="new_report",
        help="Path to new batch report JSON",
    )
    diff_p.add_argument(
        "--json", dest="json_path", default=None,
        help="Export diff result as JSON",
    )

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

    # aggregate
    agg = sub.add_parser("aggregate", help="Aggregate multiple batch run reports")
    agg.add_argument(
        "--reports", nargs="+", required=True,
        help="Paths to batch comparison JSON report files",
    )
    agg.add_argument(
        "--json", default=None, dest="json_path",
        help="Export aggregated report as JSON to this path",
    )

    # watch
    wp = sub.add_parser("watch", help="Continuous divergence monitoring")
    wp.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    wp.add_argument("--target", required=True, help="Target endpoint URL")
    wp.add_argument("--prompt", required=True, help="Prompt to compare")
    wp.add_argument("--model", default=None, help="Model name")
    wp.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    wp.add_argument("--api-key", default=None, help="API key for endpoints")
    wp.add_argument(
        "--interval", type=float, default=60.0,
        help="Seconds between iterations (default: 60)",
    )
    wp.add_argument(
        "--max-iterations", type=int, default=None,
        help="Stop after N iterations (default: unlimited)",
    )
    wp.add_argument(
        "--alert-threshold", type=int, default=None,
        help="Exit code 1 after N consecutive failures",
    )
    wp.add_argument("--log", default=None, dest="log_path", help="JSON log file path")
    wp.add_argument("--retries", type=int, default=None, help="Max retry attempts")
    wp.add_argument("--retry-delay", type=float, default=None, help="Base retry delay (seconds)")
    wp.add_argument("--skip-healthcheck", action="store_true", help="Skip pre-flight healthcheck")
    _add_sampling_args(wp)

    sub.add_parser("profiles", help="List available named profiles")

    # shell completion
    comp = sub.add_parser("completion", help="Generate shell completion script")
    comp.add_argument("shell", choices=["bash", "zsh", "fish"], help="Target shell")
    comp.add_argument(
        "--output", default=None,
        help="Write completion script to file instead of stdout",
    )

    # bisect: auto-bisect divergence by context length
    bis = sub.add_parser("bisect", help="Binary search for min context length causing divergence")
    bis.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    bis.add_argument("--target", required=True, help="Target endpoint URL")
    bis.add_argument("--prompt", required=True, help="Full prompt text to bisect")
    bis.add_argument("--model", default="default", help="Model name")
    bis.add_argument("--api-key", default=None, help="API key for endpoints")
    bis.add_argument("--min-length", type=int, default=None, help="Minimum prefix length")
    bis.add_argument("--max-length", type=int, default=None, help="Maximum prefix length")
    bis.add_argument("--retries", type=int, default=None, help="Max retry attempts (default: 3)")
    bis.add_argument(
        "--retry-delay", type=float, default=None,
        help="Base retry delay in seconds (default: 1.0)",
    )
    bis.add_argument(
        "--timeout", type=float, default=None,
        help="HTTP request timeout in seconds (default: 120.0)",
    )
    bis.add_argument("--json", default=None, metavar="PATH", help="Export result as JSON")
    _add_sampling_args(bis)

    # snapshot capture
    sc = sub.add_parser("snapshot", help="Capture baseline outputs as a snapshot")
    sc.add_argument("action", choices=["capture"], help="Snapshot action")
    sc.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    sc.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    sc.add_argument("--output", required=True, help="Output snapshot JSON path")
    sc.add_argument("--model", default=None, help="Model name (default: default)")
    sc.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    sc.add_argument("--api-key", default=None, help="API key for endpoint")
    sc.add_argument(
        "--concurrency", type=int, default=None,
        help="Max concurrent requests (default: 5)",
    )
    sc.add_argument("--retries", type=int, default=None, help="Max retry attempts (default: 3)")
    sc.add_argument(
        "--retry-delay", type=float, default=None,
        help="Base retry delay in seconds (default: 1.0)",
    )
    sc.add_argument(
        "--timeout", type=float, default=None,
        help="HTTP request timeout in seconds (default: 120.0)",
    )
    sc.add_argument(
        "--rate-limit", type=float, default=None,
        help="Max requests per second to the endpoint",
    )
    sc.add_argument(
        "--template", default=None,
        help="Prompt template: built-in name or path to YAML/TOML file",
    )
    sc.add_argument(
        "--no-progress", action="store_true", default=False,
        help="Disable progress bar",
    )
    _add_sampling_args(sc)

    # history
    hist = sub.add_parser("history", help="Result history & trend tracking")
    hist_sub = hist.add_subparsers(dest="history_action")
    hist_save = hist_sub.add_parser("save", help="Save a batch report to history")
    hist_save.add_argument("--report", required=True, help="Path to batch report JSON")
    hist_save.add_argument("--tag", default="", help="Label for this run")
    hist_save.add_argument("--history-dir", default=None, help="History directory")
    hist_list = hist_sub.add_parser("list", help="List saved history entries")
    hist_list.add_argument("--history-dir", default=None, help="History directory")
    hist_trend = hist_sub.add_parser("trend", help="Show divergence rate trend")
    hist_trend.add_argument("--last", type=int, default=None, help="Show last N entries")
    hist_trend.add_argument(
        "--fail-on-regression", action="store_true", default=False,
        help="Exit 1 if divergence rate increased in the latest run",
    )
    hist_trend.add_argument("--history-dir", default=None, help="History directory")
    hist_purge = hist_sub.add_parser("purge", help="Remove old history entries")
    hist_purge.add_argument(
        "--older-than", type=int, required=True,
        help="Remove entries older than N days",
    )
    hist_purge.add_argument(
        "--keep-last", type=int, default=0,
        help="Always keep the most recent N entries (default: 0)",
    )
    hist_purge.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Show what would be removed without deleting",
    )
    hist_purge.add_argument("--history-dir", default=None, help="History directory")

    # benchmark
    bm = sub.add_parser("benchmark", help="Benchmark endpoint latency")
    bm.add_argument("url", help="Endpoint URL to benchmark")
    bm.add_argument("--prompt", default="Hello", help="Prompt to send (default: Hello)")
    bm.add_argument("--model", default=None, help="Model name")
    bm.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    bm.add_argument("--api-key", default=None, help="API key")
    bm.add_argument(
        "--requests", type=int, default=10, help="Number of requests (default: 10)",
    )
    bm.add_argument(
        "--concurrency", type=int, default=1, help="Concurrent requests (default: 1)",
    )
    bm.add_argument("--json", default=None, dest="json_path", help="Export JSON report to path")
    bm.add_argument(
        "--rate-limit", type=float, default=None,
        help="Max requests per second",
    )
    _add_sampling_args(bm)

    # init
    init_cmd = sub.add_parser("init", help="Generate a starter xpyd-acc.toml config file")
    init_cmd.add_argument(
        "--output", "-o", default="xpyd-acc.toml",
        help="Output path (default: xpyd-acc.toml)",
    )
    init_cmd.add_argument(
        "--force", action="store_true", default=False,
        help="Overwrite existing file",
    )

    # config validate
        # Filter subcommand (M42)
    flt = sub.add_parser("filter", help="Filter samples from a batch report")
    flt.add_argument("--input", required=True, help="Input batch report JSON")
    flt.add_argument("--output", required=True, help="Output filtered report JSON")
    flt.add_argument(
        "--classification", default=None,
        help="Filter by classification (likely_bug, likely_uncertainty, match, unknown)",
    )
    flt.add_argument(
        "--divergent-only", action="store_true", default=False,
        help="Keep only divergent samples",
    )
    flt.add_argument(
        "--matched-only", action="store_true", default=False,
        help="Keep only matched samples",
    )
    flt.add_argument(
        "--min-logprob-gap", type=float, default=None,
        help="Minimum logprob gap threshold",
    )
    flt.add_argument(
        "--max-logprob-gap", type=float, default=None,
        help="Maximum logprob gap threshold",
    )
    flt.add_argument(
        "--min-context-length", type=int, default=None,
        help="Minimum context length",
    )
    flt.add_argument(
        "--max-context-length", type=int, default=None,
        help="Maximum context length",
    )
    flt.add_argument(
        "--search", default=None,
        help="Filter by text in prompt or output (case-insensitive)",
    )
    flt.add_argument(
        "--annotation-label", default=None,
        help="Filter by annotation label",
    )
    flt.add_argument(
        "--annotated", action="store_true", default=False,
        help="Only samples with annotations",
    )
    flt.add_argument(
        "--unannotated", action="store_true", default=False,
        help="Only samples without annotations",
    )

    cfg_cmd = sub.add_parser("config", help="Configuration utilities")
    cfg_sub = cfg_cmd.add_subparsers(dest="config_command")
    cfg_validate = cfg_sub.add_parser("validate", help="Validate a TOML config file")
    cfg_validate.add_argument(
        "path", nargs="?", default="xpyd-acc.toml",
        help="Path to config file (default: xpyd-acc.toml)",
    )

    # cluster (M49)
    cluster_cmd = sub.add_parser("cluster", help="Cluster divergent samples by divergence pattern")
    cluster_cmd.add_argument("--input", required=True, help="Path to batch report JSON file")
    cluster_cmd.add_argument(
        "--clusters", type=int, default=None,
        help="Number of clusters (auto-select if omitted)",
    )
    cluster_cmd.add_argument(
        "--json", dest="cluster_json", default=None, help="Export clusters as JSON",
    )

    # summary (M48)
    summary_cmd = sub.add_parser("summary", help="Compact summary of a batch report")
    summary_cmd.add_argument("report", help="Path to batch report JSON file")
    summary_cmd.add_argument(
        "--format", dest="summary_format", default="oneline",
        choices=["oneline", "json", "kv"],
        help="Output format (default: oneline)",
    )

    ann_cmd = sub.add_parser("annotate", help="Add notes and labels to batch report samples")
    ann_cmd.add_argument("--report", required=True, help="Path to batch report JSON")
    ann_cmd.add_argument("--sample", default=None, help="Sample ID to annotate")
    ann_cmd.add_argument("--note", default=None, help="Free-text note for the sample")
    ann_cmd.add_argument("--label", default=None, help="Classification label for the sample")
    ann_cmd.add_argument("--clear", action="store_true", help="Clear annotations for the sample")
    ann_cmd.add_argument("--list", dest="list_annotations", action="store_true",
                         help="List all annotations for the report")

    explain_cmd = sub.add_parser("explain", help="Deep-dive analysis of a single sample")
    explain_cmd.add_argument("--report", required=True, help="Path to batch report JSON")
    explain_cmd.add_argument("--sample", required=True, help="Sample ID to analyze")
    explain_cmd.add_argument(
        "--json", dest="explain_json", default=None,
        help="Export analysis as JSON",
    )

    # --- fingerprint ---
    fp_cmd = sub.add_parser("fingerprint", help="Model fingerprinting via deterministic probes")
    fp_cmd.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    fp_cmd.add_argument("--target", default=None, help="Target endpoint URL (compare two)")
    fp_cmd.add_argument("--model", default="default", help="Model name")
    fp_cmd.add_argument("--api-key", default=None, help="API key")
    fp_cmd.add_argument("--max-tokens", type=int, default=16, help="Max tokens per probe")
    fp_cmd.add_argument(
        "--json", dest="fp_json", default=None,
        help="Export fingerprint(s) as JSON",
    )
    fp_cmd.add_argument("--retries", type=int, default=3, help="Retry count")
    fp_cmd.add_argument("--retry-delay", type=float, default=1.0, help="Retry base delay")
    fp_cmd.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout per request")

    # --- reproducibility ---
    repro_cmd = sub.add_parser("reproducibility", help="Multi-run consistency measurement")
    repro_cmd.add_argument("--url", default=None, help="Single endpoint URL to measure")
    repro_cmd.add_argument("--baseline", default=None, help="Baseline endpoint URL (dual mode)")
    repro_cmd.add_argument("--target", default=None, help="Target endpoint URL (dual mode)")
    repro_cmd.add_argument("--prompt", required=True, help="Prompt text to send")
    repro_cmd.add_argument("--model", default="default", help="Model name")
    repro_cmd.add_argument("--api-key", default=None, help="API key")
    repro_cmd.add_argument("--max-tokens", type=int, default=256, help="Max tokens per request")
    repro_cmd.add_argument("--runs", type=int, default=5, help="Number of runs (default 5)")
    repro_cmd.add_argument(
        "--json", dest="repro_json", default=None,
        help="Export report as JSON",
    )
    repro_cmd.add_argument(
        "--threshold", type=float, default=None,
        help="Exit 1 if majority fraction below threshold (0.0-1.0)",
    )
    repro_cmd.add_argument("--retries", type=int, default=3, help="Retry count")
    repro_cmd.add_argument("--retry-delay", type=float, default=1.0, help="Retry base delay")
    repro_cmd.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout")
    _add_sampling_args(repro_cmd)

    ds = sub.add_parser("dataset-stats", help="Analyze dataset before batch comparison")
    ds.add_argument("dataset", help="Path to dataset file (JSONL, CSV, JSON)")
    ds.add_argument("--template", help="Path to prompt template file")
    ds.add_argument("--json", dest="json_path", help="Export stats as JSON")

    args = parser.parse_args(argv)

    # Setup logging from verbosity flags
    from xpyd_acc.log import setup_logging
    verbosity = -1 if args.quiet else args.verbose
    setup_logging(verbosity)

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

    # Apply named profile if --profile was specified
    if hasattr(args, "profile") and args.profile is not None:
        from xpyd_acc.profiles import apply_profile, parse_profiles, resolve_profile

        user_profiles = parse_profiles(config.profiles_raw) if config is not None else None
        try:
            profile = resolve_profile(args.profile, user_profiles)
        except KeyError as exc:
            parser.error(str(exc))
        apply_profile(vars(args), profile)

    # Handle 'cluster' subcommand (M49)
    if args.command == "cluster":
        _run_cluster(args)
        return

    # Handle 'summary' subcommand (M48)
    if args.command == "summary":
        _run_summary(args)
        return

    # Handle 'annotate' subcommand (M54)
    if args.command == "annotate":
        _run_annotate(args)
        return

    # Handle 'explain' subcommand (M52)
    if args.command == "explain":
        _run_explain(args)
        return

    # Handle 'reproducibility' subcommand (M62)
    if args.command == "reproducibility":
        _run_reproducibility(args)
        return

    # Handle 'fingerprint' subcommand (M55)
    if args.command == "fingerprint":
        _run_fingerprint(args)
        return

    # Handle 'filter' subcommand (M42)
    if args.command == "filter":
        _run_filter(args)
        return

    # Handle 'init' subcommand
    if args.command == "init":
        _run_init(args)
        return

    # Handle 'config' subcommand
    if args.command == "config":
        _run_config(args)
        return

    # Handle 'profiles' subcommand
    if args.command == "profiles":
        _run_profiles(config)
        return

    # Handle 'completion' subcommand
    if args.command == "completion":
        _run_completion(args, parser)
        return

    # Apply environment variable defaults (priority: CLI > env > config > defaults)
    from xpyd_acc.env import get_env_defaults

    env = get_env_defaults()
    _ENV_MAPPING: dict[str, str | None] = {
        "api_key": env.api_key,
        "baseline": env.baseline_url,
        "target": env.target_url,
        "model": env.model,
    }
    for key, env_val in _ENV_MAPPING.items():
        if env_val is not None and hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, env_val)

    # Apply numeric env defaults separately (typed)
    if env.temperature is not None and hasattr(args, "temperature") and args.temperature is None:
        args.temperature = env.temperature
    if env.top_p is not None and hasattr(args, "top_p") and args.top_p is None:
        args.top_p = env.top_p
    if env.seed is not None and hasattr(args, "seed") and args.seed is None:
        args.seed = env.seed
    if env.timeout is not None and hasattr(args, "timeout") and args.timeout is None:
        args.timeout = env.timeout
    if env.rate_limit is not None and hasattr(args, "rate_limit") and args.rate_limit is None:
        args.rate_limit = env.rate_limit
    if env.max_tokens is not None and hasattr(args, "max_tokens") and args.max_tokens is None:
        args.max_tokens = env.max_tokens
    if env.concurrency is not None and hasattr(args, "concurrency") and args.concurrency is None:
        args.concurrency = env.concurrency

    # Apply hardcoded defaults for any remaining None values
    _FINAL_DEFAULTS: dict[str, object] = {
        "model": "default",
        "max_tokens": 64,
        "api_key": "no-key",
        "concurrency": 5,
        "logprob_gap_threshold": 0.1,
        "output": "report.html",
        "retries": 3,
        "retry_delay": 1.0,
        "max_abs_threshold": 1e-3,
        "cosine_threshold": 0.999,
        "kv_max_abs_threshold": 1e-3,
        "kv_cosine_threshold": 0.999,
        "timeout": 120.0,
    }
    for key, default in _FINAL_DEFAULTS.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, default)

    # Stash config on args for subcommands that need it
    args._config = config

    if args.command == "batch-compare":
        asyncio.run(_run_batch_compare(args))
    elif args.command == "compare-output":
        _run_compare_output(args)
    elif args.command == "compare-logprobs":
        asyncio.run(_run_compare_logprobs(args))
    elif args.command == "compare-streaming":
        asyncio.run(_run_compare_streaming(args))
    elif args.command == "healthcheck":
        asyncio.run(_run_healthcheck(args))
    elif args.command == "detect":
        asyncio.run(_run_detect(args))
    elif args.command == "check-kv":
        _run_check_kv(args)
    elif args.command == "diagnose":
        asyncio.run(_run_diagnose(args))
    elif args.command == "report":
        _run_report(args)
    elif args.command == "regression":
        _run_regression(args)
    elif args.command == "diff":
        _run_diff(args)
    elif args.command == "aggregate":
        _run_aggregate(args)
    elif args.command == "watch":
        _run_watch(args)
    elif args.command == "snapshot":
        asyncio.run(_run_snapshot(args))
    elif args.command == "cache":
        _run_cache(args)
    elif args.command == "history":
        _run_history(args)
    elif args.command == "benchmark":
        asyncio.run(_run_benchmark(args))
    elif args.command == "bisect":
        asyncio.run(_run_bisect(args))
    elif args.command == "dataset-stats":
        _run_dataset_stats(args)
    else:
        print(f"xpyd-acc {args.command} — not yet implemented")


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


async def _preflight_healthcheck(
    urls: list[str], api_key: str = "no-key", timeout: float = 10.0,
) -> None:
    """Run pre-flight health check; exit if any endpoint is unhealthy."""
    from xpyd_acc.healthcheck import check_endpoints, format_healthcheck

    results = await check_endpoints(urls, api_key=api_key, timeout=timeout)
    print(format_healthcheck(results))
    if not all(r.healthy for r in results):
        print("\nAborting: unhealthy endpoint(s). Use --skip-healthcheck to bypass.")
        sys.exit(1)
    print()


def _run_completion(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Generate and output a shell completion script."""
    from xpyd_acc.completion import GENERATORS

    generator = GENERATORS[args.shell]
    script = generator(parser)

    if args.output:
        from pathlib import Path

        Path(args.output).write_text(script)
        print(f"Completion script written to {args.output}")
    else:
        print(script, end="")


def _run_profiles(config: object) -> None:
    """List all available named profiles."""
    from xpyd_acc.profiles import list_profiles, parse_profiles

    user_profiles = parse_profiles(config.profiles_raw) if config is not None else None
    all_profiles = list_profiles(user_profiles)

    if not all_profiles:
        print("No profiles available.")
        return

    print("Available profiles:\n")
    for name in sorted(all_profiles):
        profile = all_profiles[name]
        settings = profile.to_dict()
        if settings:
            parts = [f"{k}={v}" for k, v in settings.items()]
            print(f"  {name}: {', '.join(parts)}")
        else:
            print(f"  {name}: (empty)")


async def _run_healthcheck(args: argparse.Namespace) -> None:
    """Run standalone endpoint health check."""
    from xpyd_acc.healthcheck import check_endpoints, format_healthcheck

    results = await check_endpoints(
        args.url, api_key=args.api_key or "no-key", timeout=args.timeout,
    )
    print(format_healthcheck(results))
    if not all(r.healthy for r in results):
        sys.exit(1)


async def _run_batch_compare(args: argparse.Namespace) -> None:
    """Run batch dataset comparison."""
    # Validate --baseline vs --snapshot
    snapshot_path = getattr(args, "snapshot", None)
    has_baseline = args.baseline is not None
    if snapshot_path and has_baseline:
        print("Error: --baseline and --snapshot are mutually exclusive")
        sys.exit(2)
    if not snapshot_path and not has_baseline:
        print("Error: one of --baseline or --snapshot is required")
        sys.exit(2)

    # Handle snapshot replay mode
    if snapshot_path:
        await _run_batch_with_snapshot(args)
        return

    # Handle rerun mode
    if getattr(args, "rerun", None):
        await _run_rerun(args)
        return

    # Normalize target: always a list from argparse action="append"
    target_urls: list[str] = args.target if isinstance(args.target, list) else [args.target]

    # Handle dry run mode
    if getattr(args, "dry_run", False):
        from xpyd_acc.dry_run import format_dry_run, run_dry_run

        result = await run_dry_run(
            args.dataset,
            args.baseline,
            target_urls[0],
            template=args.template,
            skip_healthcheck=args.skip_healthcheck,
            model=args.model,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
            concurrency=args.concurrency,
            retries=args.retries,
            retry_delay=args.retry_delay,
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

    # Apply template if specified
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

    # Set up Rich progress bar unless disabled or non-TTY
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
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
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
        # Only pass config if any tolerance is enabled
        effective_match = match_config if (
            match_config.normalize_whitespace
            or match_config.ignore_case
            or match_config.numeric_tolerance is not None
        ) else None

        # Resolve output normalizers
        normalizer_specs = getattr(args, "normalizers", None) or []
        resolved_normalizers = None
        if normalizer_specs:
            from xpyd_acc.normalizers import resolve_normalizers
            resolved_normalizers = resolve_normalizers(normalizer_specs)

        # Set up response cache
        batch_cache = None
        if not getattr(args, "no_cache", False):
            from xpyd_acc.cache import DEFAULT_CACHE_DIR, DEFAULT_TTL, ResponseCache

            cache_dir = getattr(args, "cache_dir", None) or DEFAULT_CACHE_DIR
            cache_ttl = getattr(args, "cache_ttl", None) or DEFAULT_TTL
            batch_cache = ResponseCache(cache_dir=cache_dir, ttl=cache_ttl)

        is_multi = len(target_urls) > 1

        if is_multi:
            multi_report = await run_multi_batch(
                samples,
                args.baseline,
                target_urls,
                model=args.model,
                max_tokens=args.max_tokens,
                api_key=args.api_key,
                logprob_gap_threshold=args.logprob_gap_threshold,
                concurrency=args.concurrency,
                retries=args.retries,
                retry_delay=args.retry_delay,
                on_progress=on_progress if use_progress else None,
                match_config=effective_match,
                sampling_params=sampling,
                timeout=args.timeout,
                skip_validation=getattr(args, "skip_validation", False),
            )
            report = None  # not used in multi-target path
        else:
            multi_report = None
            from xpyd_acc.rate_limit import RateLimiter
            _rl = RateLimiter(getattr(args, "rate_limit", None))
            report = await run_batch(
                samples,
                args.baseline,
                target_urls[0],
                model=args.model,
                max_tokens=args.max_tokens,
                api_key=args.api_key,
                logprob_gap_threshold=args.logprob_gap_threshold,
                concurrency=args.concurrency,
                retries=args.retries,
                retry_delay=args.retry_delay,
                on_progress=on_progress if use_progress else None,
                match_config=effective_match,
                sampling_params=sampling,
                timeout=args.timeout,
                deduplicate=getattr(args, "deduplicate", False),
                enable_request_ids=not getattr(args, "no_request_id", False),
                cache=batch_cache,
                rate_limiter=_rl,
                normalizers=resolved_normalizers,
                skip_validation=getattr(args, "skip_validation", False),
                checkpoint_path=getattr(args, "checkpoint", None),
                checkpoint_clear=getattr(args, "checkpoint_clear", False),
            )
    finally:
        if progress_ctx is not None:
            progress_ctx.stop()

    # Print cache stats if caching was active
    if batch_cache is not None:
        cs = batch_cache.stats()
        if cs.hits + cs.misses > 0:
            print(f"\nCache: {cs.hits} hits, {cs.misses} misses ({cs.hit_rate:.0%} hit rate)")

    # Apply cost estimation if pricing is configured
    _apply_cost_to_report(args, report if report is not None else multi_report)

    if multi_report is not None:
        # Multi-target mode
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
    else:
        # Apply confidence interval if requested
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

        # Send webhook notification if configured
        await _maybe_send_webhook(args, report)

        # Check truncation threshold
        _check_truncation_threshold(args, report)

        fail_threshold = _resolve_fail_threshold(args, getattr(args, "_config", None))
        if fail_threshold is not None:
            # When confidence is enabled, use CI lower bound for threshold check
            if getattr(args, "confidence", False) and report.divergence_ci_lower is not None:
                check_val = report.divergence_ci_lower
                label = f"CI lower bound {check_val:.1%}"
            else:
                check_val = report.divergence_rate
                label = f"divergence rate {check_val:.1%}"

            if check_val > fail_threshold:
                print(
                    f"\n✗ FAIL: {label}"
                    f" exceeds threshold {fail_threshold:.1%}",
                )
                sys.exit(1)
            else:
                print(
                    f"\n✓ PASS: {label}"
                    f" within threshold {fail_threshold:.1%}",
                )
        elif report.divergent_samples > 0:
            sys.exit(1)


async def _maybe_send_webhook(
    args: argparse.Namespace,
    report: object,
) -> None:
    """Send webhook notification if configured."""
    import os

    from xpyd_acc.notify import WebhookPayload, resolve_webhook_config, send_webhook

    config = resolve_webhook_config(
        cli_url=getattr(args, "webhook", None),
        cli_headers=getattr(args, "webhook_headers", None),
        cli_always=getattr(args, "webhook_always", False),
        env_url=os.environ.get("XPYD_ACC_WEBHOOK_URL"),
        toml_config=getattr(args, "_config", None),
    )
    if config is None:
        return

    total = getattr(report, "total_samples", 0)
    divergent = getattr(report, "divergent_samples", 0)
    rate = getattr(report, "divergence_rate", 0.0)

    payload = WebhookPayload(
        event="batch_complete",
        divergence_detected=divergent > 0,
        total_samples=total,
        divergent_samples=divergent,
        divergence_rate=rate,
    )
    await send_webhook(config, payload)


def _apply_cost_to_report(
    args: argparse.Namespace,
    report: object,
) -> None:
    """Apply cost estimation to report usage if pricing is configured."""
    from xpyd_acc.cost import CostConfig

    input_price = getattr(args, "input_price", None)
    output_price = getattr(args, "output_price", None)

    # Fall back to TOML config
    config = getattr(args, "_config", None)
    if config is not None:
        cost_cfg = getattr(config, "cost", None)
        if cost_cfg is not None:
            if input_price is None:
                input_price = getattr(cost_cfg, "input_price_per_m", None) or None
            if output_price is None:
                output_price = getattr(cost_cfg, "output_price_per_m", None) or None

    # If neither price is set, nothing to do
    if not input_price and not output_price:
        return

    usage = getattr(report, "usage", None)
    if usage is None:
        return

    cost_config = CostConfig(
        input_price_per_m=input_price or 0.0,
        output_price_per_m=output_price or 0.0,
    )
    usage.apply_cost(cost_config)


def _maybe_apply_confidence(
    args: argparse.Namespace,
    report: object,
) -> None:
    """Apply confidence interval to report if --confidence flag is set."""
    confidence = getattr(args, "confidence", False)
    # Also check TOML config
    config = getattr(args, "_config", None)
    if not confidence and config is not None:
        confidence = getattr(getattr(config, "batch", None), "confidence", False)
    if confidence and report.total_samples > 0:
        from xpyd_acc.batch_compare import apply_confidence
        level = getattr(args, "confidence_level", 0.95)
        apply_confidence(report, level)


def _check_truncation_threshold(
    args: argparse.Namespace,
    report: object,
) -> None:
    """Check if truncated sample ratio exceeds --warn-truncated threshold.

    Exits with code 2 if the threshold is exceeded.
    """
    warn_truncated = getattr(args, "warn_truncated", None)
    if warn_truncated is None:
        return

    total = getattr(report, "total_samples", 0)
    truncated = getattr(report, "truncated_count", 0)
    if total == 0:
        return

    ratio = truncated / total
    if ratio > warn_truncated:
        print(
            f"\n⚠ TRUNCATION WARNING: {truncated}/{total} samples truncated "
            f"({ratio:.1%}) exceeds threshold {warn_truncated:.1%}",
        )
        sys.exit(2)
    elif truncated > 0:
        print(
            f"\nTruncation: {truncated}/{total} samples ({ratio:.1%}) "
            f"within threshold {warn_truncated:.1%}",
        )


def _resolve_fail_threshold(
    args: argparse.Namespace, config: object,
) -> float | None:
    """Resolve fail threshold from CLI > env > config > None."""
    import os

    # CLI flag (already merged from config by merge_cli_args)
    cli_val = getattr(args, "fail_threshold", None)
    if cli_val is not None:
        return cli_val

    # Environment variable
    env_val = os.environ.get("XPYD_ACC_FAIL_THRESHOLD")
    if env_val is not None:
        try:
            return float(env_val)
        except ValueError:
            pass

    return None


async def _run_rerun(args: argparse.Namespace) -> None:
    """Run selective sample rerun from a previous report."""
    from pathlib import Path

    from xpyd_acc.batch_compare import (
        export_csv,
        export_markdown,
        format_report,
        run_batch,
    )
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

    # Apply template if specified
    if args.template:
        from xpyd_acc.templates import resolve_template

        template = resolve_template(args.template)
        print(f"Using template: {template.name}")
        for sample in plan.divergent_samples:
            variables = {"prompt": sample.prompt, **sample.metadata}
            sample.prompt = template.render(variables)

    # Set up Rich progress bar unless disabled or non-TTY
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
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
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
            match_config.normalize_whitespace
            or match_config.ignore_case
            or match_config.numeric_tolerance is not None
        ) else None

        report = await run_batch(
            plan.divergent_samples,
            args.baseline,
            rerun_target,
            model=args.model,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
            logprob_gap_threshold=args.logprob_gap_threshold,
            concurrency=args.concurrency,
            retries=args.retries,
            retry_delay=args.retry_delay,
            on_progress=on_progress if use_progress else None,
            match_config=effective_match,
            sampling_params=sampling,
            timeout=args.timeout,
            deduplicate=getattr(args, "deduplicate", False),
            enable_request_ids=not getattr(args, "no_request_id", False),
            skip_validation=getattr(args, "skip_validation", False),
        )
    finally:
        if progress_ctx is not None:
            progress_ctx.stop()

    # Handle merge mode
    if args.rerun_merge:
        report = merge_rerun_results(args.rerun, report)
        # Overwrite the original report
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

    # Apply fail threshold to rerun mode too
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

        # Collect with timing info
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


async def _run_batch_with_snapshot(args: argparse.Namespace) -> None:
    """Run batch comparison using a saved snapshot as baseline."""
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

    # Apply template if specified
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
        match_config.normalize_whitespace
        or match_config.ignore_case
        or match_config.numeric_tolerance is not None
    ) else None

    import asyncio

    semaphore = asyncio.Semaphore(args.concurrency or 5)
    results: list[SampleResult] = []
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
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )

    def on_progress_update(done: int, total: int) -> None:
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
            sample_id=sample.id,
            prompt=sample.prompt,
            baseline_output=baseline_text,
            target_output=target_text,
            exact_match=exact,
            first_divergence_index=div_idx,
            baseline_logprob_at_divergence=b_lp_at_div,
            target_logprob_at_divergence=t_lp_at_div,
            logprob_gap=gap,
            classification=classification,
            context_length=ctx_len,
        )

    if progress_ctx is not None:
        progress_ctx.start()
        task_id = progress_ctx.add_task("Comparing samples", total=total)

    try:
        tasks = [process_one(s) for s in samples]
        results = list(await asyncio.gather(*tasks))
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

    # Apply template if specified
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
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )

    def on_progress(completed: int, total: int) -> None:
        if progress_ctx is not None and task_id is not None:
            progress_ctx.update(task_id, completed=completed)

    if progress_ctx is not None:
        progress_ctx.start()
        task_id = progress_ctx.add_task("Capturing snapshot", total=len(samples))

    try:
        snap = await capture_snapshot(
            samples,
            args.baseline,
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
        action = "Would remove" if args.dry_run else "Removed"
        if not removed:
            print("Nothing to purge.")
        else:
            console = Console()
            table = Table(title=f"{action} {len(removed)} entries")
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


def _run_summary(args: argparse.Namespace) -> None:
    """Run the summary subcommand."""
    import sys
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

    # Apply annotation-based filters
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

        # Recalculate stats
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


def _run_init(args: argparse.Namespace) -> None:
    """Generate a starter config file."""
    from pathlib import Path

    from xpyd_acc.config_validate import generate_starter_config

    output = Path(args.output)
    try:
        path = generate_starter_config(output, force=args.force)
        print(f"Created config file: {path}")
    except FileExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _run_config(args: argparse.Namespace) -> None:
    """Handle config subcommands."""
    if not hasattr(args, "config_command") or args.config_command is None:
        print("Usage: xpyd-acc config {validate}")
        return

    if args.config_command == "validate":
        from pathlib import Path

        from xpyd_acc.config_validate import validate_config

        path = Path(args.path)
        try:
            issues = validate_config(path)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if not issues:
            print(f"✅ {path} is valid")
            sys.exit(0)
        else:
            has_errors = False
            for issue in issues:
                print(issue)
                if issue.startswith("error:"):
                    has_errors = True
            if has_errors:
                sys.exit(1)
            else:
                print(f"\n⚠️  {path} has warnings but no errors")
                sys.exit(0)


async def _run_bisect(args: argparse.Namespace) -> None:
    """Run bisect to find minimum divergence context length."""
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
        import pathlib
        pathlib.Path(json_path).write_text(result.to_json())
        print(f"   Exported to {json_path}")


def _run_cluster(args: argparse.Namespace) -> None:
    """Run the cluster subcommand."""
    import json
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
            str(c.cluster_id),
            str(c.size),
            f"{c.avg_divergence_index:.1f}",
            f"{c.avg_logprob_gap:.4f}",
            f"{c.avg_context_length:.0f}",
            c.representative_sample_id,
        )

    console.print(table)

    if args.cluster_json:
        result.to_json(args.cluster_json)
        console.print(f"\n  Exported to {args.cluster_json}")


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


def _run_fingerprint(args: argparse.Namespace) -> None:
    """Handle the 'fingerprint' subcommand (M55)."""
    import asyncio
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


def _run_reproducibility(args: argparse.Namespace) -> None:
    """Handle the 'reproducibility' subcommand (M62)."""
    import asyncio
    from pathlib import Path

    from xpyd_acc.reproducibility import (
        format_reproducibility,
        run_reproducibility,
    )

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

        # Check threshold
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
