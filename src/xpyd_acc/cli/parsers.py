"""Subcommand parser registration — all argparse subparsers defined here."""

from __future__ import annotations

import argparse

from ._common import _add_sampling_args


def register_all(sub: argparse._SubParsersAction) -> None:
    """Register all subcommands on the given subparser action."""
    _register_compare(sub)
    _register_diagnose(sub)
    _register_batch(sub)
    _register_report(sub)
    _register_streaming(sub)
    _register_healthcheck(sub)
    _register_detect(sub)
    _register_regression(sub)
    _register_diff(sub)
    _register_ab_test(sub)
    _register_concurrency_sweep(sub)
    _register_entropy(sub)
    _register_length_bias(sub)
    _register_sensitivity(sub)
    _register_check_kv(sub)
    _register_aggregate(sub)
    _register_watch(sub)
    _register_profiles(sub)
    _register_completion(sub)
    _register_bisect(sub)
    _register_snapshot(sub)
    _register_history(sub)
    _register_benchmark(sub)
    _register_init(sub)
    _register_filter(sub)
    _register_config(sub)
    _register_cluster(sub)
    _register_summary(sub)
    _register_annotate(sub)
    _register_explain(sub)
    _register_fingerprint(sub)
    _register_root_cause(sub)
    _register_reproducibility(sub)
    _register_dataset_stats(sub)
    _register_cache(sub)
def _register_compare(sub):
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

    oc = sub.add_parser("compare-output", help="Compare text outputs from two endpoints")
    oc_input = oc.add_mutually_exclusive_group(required=True)
    oc_input.add_argument("--baseline-text", help="Baseline output text (inline)")
    oc_input.add_argument("--baseline-file", help="Path to file with baseline output")
    oc.add_argument("--target-text", help="Target output text (inline)")
    oc.add_argument("--target-file", help="Path to file with target output")
def _register_diagnose(sub):
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
def _register_batch(sub):
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
    bc.add_argument("--no-progress", action="store_true", default=False,
        help="Disable progress bar")
    bc.add_argument("--skip-healthcheck", action="store_true", default=False,
        help="Skip healthcheck")
    bc.add_argument("--skip-validation", action="store_true", default=False,
        help="Skip response validation")
    bc.add_argument("--template", default=None, help="Prompt template: built-in name or path")
    bc.add_argument("--dry-run", action="store_true", default=False,
        help="Validate without API requests")
    bc.add_argument("--normalize-whitespace", action="store_true", default=False,
        help="Normalize whitespace")
    bc.add_argument("--ignore-case", action="store_true", default=False,
        help="Case-insensitive matching")
    bc.add_argument("--numeric-tolerance", type=float, default=None, help="Numeric tolerance")
    bc.add_argument("--rerun", default=None, metavar="REPORT_JSON", help="Rerun divergent samples")
    bc.add_argument("--rerun-merge", action="store_true", default=False,
        help="Merge rerun results back")
    bc.add_argument("--timeout", type=float, default=None,
        help="HTTP timeout in seconds (default: 120.0)")
    bc.add_argument("--fail-threshold", type=float, default=None, help="Fail threshold (0.0–1.0)")
    bc.add_argument("--confidence", action="store_true", default=False,
        help="Compute confidence interval")
    bc.add_argument("--confidence-level", type=float, default=0.95,
        help="Confidence level (default: 0.95)")
    _add_sampling_args(bc)
    bc.add_argument("--no-request-id", action="store_true", default=False,
        help="Disable X-Request-ID")
    bc.add_argument("--deduplicate", action="store_true", default=False, help="Deduplicate prompts")
    bc.add_argument(
        "--normalizer", action="append", default=None, dest="normalizers",
        help="Output normalizer (repeatable). Built-in: strip_thinking_tags, "
             "normalize_json, normalize_numbers. Or module:function for custom.",
    )
    bc.add_argument("--rate-limit", type=float, default=None, help="Max requests per second")
    bc.add_argument("--cache-dir", default=None, help="Response cache directory")
    bc.add_argument("--no-cache", action="store_true", default=False, help="Disable caching")
    bc.add_argument("--warn-truncated", type=float, default=None,
        help="Truncation warning threshold")
    bc.add_argument("--cache-ttl", type=float, default=None,
        help="Cache TTL in seconds (default: 3600)")
    bc.add_argument("--webhook", default=None, metavar="URL", help="Webhook URL for alerts")
    bc.add_argument(
        "--webhook-header", action="append", default=None, dest="webhook_headers",
        help="Custom webhook header as 'Key: Value' (repeatable)",
    )
    bc.add_argument("--webhook-always", action="store_true", default=False,
        help="Send webhook always")
    bc.add_argument("--input-price", type=float, default=None,
        help="Input price per 1M tokens (USD)")
    bc.add_argument("--output-price", type=float, default=None,
        help="Output price per 1M tokens (USD)")
    bc.add_argument("--checkpoint", default=None, metavar="PATH",
        help="Checkpoint file for resumable runs")
    bc.add_argument("--checkpoint-clear", action="store_true", default=False,
        help="Clear checkpoint")
    bc.add_argument(
        "--header", action="append", default=None, dest="headers",
        help="Custom HTTP header as 'Key: Value' (repeatable). Overrides defaults.",
    )
def _register_streaming(sub):
    cs = sub.add_parser("compare-streaming", help="Compare SSE streaming outputs")
    cs.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    cs.add_argument("--target", required=True, help="Target endpoint URL")
    cs.add_argument("--prompt", required=True, help="Prompt to send")
    cs.add_argument("--model", default=None, help="Model name (default: default)")
    cs.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    cs.add_argument("--api-key", default=None, help="API key for both endpoints")
    cs.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    cs.add_argument("--skip-healthcheck", action="store_true", default=False,
        help="Skip healthcheck")
    cs.add_argument("--timing", action="store_true", default=False, help="Enable timing analysis")
    cs.add_argument("--normalize-whitespace", action="store_true", default=False,
        help="Normalize whitespace")
    cs.add_argument("--ignore-case", action="store_true", default=False,
        help="Case-insensitive matching")
    cs.add_argument("--numeric-tolerance", type=float, default=None, help="Numeric tolerance")
    _add_sampling_args(cs)
def _register_report(sub):
    rp = sub.add_parser("report", help="Generate HTML report from batch comparison JSON")
    rp.add_argument("--input", required=True, help="Path to batch results JSON file")
    rp.add_argument("--output", default=None, help="Output HTML file path (default: report.html)")
def _register_healthcheck(sub):
    hc = sub.add_parser("healthcheck", help="Check endpoint health")
    hc.add_argument("url", nargs="+", help="Endpoint URL(s) to check")
    hc.add_argument("--api-key", default=None, help="API key for endpoints")
    hc.add_argument("--timeout", type=float, default=10.0, help="Timeout per endpoint in seconds")
def _register_detect(sub):
    det = sub.add_parser("detect", help="Detect xPyD endpoint type")
    det.add_argument("url", help="Endpoint URL to probe")
    det.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")
def _register_regression(sub):
    reg = sub.add_parser("regression", help="Detect regressions between two batch runs")
    reg.add_argument("--baseline", required=True, help="Path to baseline batch result JSON")
    reg.add_argument("--current", required=True, help="Path to current batch result JSON")
    reg.add_argument("--json", dest="json_path", default=None,
        help="Export regression report as JSON")
def _register_diff(sub):
    diff_p = sub.add_parser("diff", help="Side-by-side comparison of two batch reports")
    diff_p.add_argument("--old", required=True, help="Path to old batch report JSON")
    diff_p.add_argument("--new", required=True, dest="new_report",
        help="Path to new batch report JSON")
    diff_p.add_argument("--json", dest="json_path", default=None, help="Export diff result as JSON")
def _register_ab_test(sub):
    ab = sub.add_parser("ab-test", help="A/B test divergence rates from two batch reports")
    ab.add_argument("--report-a", required=True, help="Path to first batch report JSON")
    ab.add_argument("--report-b", required=True, help="Path to second batch report JSON")
    ab.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    ab.add_argument("--json", dest="json_path", default=None, help="Export A/B test result as JSON")
def _register_concurrency_sweep(sub):
    csweep = sub.add_parser(
        "concurrency-sweep",
        help="Measure divergence at different concurrency levels",
    )
    csweep.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    csweep.add_argument("--target", required=True, help="Target endpoint URL")
    csweep.add_argument("--dataset", required=True, help="Path to dataset file")
    csweep.add_argument("--levels", required=True,
        help="Comma-separated concurrency levels (e.g. 1,2,4,8)")
    csweep.add_argument("--model", default=None, help="Model name")
    csweep.add_argument("--api-key", default=None, help="API key")
    csweep.add_argument("--max-tokens", type=int, default=256, help="Max tokens per request")
    csweep.add_argument("--template", default=None, help="Prompt template path")
    csweep.add_argument("--retries", type=int, default=3, help="Retry count")
    csweep.add_argument("--retry-delay", type=float, default=1.0, help="Retry delay")
    csweep.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout")
    csweep.add_argument("--skip-validation", action="store_true", help="Skip response validation")
    csweep.add_argument("--json", dest="json_path", default=None, help="Export results as JSON")
    _add_sampling_args(csweep)
def _register_entropy(sub):
    ent = sub.add_parser("entropy", help="Analyze output entropy from logprob files")
    ent.add_argument("--baseline-logprobs", required=True, help="Path to baseline logprobs JSON")
    ent.add_argument("--target-logprobs", default=None,
        help="Path to target logprobs JSON (optional)")
    ent.add_argument("--divergence-index", type=int, default=None, help="Token index of divergence")
    ent.add_argument("--context-window", type=int, default=5, help="Context window (default: 5)")
    ent.add_argument("--json", dest="json_path", default=None, help="Export results as JSON")
def _register_length_bias(sub):
    lbias = sub.add_parser("length-bias", help="Detect output length bias in batch reports")
    lbias.add_argument("--report", required=True, help="Path to batch report JSON file")
    lbias.add_argument("--alpha", type=float, default=0.05,
        help="Significance level (default: 0.05)")
    lbias.add_argument("--json", dest="json_path", default=None, help="Export results as JSON")
def _register_sensitivity(sub):
    sens = sub.add_parser("sensitivity", help="Prompt sensitivity analysis")
    sens.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    sens.add_argument("--target", required=True, help="Target endpoint URL")
    sens.add_argument("--prompt", required=True, help="Prompt to test")
    sens.add_argument("--model", default=None, help="Model name (default: default)")
    sens.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    sens.add_argument("--api-key", default=None, help="API key for both endpoints")
    sens.add_argument("--perturbations", type=int, default=5,
        help="Number of perturbations (default: 5)")
    sens.add_argument("--retries", type=int, default=None, help="Max retry attempts (default: 3)")
    sens.add_argument("--retry-delay", type=float, default=None,
        help="Base retry delay (default: 1.0)")
    sens.add_argument("--json", dest="json_path", default=None, help="Export results as JSON")
    _add_sampling_args(sens)
def _register_check_kv(sub):
    kv = sub.add_parser("check-kv", help="Check KV cache numerical accuracy")
    kv.add_argument("--baseline", required=True, help="Path to baseline KV cache (.npz)")
    kv.add_argument("--target", required=True, help="Path to target KV cache (.npz)")
    kv.add_argument("--max-abs-threshold", type=float, default=None,
        help="Max abs diff threshold (default: 1e-3)")
    kv.add_argument("--cosine-threshold", type=float, default=None,
        help="Cosine threshold (default: 0.999)")
    kv.add_argument("--json", action="store_true", help="Output report as JSON")
def _register_aggregate(sub):
    agg = sub.add_parser("aggregate", help="Aggregate multiple batch run reports")
    agg.add_argument("--reports", nargs="+", required=True, help="Paths to batch report JSON files")
    agg.add_argument("--json", default=None, dest="json_path",
        help="Export aggregated report as JSON")
def _register_watch(sub):
    wp = sub.add_parser("watch", help="Continuous divergence monitoring")
    wp.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    wp.add_argument("--target", required=True, help="Target endpoint URL")
    wp.add_argument("--prompt", required=True, help="Prompt to compare")
    wp.add_argument("--model", default=None, help="Model name")
    wp.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    wp.add_argument("--api-key", default=None, help="API key for endpoints")
    wp.add_argument("--interval", type=float, default=60.0,
        help="Seconds between iterations (default: 60)")
    wp.add_argument("--max-iterations", type=int, default=None, help="Stop after N iterations")
    wp.add_argument("--alert-threshold", type=int, default=None,
        help="Exit 1 after N consecutive failures")
    wp.add_argument("--log", default=None, dest="log_path", help="JSON log file path")
    wp.add_argument("--retries", type=int, default=None, help="Max retry attempts")
    wp.add_argument("--retry-delay", type=float, default=None, help="Base retry delay (seconds)")
    wp.add_argument("--skip-healthcheck", action="store_true", help="Skip pre-flight healthcheck")
    _add_sampling_args(wp)
def _register_profiles(sub):
    sub.add_parser("profiles", help="List available named profiles")
def _register_completion(sub):
    comp = sub.add_parser("completion", help="Generate shell completion script")
    comp.add_argument("shell", choices=["bash", "zsh", "fish"], help="Target shell")
    comp.add_argument("--output", default=None, help="Write script to file instead of stdout")
def _register_bisect(sub):
    bis = sub.add_parser("bisect", help="Binary search for min context length causing divergence")
    bis.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    bis.add_argument("--target", required=True, help="Target endpoint URL")
    bis.add_argument("--prompt", required=True, help="Full prompt text to bisect")
    bis.add_argument("--model", default="default", help="Model name")
    bis.add_argument("--api-key", default=None, help="API key for endpoints")
    bis.add_argument("--min-length", type=int, default=None, help="Minimum prefix length")
    bis.add_argument("--max-length", type=int, default=None, help="Maximum prefix length")
    bis.add_argument("--retries", type=int, default=None, help="Max retry attempts (default: 3)")
    bis.add_argument("--retry-delay", type=float, default=None,
        help="Base retry delay (default: 1.0)")
    bis.add_argument("--timeout", type=float, default=None, help="HTTP timeout (default: 120.0)")
    bis.add_argument("--json", default=None, metavar="PATH", help="Export result as JSON")
    _add_sampling_args(bis)
def _register_snapshot(sub):
    sc = sub.add_parser("snapshot", help="Capture baseline outputs as a snapshot")
    sc.add_argument("action", choices=["capture"], help="Snapshot action")
    sc.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    sc.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    sc.add_argument("--output", required=True, help="Output snapshot JSON path")
    sc.add_argument("--model", default=None, help="Model name (default: default)")
    sc.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    sc.add_argument("--api-key", default=None, help="API key for endpoint")
    sc.add_argument("--concurrency", type=int, default=None,
        help="Max concurrent requests (default: 5)")
    sc.add_argument("--retries", type=int, default=None, help="Max retry attempts (default: 3)")
    sc.add_argument("--retry-delay", type=float, default=None,
        help="Base retry delay (default: 1.0)")
    sc.add_argument("--timeout", type=float, default=None, help="HTTP timeout (default: 120.0)")
    sc.add_argument("--rate-limit", type=float, default=None, help="Max requests per second")
    sc.add_argument("--template", default=None, help="Prompt template path")
    sc.add_argument("--no-progress", action="store_true", default=False,
        help="Disable progress bar")
    _add_sampling_args(sc)
def _register_history(sub):
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
    hist_trend.add_argument("--fail-on-regression", action="store_true", default=False,
        help="Exit 1 on regression")
    hist_trend.add_argument("--history-dir", default=None, help="History directory")
    hist_purge = hist_sub.add_parser("purge", help="Remove old history entries")
    hist_purge.add_argument("--older-than", type=int, required=True,
        help="Remove entries older than N days")
    hist_purge.add_argument("--keep-last", type=int, default=0,
        help="Always keep last N entries (default: 0)")
    hist_purge.add_argument("--dry-run", action="store_true", default=False,
        help="Show what would be removed")
    hist_purge.add_argument("--history-dir", default=None, help="History directory")
def _register_benchmark(sub):
    bm = sub.add_parser("benchmark", help="Benchmark endpoint latency")
    bm.add_argument("url", help="Endpoint URL to benchmark")
    bm.add_argument("--prompt", default="Hello", help="Prompt to send (default: Hello)")
    bm.add_argument("--model", default=None, help="Model name")
    bm.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: 64)")
    bm.add_argument("--api-key", default=None, help="API key")
    bm.add_argument("--requests", type=int, default=10, help="Number of requests (default: 10)")
    bm.add_argument("--concurrency", type=int, default=1, help="Concurrent requests (default: 1)")
    bm.add_argument("--json", default=None, dest="json_path", help="Export JSON report")
    bm.add_argument("--rate-limit", type=float, default=None, help="Max requests per second")
    _add_sampling_args(bm)
def _register_init(sub):
    init_cmd = sub.add_parser("init", help="Generate a starter xpyd-acc.toml config file")
    init_cmd.add_argument("--output", "-o", default="xpyd-acc.toml", help="Output path")
    init_cmd.add_argument("--force", action="store_true", default=False,
        help="Overwrite existing file")
def _register_filter(sub):
    flt = sub.add_parser("filter", help="Filter samples from a batch report")
    flt.add_argument("--input", required=True, help="Input batch report JSON")
    flt.add_argument("--output", required=True, help="Output filtered report JSON")
    flt.add_argument("--classification", default=None, help="Filter by classification")
    flt.add_argument("--divergent-only", action="store_true", default=False,
        help="Keep only divergent")
    flt.add_argument("--matched-only", action="store_true", default=False, help="Keep only matched")
    flt.add_argument("--min-logprob-gap", type=float, default=None, help="Minimum logprob gap")
    flt.add_argument("--max-logprob-gap", type=float, default=None, help="Maximum logprob gap")
    flt.add_argument("--min-context-length", type=int, default=None, help="Minimum context length")
    flt.add_argument("--max-context-length", type=int, default=None, help="Maximum context length")
    flt.add_argument("--search", default=None, help="Filter by text in prompt/output")
    flt.add_argument("--annotation-label", default=None, help="Filter by annotation label")
    flt.add_argument("--annotated", action="store_true", default=False,
        help="Only annotated samples")
    flt.add_argument("--unannotated", action="store_true", default=False,
        help="Only unannotated samples")
def _register_config(sub):
    cfg_cmd = sub.add_parser("config", help="Configuration utilities")
    cfg_sub = cfg_cmd.add_subparsers(dest="config_command")
    cfg_validate = cfg_sub.add_parser("validate", help="Validate a TOML config file")
    cfg_validate.add_argument("path", nargs="?", default="xpyd-acc.toml", help="Config file path")
def _register_cluster(sub):
    cluster_cmd = sub.add_parser("cluster", help="Cluster divergent samples by pattern")
    cluster_cmd.add_argument("--input", required=True, help="Path to batch report JSON")
    cluster_cmd.add_argument("--clusters", type=int, default=None,
        help="Number of clusters (auto if omitted)")
    cluster_cmd.add_argument("--json", dest="cluster_json", default=None,
        help="Export clusters as JSON")
def _register_summary(sub):
    summary_cmd = sub.add_parser("summary", help="Compact summary of a batch report")
    summary_cmd.add_argument("report", help="Path to batch report JSON file")
    summary_cmd.add_argument(
        "--format", dest="summary_format", default="oneline",
        choices=["oneline", "json", "kv"], help="Output format (default: oneline)",
    )
def _register_annotate(sub):
    ann_cmd = sub.add_parser("annotate", help="Add notes and labels to batch report samples")
    ann_cmd.add_argument("--report", required=True, help="Path to batch report JSON")
    ann_cmd.add_argument("--sample", default=None, help="Sample ID to annotate")
    ann_cmd.add_argument("--note", default=None, help="Free-text note")
    ann_cmd.add_argument("--label", default=None, help="Classification label")
    ann_cmd.add_argument("--clear", action="store_true", help="Clear annotations for sample")
    ann_cmd.add_argument("--list", dest="list_annotations", action="store_true",
        help="List all annotations")
def _register_explain(sub):
    explain_cmd = sub.add_parser("explain", help="Deep-dive analysis of a single sample")
    explain_cmd.add_argument("--report", required=True, help="Path to batch report JSON")
    explain_cmd.add_argument("--sample", required=True, help="Sample ID to analyze")
    explain_cmd.add_argument("--json", dest="explain_json", default=None, help="Export as JSON")
def _register_fingerprint(sub):
    fp_cmd = sub.add_parser("fingerprint", help="Model fingerprinting via deterministic probes")
    fp_cmd.add_argument("--baseline", required=True, help="Baseline endpoint URL")
    fp_cmd.add_argument("--target", default=None, help="Target endpoint URL")
    fp_cmd.add_argument("--model", default="default", help="Model name")
    fp_cmd.add_argument("--api-key", default=None, help="API key")
    fp_cmd.add_argument("--max-tokens", type=int, default=16, help="Max tokens per probe")
    fp_cmd.add_argument("--json", dest="fp_json", default=None, help="Export fingerprint as JSON")
    fp_cmd.add_argument("--retries", type=int, default=3, help="Retry count")
    fp_cmd.add_argument("--retry-delay", type=float, default=1.0, help="Retry delay")
    fp_cmd.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout")
def _register_reproducibility(sub):
    repro_cmd = sub.add_parser("reproducibility", help="Multi-run consistency measurement")
    repro_cmd.add_argument("--url", default=None, help="Single endpoint URL")
    repro_cmd.add_argument("--baseline", default=None, help="Baseline endpoint URL (dual mode)")
    repro_cmd.add_argument("--target", default=None, help="Target endpoint URL (dual mode)")
    repro_cmd.add_argument("--prompt", required=True, help="Prompt text")
    repro_cmd.add_argument("--model", default="default", help="Model name")
    repro_cmd.add_argument("--api-key", default=None, help="API key")
    repro_cmd.add_argument("--max-tokens", type=int, default=256, help="Max tokens")
    repro_cmd.add_argument("--runs", type=int, default=5, help="Number of runs (default 5)")
    repro_cmd.add_argument("--json", dest="repro_json", default=None, help="Export report as JSON")
    repro_cmd.add_argument("--threshold", type=float, default=None,
        help="Majority fraction threshold")
    repro_cmd.add_argument("--retries", type=int, default=3, help="Retry count")
    repro_cmd.add_argument("--retry-delay", type=float, default=1.0, help="Retry delay")
    repro_cmd.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout")
    _add_sampling_args(repro_cmd)
def _register_dataset_stats(sub):
    ds = sub.add_parser("dataset-stats", help="Analyze dataset before batch comparison")
    ds.add_argument("dataset", help="Path to dataset file (JSONL, CSV, JSON)")
    ds.add_argument("--template", help="Path to prompt template file")
    ds.add_argument("--json", dest="json_path", help="Export stats as JSON")
def _register_cache(sub):
    cache_cmd = sub.add_parser("cache", help="Manage response cache")
    cache_sub = cache_cmd.add_subparsers(dest="cache_action")
    cache_clear = cache_sub.add_parser("clear", help="Remove all cached responses")
    cache_clear.add_argument("--cache-dir", default=None, help="Cache directory")
    cache_stats = cache_sub.add_parser("stats", help="Show cache statistics")
    cache_stats.add_argument("--cache-dir", default=None, help="Cache directory")


def _register_root_cause(sub):
    rc_cmd = sub.add_parser("root-cause", help="Analyze divergence root cause from batch report")
    rc_cmd.add_argument("--report", required=True, help="Path to batch report JSON")
    rc_cmd.add_argument("--json", dest="rc_json", default=None, help="Export analysis as JSON")
