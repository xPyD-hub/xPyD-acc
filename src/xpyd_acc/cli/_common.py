"""Shared argument builders and utility functions for CLI subcommands."""

from __future__ import annotations

import argparse
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
    """Check if truncated sample ratio exceeds --warn-truncated threshold."""
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
