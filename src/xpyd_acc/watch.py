"""Watch mode: continuous divergence monitoring."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table

from xpyd_acc.log import get_logger
from xpyd_acc.logprobs import LogprobsCollector, LogprobsComparator
from xpyd_acc.sampling import SamplingParams

logger = get_logger("watch")


@dataclass
class WatchIteration:
    """Result of a single watch iteration."""

    iteration: int
    timestamp: float
    passed: bool
    first_divergence_index: int | None
    baseline_token: str | None
    target_token: str | None
    latency_seconds: float
    error: str | None = None


@dataclass
class WatchSummary:
    """Summary of a watch session."""

    total_iterations: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    avg_latency: float
    consecutive_failures_at_end: int
    iterations: list[WatchIteration] = field(default_factory=list)


def _build_table(iterations: list[WatchIteration], summary_so_far: dict[str, Any]) -> Table:
    """Build a rich table showing recent iterations and rolling stats."""
    table = Table(title="xpyd-acc watch", show_header=True)
    table.add_column("#", style="dim", width=6)
    table.add_column("Status", width=8)
    table.add_column("Divergence", width=14)
    table.add_column("Latency", width=10)
    table.add_column("Time", width=12)

    for it in iterations[-10:]:
        status = "[green]PASS[/green]" if it.passed else (
            "[yellow]ERROR[/yellow]" if it.error else "[red]FAIL[/red]"
        )
        div = str(it.first_divergence_index) if it.first_divergence_index is not None else "-"
        latency = f"{it.latency_seconds:.2f}s"
        ts = time.strftime("%H:%M:%S", time.localtime(it.timestamp))
        table.add_row(str(it.iteration), status, div, latency, ts)

    total = summary_so_far.get("total", 0)
    passed = summary_so_far.get("passed", 0)
    rate = f"{passed / total * 100:.1f}%" if total > 0 else "-"
    table.add_section()
    table.add_row("", f"[bold]Pass rate: {rate}[/bold]", f"Total: {total}", "", "")

    return table


async def run_watch(
    baseline_url: str,
    target_url: str,
    prompt: str,
    model: str = "default",
    max_tokens: int = 64,
    api_key: str | None = None,
    interval: float = 60.0,
    max_iterations: int | None = None,
    alert_threshold: int | None = None,
    log_path: str | None = None,
    retries: int = 3,
    retry_delay: float = 1.0,
    sampling_params: SamplingParams | None = None,
    no_live: bool = False,
) -> WatchSummary:
    """Run continuous watch loop.

    Args:
        baseline_url: Baseline endpoint URL.
        target_url: Target endpoint URL.
        prompt: Prompt to compare.
        model: Model name.
        max_tokens: Max tokens to generate.
        api_key: API key for endpoints.
        interval: Seconds between iterations.
        max_iterations: Stop after N iterations (None = unlimited).
        alert_threshold: Exit after N consecutive failures (None = disabled).
        log_path: Path to write JSON log.
        retries: HTTP retry count.
        retry_delay: Base retry delay.
        sampling_params: Sampling parameters.
        no_live: Disable live display (for testing / non-TTY).

    Returns:
        WatchSummary with all iteration results.
    """
    console = Console()
    iterations: list[WatchIteration] = []
    passed_count = 0
    failed_count = 0
    error_count = 0
    consecutive_failures = 0
    total_latency = 0.0

    key = api_key or "no-key"
    baseline_collector = LogprobsCollector(base_url=baseline_url, api_key=key, model=model)
    target_collector = LogprobsCollector(base_url=target_url, api_key=key, model=model)
    comparator = LogprobsComparator()

    log_file = None
    if log_path:
        log_file = Path(log_path).open("w")
        log_file.write("[\n")

    try:
        iteration_num = 0
        use_live = not no_live

        context_mgr: Any
        if use_live:
            context_mgr = Live(console=console, refresh_per_second=2)
        else:
            from contextlib import nullcontext
            context_mgr = nullcontext()

        with context_mgr as live:
            while True:
                iteration_num += 1
                start_time = time.time()
                logger.info("Watch iteration %d starting", iteration_num)

                try:
                    baseline_result = await baseline_collector.collect(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        retries=retries,
                        retry_delay=retry_delay,
                        sampling_params=sampling_params,
                    )
                    target_result = await target_collector.collect(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        retries=retries,
                        retry_delay=retry_delay,
                        sampling_params=sampling_params,
                    )

                    report = comparator.compare(baseline_result, target_result)
                    latency = time.time() - start_time

                    if report.match:
                        it = WatchIteration(
                            iteration=iteration_num,
                            timestamp=time.time(),
                            passed=True,
                            first_divergence_index=None,
                            baseline_token=None,
                            target_token=None,
                            latency_seconds=latency,
                        )
                        passed_count += 1
                        consecutive_failures = 0
                    else:
                        div = report.divergence
                        it = WatchIteration(
                            iteration=iteration_num,
                            timestamp=time.time(),
                            passed=False,
                            first_divergence_index=div.token_index if div else None,
                            baseline_token=div.expected_token if div else None,
                            target_token=div.actual_token if div else None,
                            latency_seconds=latency,
                        )
                        failed_count += 1
                        consecutive_failures += 1

                except Exception as e:
                    latency = time.time() - start_time
                    it = WatchIteration(
                        iteration=iteration_num,
                        timestamp=time.time(),
                        passed=False,
                        first_divergence_index=None,
                        baseline_token=None,
                        target_token=None,
                        latency_seconds=latency,
                        error=str(e),
                    )
                    error_count += 1
                    consecutive_failures += 1
                    logger.warning("Watch iteration %d error: %s", iteration_num, e)

                iterations.append(it)
                total_latency += it.latency_seconds

                if log_file:
                    if iteration_num > 1:
                        log_file.write(",\n")
                    json.dump(asdict(it), log_file)
                    log_file.flush()

                if use_live and live is not None:
                    summary_so_far = {"total": iteration_num, "passed": passed_count}
                    live.update(_build_table(iterations, summary_so_far))

                if alert_threshold is not None and consecutive_failures >= alert_threshold:
                    logger.warning(
                        "Alert threshold reached: %d consecutive failures", consecutive_failures
                    )
                    break

                if max_iterations is not None and iteration_num >= max_iterations:
                    logger.info("Max iterations reached: %d", max_iterations)
                    break

                await asyncio.sleep(interval)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Watch interrupted by user")
    finally:
        if log_file:
            log_file.write("\n]\n")
            log_file.close()

    total = len(iterations)
    avg_latency = total_latency / total if total > 0 else 0.0
    pass_rate = passed_count / total if total > 0 else 0.0

    summary = WatchSummary(
        total_iterations=total,
        passed=passed_count,
        failed=failed_count,
        errors=error_count,
        pass_rate=pass_rate,
        avg_latency=avg_latency,
        consecutive_failures_at_end=consecutive_failures,
        iterations=iterations,
    )

    console.print()
    console.print("[bold]Watch Summary[/bold]")
    console.print(f"  Total iterations: {total}")
    console.print(f"  Passed: [green]{passed_count}[/green]")
    console.print(f"  Failed: [red]{failed_count}[/red]")
    console.print(f"  Errors: [yellow]{error_count}[/yellow]")
    console.print(f"  Pass rate: {pass_rate * 100:.1f}%")
    console.print(f"  Avg latency: {avg_latency:.2f}s")

    return summary
