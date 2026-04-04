"""Endpoint latency benchmarking."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

from xpyd_acc.log import get_logger
from xpyd_acc.sampling import SamplingParams

logger = get_logger("benchmark")


@dataclass
class LatencyStats:
    """Aggregated latency statistics."""

    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    total_seconds: float
    errors: int


@dataclass
class BenchmarkResult:
    """Full benchmark result."""

    url: str
    model: str
    requests: int
    concurrency: int
    latencies_ms: list[float] = field(default_factory=list)
    error_count: int = 0
    stats: LatencyStats | None = None

    def compute_stats(self) -> LatencyStats:
        """Compute latency statistics from collected latencies."""
        if not self.latencies_ms:
            self.stats = LatencyStats(
                count=0,
                min_ms=0.0,
                max_ms=0.0,
                mean_ms=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                total_seconds=0.0,
                errors=self.error_count,
            )
            return self.stats

        sorted_lats = sorted(self.latencies_ms)
        n = len(sorted_lats)

        def percentile(p: float) -> float:
            k = (n - 1) * p / 100.0
            f = int(k)
            c = f + 1
            if c >= n:
                return sorted_lats[-1]
            return sorted_lats[f] + (k - f) * (sorted_lats[c] - sorted_lats[f])

        self.stats = LatencyStats(
            count=n,
            min_ms=sorted_lats[0],
            max_ms=sorted_lats[-1],
            mean_ms=statistics.mean(sorted_lats),
            p50_ms=percentile(50),
            p95_ms=percentile(95),
            p99_ms=percentile(99),
            total_seconds=sum(sorted_lats) / 1000.0,
            errors=self.error_count,
        )
        return self.stats

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = {
            "url": self.url,
            "model": self.model,
            "requests": self.requests,
            "concurrency": self.concurrency,
            "error_count": self.error_count,
            "stats": asdict(self.stats) if self.stats else None,
            "latencies_ms": self.latencies_ms,
        }
        return json.dumps(data, indent=2)


async def _send_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: str | None,
    sampling_params: SamplingParams | None,
) -> float | None:
    """Send a single request and return latency in ms, or None on error."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if sampling_params:
        if sampling_params.temperature is not None:
            body["temperature"] = sampling_params.temperature
        if sampling_params.top_p is not None:
            body["top_p"] = sampling_params.top_p
        if sampling_params.seed is not None:
            body["seed"] = sampling_params.seed

    start = time.perf_counter()
    try:
        chat_url = url.rstrip("/") + "/v1/chat/completions"
        resp = await client.post(chat_url, json=body, headers=headers, timeout=120.0)
        resp.raise_for_status()
        elapsed = (time.perf_counter() - start) * 1000.0
        logger.debug("Request completed in %.1fms (status %d)", elapsed, resp.status_code)
        return elapsed
    except Exception as exc:
        logger.warning("Request failed: %s", exc)
        return None


async def run_benchmark(
    url: str,
    prompt: str = "Hello",
    model: str = "default",
    max_tokens: int = 64,
    api_key: str | None = None,
    requests: int = 10,
    concurrency: int = 1,
    sampling_params: SamplingParams | None = None,
    json_path: str | None = None,
) -> BenchmarkResult:
    """Run latency benchmark against an endpoint.

    Args:
        url: Endpoint URL.
        prompt: Prompt to send.
        model: Model name.
        max_tokens: Max tokens to generate.
        api_key: API key.
        requests: Number of requests to send.
        concurrency: Max concurrent requests.
        sampling_params: Sampling parameters.
        json_path: Optional path to export JSON results.

    Returns:
        BenchmarkResult with latency statistics.
    """
    console = Console()
    result = BenchmarkResult(
        url=url,
        model=model,
        requests=requests,
        concurrency=concurrency,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(client: httpx.AsyncClient) -> float | None:
        async with semaphore:
            return await _send_request(
                client, url, model, prompt, max_tokens, api_key, sampling_params,
            )

    console.print(f"[bold]Benchmarking[/bold] {url}")
    console.print(f"  Requests: {requests}, Concurrency: {concurrency}, Model: {model}")
    console.print()

    async with httpx.AsyncClient() as client:
        tasks = [bounded_request(client) for _ in range(requests)]
        latencies = await asyncio.gather(*tasks)

    for lat in latencies:
        if lat is not None:
            result.latencies_ms.append(lat)
        else:
            result.error_count += 1

    stats = result.compute_stats()

    # Rich output
    table = Table(title="Latency Statistics")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Requests", str(stats.count))
    table.add_row("Errors", str(stats.errors))
    table.add_row("Min", f"{stats.min_ms:.1f} ms")
    table.add_row("Max", f"{stats.max_ms:.1f} ms")
    table.add_row("Mean", f"{stats.mean_ms:.1f} ms")
    table.add_row("P50", f"{stats.p50_ms:.1f} ms")
    table.add_row("P95", f"{stats.p95_ms:.1f} ms")
    table.add_row("P99", f"{stats.p99_ms:.1f} ms")
    table.add_row("Total", f"{stats.total_seconds:.2f} s")

    console.print(table)

    if json_path:
        Path(json_path).write_text(result.to_json())
        console.print(f"\nJSON report saved to {json_path}")

    return result
