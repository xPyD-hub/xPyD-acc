"""Reproducibility score — multi-run consistency measurement."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Callable

import httpx

from .log import get_logger
from .retry import retry_async

logger = get_logger(__name__)


@dataclass
class ReproducibilityResult:
    """Result of reproducibility measurement for one endpoint."""

    url: str
    runs: int
    outputs: list[str]
    unique_count: int
    majority_fraction: float
    avg_pairwise_distance: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


@dataclass
class ReproducibilityReport:
    """Report comparing reproducibility of one or two endpoints."""

    baseline: ReproducibilityResult | None = None
    target: ReproducibilityResult | None = None
    single: ReproducibilityResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        d: dict[str, Any] = {}
        if self.single is not None:
            d["single"] = self.single.to_dict()
        if self.baseline is not None:
            d["baseline"] = self.baseline.to_dict()
        if self.target is not None:
            d["target"] = self.target.to_dict()
        return d

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[len(b)]


def _compute_result(url: str, outputs: list[str]) -> ReproducibilityResult:
    """Compute reproducibility metrics from a list of outputs."""
    n = len(outputs)
    unique = list(set(outputs))
    unique_count = len(unique)

    # Majority fraction
    from collections import Counter

    counts = Counter(outputs)
    majority_fraction = counts.most_common(1)[0][1] / n if n > 0 else 0.0

    # Average pairwise edit distance
    if n < 2:
        avg_dist = 0.0
    else:
        total_dist = sum(
            _edit_distance(a, b) for a, b in combinations(outputs, 2)
        )
        pair_count = n * (n - 1) // 2
        avg_dist = total_dist / pair_count

    return ReproducibilityResult(
        url=url,
        runs=n,
        outputs=outputs,
        unique_count=unique_count,
        majority_fraction=round(majority_fraction, 4),
        avg_pairwise_distance=round(avg_dist, 4),
    )


async def _collect_outputs(
    url: str,
    prompt: str,
    model: str,
    runs: int,
    *,
    api_key: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 120.0,
) -> list[str]:
    """Send prompt N times and collect outputs."""
    outputs: list[str] = []

    async def _single_request() -> str:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if seed is not None:
            body["seed"] = seed

        async def _do_request() -> str:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{url}/v1/chat/completions",
                    json=body,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

        result = await retry_async(_do_request, retries=retries, base_delay=retry_delay)
        return result.value

    tasks = [_single_request() for _ in range(runs)]
    outputs = await asyncio.gather(*tasks)
    return list(outputs)


async def run_reproducibility(
    *,
    url: str | None = None,
    baseline_url: str | None = None,
    target_url: str | None = None,
    prompt: str,
    model: str,
    runs: int = 5,
    api_key: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 120.0,
    on_progress: Callable[[str, int], None] | None = None,
) -> ReproducibilityReport:
    """Run reproducibility measurement.

    Either ``url`` (single endpoint) or both ``baseline_url`` and
    ``target_url`` (dual mode) must be provided.
    """
    kwargs = dict(
        prompt=prompt,
        model=model,
        runs=runs,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        retries=retries,
        retry_delay=retry_delay,
        timeout=timeout,
    )

    if url is not None:
        outputs = await _collect_outputs(url, **kwargs)
        result = _compute_result(url, outputs)
        return ReproducibilityReport(single=result)

    if baseline_url is not None and target_url is not None:
        baseline_out, target_out = await asyncio.gather(
            _collect_outputs(baseline_url, **kwargs),
            _collect_outputs(target_url, **kwargs),
        )
        return ReproducibilityReport(
            baseline=_compute_result(baseline_url, baseline_out),
            target=_compute_result(target_url, target_out),
        )

    msg = "Either 'url' or both 'baseline_url' and 'target_url' required"
    raise ValueError(msg)


def format_reproducibility(report: ReproducibilityReport) -> str:
    """Format reproducibility report for terminal output."""
    lines: list[str] = []
    lines.append("Reproducibility Report")
    lines.append("=" * 40)

    def _fmt(label: str, r: ReproducibilityResult) -> None:
        lines.append(f"\n{label}: {r.url}")
        lines.append(f"  Runs:                    {r.runs}")
        lines.append(f"  Unique outputs:          {r.unique_count}")
        lines.append(f"  Majority fraction:       {r.majority_fraction:.2%}")
        lines.append(f"  Avg pairwise distance:   {r.avg_pairwise_distance:.2f}")

    if report.single is not None:
        _fmt("Endpoint", report.single)
    if report.baseline is not None:
        _fmt("Baseline", report.baseline)
    if report.target is not None:
        _fmt("Target", report.target)

    return "\n".join(lines)
