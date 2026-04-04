"""Streaming (SSE) output comparison for two OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import AsyncIterator

import httpx


@dataclass
class StreamToken:
    """A single token received from an SSE stream."""

    index: int
    token: str
    timestamp: float  # monotonic time when received


@dataclass
class StreamingDivergence:
    """First point where two streaming outputs diverge."""

    token_index: int
    expected_token: str
    actual_token: str
    expected_time: float
    actual_time: float


@dataclass
class StreamingComparisonReport:
    """Result of comparing two SSE streaming outputs."""

    baseline_tokens: list[StreamToken]
    target_tokens: list[StreamToken]
    divergence: StreamingDivergence | None
    total_tokens_compared: int
    match: bool
    baseline_total: int
    target_total: int
    elapsed: float  # total wall-clock time


class StreamingCollector:
    """Connect to an OpenAI-compatible SSE endpoint and yield tokens."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "no-key",
        model: str = "default",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def stream(
        self,
        prompt: str,
        max_tokens: int = 64,
        *,
        timeout: float = 60.0,
    ) -> AsyncIterator[StreamToken]:
        """Send prompt and yield tokens as they arrive via SSE.

        Yields:
            StreamToken for each content delta received.
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }

        index = 0
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for token in _parse_sse(resp):
                    yield StreamToken(
                        index=index,
                        token=token,
                        timestamp=time.monotonic(),
                    )
                    index += 1


async def _parse_sse(response: httpx.Response) -> AsyncIterator[str]:
    """Parse SSE stream from an httpx streaming response, yielding content deltas."""
    async for line in response.aiter_lines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        if content:
            yield content


async def collect_stream(
    collector: StreamingCollector,
    prompt: str,
    max_tokens: int = 64,
    *,
    timeout: float = 60.0,
) -> list[StreamToken]:
    """Collect all tokens from a streaming endpoint into a list."""
    tokens: list[StreamToken] = []
    async for token in collector.stream(prompt, max_tokens, timeout=timeout):
        tokens.append(token)
    return tokens


class StreamingComparator:
    """Compare two SSE streaming outputs token-by-token."""

    async def compare(
        self,
        baseline: StreamingCollector,
        target: StreamingCollector,
        prompt: str,
        max_tokens: int = 64,
        *,
        timeout: float = 60.0,
        on_token: None = None,
    ) -> StreamingComparisonReport:
        """Run both streams in parallel and compare tokens as they arrive.

        Args:
            baseline: Collector for the baseline endpoint.
            target: Collector for the target endpoint.
            prompt: Prompt to send to both endpoints.
            max_tokens: Max tokens to generate.
            timeout: HTTP timeout.
            on_token: Optional callback(side: str, token: StreamToken).

        Returns:
            StreamingComparisonReport with comparison results.
        """
        start = time.monotonic()

        # Collect both streams concurrently
        baseline_tokens, target_tokens = await asyncio.gather(
            collect_stream(baseline, prompt, max_tokens, timeout=timeout),
            collect_stream(target, prompt, max_tokens, timeout=timeout),
        )

        elapsed = time.monotonic() - start
        return compare_token_lists(baseline_tokens, target_tokens, elapsed)


def compare_token_lists(
    baseline_tokens: list[StreamToken],
    target_tokens: list[StreamToken],
    elapsed: float = 0.0,
) -> StreamingComparisonReport:
    """Compare two lists of StreamTokens and find first divergence."""
    min_len = min(len(baseline_tokens), len(target_tokens))
    divergence: StreamingDivergence | None = None

    for i in range(min_len):
        bt = baseline_tokens[i]
        tt = target_tokens[i]
        if bt.token != tt.token:
            divergence = StreamingDivergence(
                token_index=i,
                expected_token=bt.token,
                actual_token=tt.token,
                expected_time=bt.timestamp,
                actual_time=tt.timestamp,
            )
            break

    # Length mismatch after all matching tokens
    if divergence is None and len(baseline_tokens) != len(target_tokens):
        idx = min_len
        exp = baseline_tokens[idx].token if idx < len(baseline_tokens) else "<end>"
        act = target_tokens[idx].token if idx < len(target_tokens) else "<end>"
        exp_t = baseline_tokens[idx].timestamp if idx < len(baseline_tokens) else 0.0
        act_t = target_tokens[idx].timestamp if idx < len(target_tokens) else 0.0
        divergence = StreamingDivergence(
            token_index=idx,
            expected_token=exp,
            actual_token=act,
            expected_time=exp_t,
            actual_time=act_t,
        )

    return StreamingComparisonReport(
        baseline_tokens=baseline_tokens,
        target_tokens=target_tokens,
        divergence=divergence,
        total_tokens_compared=min_len,
        match=divergence is None,
        baseline_total=len(baseline_tokens),
        target_total=len(target_tokens),
        elapsed=elapsed,
    )


def format_streaming_report(report: StreamingComparisonReport) -> str:
    """Format a streaming comparison report as human-readable text."""
    lines = [
        "=== Streaming Comparison Report ===",
        f"Baseline tokens: {report.baseline_total}",
        f"Target tokens:   {report.target_total}",
        f"Tokens compared: {report.total_tokens_compared}",
        f"Elapsed:         {report.elapsed:.2f}s",
        "",
    ]

    if report.match:
        lines.append("✅ MATCH — streaming outputs are identical")
    else:
        assert report.divergence is not None
        d = report.divergence
        lines.extend([
            "❌ DIVERGENCE DETECTED",
            f"  Token index:    {d.token_index}",
            f"  Expected token: {d.expected_token!r}",
            f"  Actual token:   {d.actual_token!r}",
        ])

    # Show token-by-token view (first 20 tokens)
    lines.append("")
    lines.append("--- Token-by-token (first 20) ---")
    max_show = min(20, max(report.baseline_total, report.target_total))
    for i in range(max_show):
        bt = report.baseline_tokens[i].token if i < report.baseline_total else "<end>"
        tt = report.target_tokens[i].token if i < report.target_total else "<end>"
        marker = "✓" if bt == tt else "✗"
        lines.append(f"  [{i:3d}] {marker} baseline={bt!r:20s} target={tt!r}")

    return "\n".join(lines)
