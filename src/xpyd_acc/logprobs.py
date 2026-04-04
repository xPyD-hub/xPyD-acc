"""Logprobs collection and comparison for two OpenAI-compatible endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx


@dataclass
class TokenLogprob:
    """A single token with its log probability."""

    index: int
    token: str
    logprob: float


@dataclass
class LogprobsResult:
    """Logprobs collected from one endpoint."""

    endpoint: str
    model: str
    tokens: list[TokenLogprob] = field(default_factory=list)


@dataclass
class DivergencePoint:
    """First point where two logprob sequences diverge."""

    token_index: int
    expected_token: str
    actual_token: str
    expected_logprob: float
    actual_logprob: float
    prob_diff: float


@dataclass
class ComparisonReport:
    """Result of comparing logprobs from two endpoints."""

    baseline: LogprobsResult
    target: LogprobsResult
    divergence: DivergencePoint | None
    total_tokens_compared: int
    match: bool


class LogprobsCollector:
    """Send a prompt to an OpenAI-compatible endpoint and collect per-token logprobs."""

    def __init__(self, base_url: str, api_key: str = "no-key", model: str = "default") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def collect(
        self,
        prompt: str,
        max_tokens: int = 64,
        *,
        timeout: float = 30.0,
        retries: int = 3,
        retry_delay: float = 1.0,
    ) -> LogprobsResult:
        """Send prompt and collect logprobs from the completions endpoint."""
        from xpyd_acc.retry import retry_async

        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": 1,
        }

        async def _do_request() -> dict:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()

        data = await retry_async(_do_request, retries=retries, base_delay=retry_delay)
        return self._parse_response(data)

    def _parse_response(self, data: dict) -> LogprobsResult:
        """Parse OpenAI-compatible chat completion response with logprobs."""
        choice = data["choices"][0]
        model = data.get("model", self.model)
        logprobs_content = choice.get("logprobs", {}).get("content", [])

        tokens: list[TokenLogprob] = []
        for i, entry in enumerate(logprobs_content):
            tokens.append(
                TokenLogprob(
                    index=i,
                    token=entry["token"],
                    logprob=entry["logprob"],
                )
            )

        return LogprobsResult(endpoint=self.base_url, model=model, tokens=tokens)


class LogprobsComparator:
    """Compare logprob sequences from two endpoints and find divergence."""

    def __init__(self, token_mismatch_threshold: float = 0.0) -> None:
        self.token_mismatch_threshold = token_mismatch_threshold

    def compare(self, baseline: LogprobsResult, target: LogprobsResult) -> ComparisonReport:
        """Compare two logprob sequences. Find first divergence point."""
        min_len = min(len(baseline.tokens), len(target.tokens))
        divergence: DivergencePoint | None = None

        for i in range(min_len):
            bt = baseline.tokens[i]
            tt = target.tokens[i]

            if bt.token != tt.token:
                divergence = DivergencePoint(
                    token_index=i,
                    expected_token=bt.token,
                    actual_token=tt.token,
                    expected_logprob=bt.logprob,
                    actual_logprob=tt.logprob,
                    prob_diff=abs(bt.logprob - tt.logprob),
                )
                break

        # If tokens all match but lengths differ, that's also a divergence
        if divergence is None and len(baseline.tokens) != len(target.tokens):
            idx = min_len
            expected = baseline.tokens[idx].token if idx < len(baseline.tokens) else "<end>"
            actual = target.tokens[idx].token if idx < len(target.tokens) else "<end>"
            expected_lp = baseline.tokens[idx].logprob if idx < len(baseline.tokens) else 0.0
            actual_lp = target.tokens[idx].logprob if idx < len(target.tokens) else 0.0
            divergence = DivergencePoint(
                token_index=idx,
                expected_token=expected,
                actual_token=actual,
                expected_logprob=expected_lp,
                actual_logprob=actual_lp,
                prob_diff=abs(expected_lp - actual_lp),
            )

        return ComparisonReport(
            baseline=baseline,
            target=target,
            divergence=divergence,
            total_tokens_compared=min_len,
            match=divergence is None,
        )

    def format_report(self, report: ComparisonReport) -> str:
        """Format a comparison report as human-readable text."""
        lines = [
            "=== Logprobs Comparison Report ===",
            f"Baseline: {report.baseline.endpoint} ({report.baseline.model})",
            f"Target:   {report.target.endpoint} ({report.target.model})",
            f"Tokens compared: {report.total_tokens_compared}",
            "",
        ]

        if report.match:
            lines.append("✅ MATCH — no divergence found")
        else:
            assert report.divergence is not None
            d = report.divergence
            lines.extend([
                "❌ DIVERGENCE DETECTED",
                f"  Token index:    {d.token_index}",
                f"  Expected token: {d.expected_token!r}",
                f"  Actual token:   {d.actual_token!r}",
                f"  Expected logprob: {d.expected_logprob:.6f}",
                f"  Actual logprob:   {d.actual_logprob:.6f}",
                f"  Prob diff:        {d.prob_diff:.6f}",
            ])

        return "\n".join(lines)
