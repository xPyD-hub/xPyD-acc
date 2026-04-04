"""Tests for logprobs collection and comparison."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xpyd_acc.logprobs import (
    LogprobsCollector,
    LogprobsComparator,
    LogprobsResult,
    TokenLogprob,
)


def _make_chat_response(tokens: list[tuple[str, float]], model: str = "test-model") -> dict:
    """Build a mock OpenAI chat completion response with logprobs."""
    content = [{"token": t, "logprob": lp} for t, lp in tokens]
    return {
        "model": model,
        "choices": [
            {
                "logprobs": {"content": content},
                "message": {"content": "".join(t for t, _ in tokens)},
            }
        ],
    }


def _make_result(
    endpoint: str, tokens: list[tuple[str, float]], model: str = "m"
) -> LogprobsResult:
    return LogprobsResult(
        endpoint=endpoint,
        model=model,
        tokens=[TokenLogprob(index=i, token=t, logprob=lp) for i, (t, lp) in enumerate(tokens)],
    )


class TestLogprobsCollector:
    @pytest.mark.asyncio
    async def test_collect_parses_response(self) -> None:
        mock_tokens = [("Hello", -0.1), (" world", -0.05)]
        mock_resp = _make_chat_response(mock_tokens, model="gpt-test")

        mock_response = MagicMock()
        mock_response.json.return_value = mock_resp
        mock_response.raise_for_status = lambda: None

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            collector = LogprobsCollector("http://localhost:8000", model="gpt-test")
            result = await collector.collect("Hi")

        assert result.model == "gpt-test"
        assert len(result.tokens) == 2
        assert result.tokens[0].token == "Hello"
        assert result.tokens[0].logprob == pytest.approx(-0.1)
        assert result.tokens[1].token == " world"

    @pytest.mark.asyncio
    async def test_collect_empty_logprobs(self) -> None:
        mock_resp = {
            "model": "m",
            "choices": [{"logprobs": {"content": []}, "message": {"content": ""}}],
        }

        mock_response = MagicMock()
        mock_response.json.return_value = mock_resp
        mock_response.raise_for_status = lambda: None

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            collector = LogprobsCollector("http://localhost:8000")
            result = await collector.collect("Hi")

        assert len(result.tokens) == 0


class TestLogprobsComparator:
    def test_match(self) -> None:
        baseline = _make_result("a", [("Hello", -0.1), (" world", -0.05)])
        target = _make_result("b", [("Hello", -0.1), (" world", -0.05)])

        cmp = LogprobsComparator()
        report = cmp.compare(baseline, target)
        assert report.match is True
        assert report.divergence is None
        assert report.total_tokens_compared == 2

    def test_divergence_at_token(self) -> None:
        baseline = _make_result("a", [("Hello", -0.1), (" world", -0.05)])
        target = _make_result("b", [("Hello", -0.1), (" there", -0.2)])

        cmp = LogprobsComparator()
        report = cmp.compare(baseline, target)
        assert report.match is False
        assert report.divergence is not None
        assert report.divergence.token_index == 1
        assert report.divergence.expected_token == " world"
        assert report.divergence.actual_token == " there"
        assert report.divergence.prob_diff == pytest.approx(0.15)

    def test_divergence_different_lengths(self) -> None:
        baseline = _make_result("a", [("Hello", -0.1), (" world", -0.05)])
        target = _make_result("b", [("Hello", -0.1)])

        cmp = LogprobsComparator()
        report = cmp.compare(baseline, target)
        assert report.match is False
        assert report.divergence is not None
        assert report.divergence.token_index == 1
        assert report.divergence.expected_token == " world"
        assert report.divergence.actual_token == "<end>"

    def test_format_report_match(self) -> None:
        baseline = _make_result("http://a", [("Hi", -0.1)])
        target = _make_result("http://b", [("Hi", -0.1)])
        cmp = LogprobsComparator()
        report = cmp.compare(baseline, target)
        text = cmp.format_report(report)
        assert "MATCH" in text

    def test_format_report_divergence(self) -> None:
        baseline = _make_result("http://a", [("Hi", -0.1)])
        target = _make_result("http://b", [("Yo", -0.3)])
        cmp = LogprobsComparator()
        report = cmp.compare(baseline, target)
        text = cmp.format_report(report)
        assert "DIVERGENCE" in text
        assert "Hi" in text
        assert "Yo" in text
