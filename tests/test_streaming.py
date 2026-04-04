"""Tests for streaming comparison module."""

from __future__ import annotations

import pytest

from xpyd_acc.streaming import (
    StreamingComparator,
    StreamToken,
    _parse_sse,
    compare_token_lists,
    format_streaming_report,
)


def _make_token(index: int, token: str, ts: float = 0.0) -> StreamToken:
    return StreamToken(index=index, token=token, timestamp=ts)


class TestCompareTokenLists:
    """Test the compare_token_lists function."""

    def test_identical_tokens(self) -> None:
        baseline = [_make_token(0, "Hello"), _make_token(1, " world")]
        target = [_make_token(0, "Hello"), _make_token(1, " world")]
        report = compare_token_lists(baseline, target, elapsed=1.0)
        assert report.match is True
        assert report.divergence is None
        assert report.total_tokens_compared == 2
        assert report.baseline_total == 2
        assert report.target_total == 2
        assert report.elapsed == 1.0

    def test_divergence_at_index(self) -> None:
        baseline = [_make_token(0, "Hello"), _make_token(1, " world")]
        target = [_make_token(0, "Hello"), _make_token(1, " earth")]
        report = compare_token_lists(baseline, target)
        assert report.match is False
        assert report.divergence is not None
        assert report.divergence.token_index == 1
        assert report.divergence.expected_token == " world"
        assert report.divergence.actual_token == " earth"

    def test_length_mismatch_baseline_longer(self) -> None:
        baseline = [_make_token(0, "a"), _make_token(1, "b"), _make_token(2, "c")]
        target = [_make_token(0, "a"), _make_token(1, "b")]
        report = compare_token_lists(baseline, target)
        assert report.match is False
        assert report.divergence is not None
        assert report.divergence.token_index == 2
        assert report.divergence.expected_token == "c"
        assert report.divergence.actual_token == "<end>"

    def test_length_mismatch_target_longer(self) -> None:
        baseline = [_make_token(0, "a")]
        target = [_make_token(0, "a"), _make_token(1, "b")]
        report = compare_token_lists(baseline, target)
        assert report.match is False
        assert report.divergence is not None
        assert report.divergence.token_index == 1
        assert report.divergence.expected_token == "<end>"
        assert report.divergence.actual_token == "b"

    def test_both_empty(self) -> None:
        report = compare_token_lists([], [])
        assert report.match is True
        assert report.divergence is None
        assert report.total_tokens_compared == 0

    def test_early_divergence(self) -> None:
        baseline = [_make_token(0, "x"), _make_token(1, "y")]
        target = [_make_token(0, "z"), _make_token(1, "y")]
        report = compare_token_lists(baseline, target)
        assert report.divergence is not None
        assert report.divergence.token_index == 0


class TestFormatStreamingReport:
    """Test the format_streaming_report function."""

    def test_match_report(self) -> None:
        tokens = [_make_token(0, "Hi")]
        report = compare_token_lists(tokens, tokens[:], elapsed=0.5)
        text = format_streaming_report(report)
        assert "MATCH" in text
        assert "0.50s" in text

    def test_divergence_report(self) -> None:
        baseline = [_make_token(0, "a"), _make_token(1, "b")]
        target = [_make_token(0, "a"), _make_token(1, "c")]
        report = compare_token_lists(baseline, target, elapsed=1.0)
        text = format_streaming_report(report)
        assert "DIVERGENCE" in text
        assert "'b'" in text
        assert "'c'" in text

    def test_token_by_token_view(self) -> None:
        tokens = [_make_token(i, f"t{i}") for i in range(5)]
        report = compare_token_lists(tokens, tokens[:], elapsed=0.1)
        text = format_streaming_report(report)
        assert "Token-by-token" in text
        assert "✓" in text


class TestParseSSE:
    """Test SSE parsing."""

    @pytest.mark.asyncio
    async def test_parse_sse_tokens(self) -> None:
        """Test parsing SSE lines from a mock response."""

        from unittest.mock import MagicMock

        # Build mock SSE lines
        lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            "",
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            "",
            "data: [DONE]",
        ]

        mock_response = MagicMock()

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_response.aiter_lines = _aiter_lines

        tokens = []
        async for token in _parse_sse(mock_response):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_parse_sse_empty_delta(self) -> None:
        """Deltas without content are skipped."""
        from unittest.mock import MagicMock

        lines = [
            'data: {"choices": [{"delta": {"role": "assistant"}}]}',
            'data: {"choices": [{"delta": {"content": "ok"}}]}',
            "data: [DONE]",
        ]

        mock_response = MagicMock()

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_response.aiter_lines = _aiter_lines

        tokens = []
        async for token in _parse_sse(mock_response):
            tokens.append(token)

        assert tokens == ["ok"]

    @pytest.mark.asyncio
    async def test_parse_sse_malformed_json(self) -> None:
        """Malformed JSON lines are skipped."""
        from unittest.mock import MagicMock

        lines = [
            "data: {invalid json}",
            'data: {"choices": [{"delta": {"content": "hi"}}]}',
            "data: [DONE]",
        ]

        mock_response = MagicMock()

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_response.aiter_lines = _aiter_lines

        tokens = []
        async for token in _parse_sse(mock_response):
            tokens.append(token)

        assert tokens == ["hi"]


class TestStreamingCollector:
    """Test StreamingCollector initialization."""

    def test_collector_init(self) -> None:
        from xpyd_acc.streaming import StreamingCollector

        c = StreamingCollector("http://localhost:8000", api_key="key", model="llama")
        assert c.base_url == "http://localhost:8000"
        assert c.api_key == "key"
        assert c.model == "llama"

    def test_trailing_slash_stripped(self) -> None:
        from xpyd_acc.streaming import StreamingCollector

        c = StreamingCollector("http://localhost:8000/")
        assert c.base_url == "http://localhost:8000"


class TestStreamingComparator:
    """Test StreamingComparator with mocked collectors."""

    @pytest.mark.asyncio
    async def test_compare_matching_streams(self) -> None:
        """Two identical streams should report match."""
        from unittest.mock import AsyncMock, patch

        from xpyd_acc.streaming import StreamingCollector

        tokens = [_make_token(0, "Hello"), _make_token(1, " world")]

        with patch("xpyd_acc.streaming.collect_stream", new_callable=AsyncMock) as mock_cs:
            mock_cs.return_value = tokens

            comparator = StreamingComparator()
            baseline = StreamingCollector("http://a")
            target = StreamingCollector("http://b")
            report = await comparator.compare(baseline, target, "test prompt")

        assert report.match is True

    @pytest.mark.asyncio
    async def test_compare_divergent_streams(self) -> None:
        """Two different streams should detect divergence."""
        from unittest.mock import patch

        from xpyd_acc.streaming import StreamingCollector

        baseline_tokens = [_make_token(0, "Hello"), _make_token(1, " world")]
        target_tokens = [_make_token(0, "Hello"), _make_token(1, " earth")]

        call_count = 0

        async def mock_collect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return baseline_tokens
            return target_tokens

        with patch("xpyd_acc.streaming.collect_stream", side_effect=mock_collect):
            comparator = StreamingComparator()
            baseline = StreamingCollector("http://a")
            target = StreamingCollector("http://b")
            report = await comparator.compare(baseline, target, "test")

        assert report.match is False
        assert report.divergence is not None
        assert report.divergence.token_index == 1


class TestCLIIntegration:
    """Test CLI integration for compare-streaming subcommand."""

    def test_compare_streaming_help(self) -> None:
        """Verify compare-streaming subcommand exists."""
        from xpyd_acc.cli import main

        with pytest.raises(SystemExit):
            main(["compare-streaming", "--help"])
