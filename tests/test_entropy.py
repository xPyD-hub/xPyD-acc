"""Tests for entropy analysis module."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from xpyd_acc.entropy import (
    EntropyComparison,
    EntropyStats,
    entropy_at_divergence,
    entropy_stats,
    format_entropy_comparison,
    format_entropy_stats,
    load_logprobs_file,
    sequence_entropy,
    token_entropy,
)

# ---------------------------------------------------------------------------
# token_entropy tests
# ---------------------------------------------------------------------------


def test_token_entropy_empty():
    assert token_entropy({}) == 0.0


def test_token_entropy_single_token():
    assert token_entropy({"hello": -0.1}) == 0.0


def test_token_entropy_uniform_two():
    """Two equally likely tokens → ln(2)."""
    lp = math.log(0.5)
    ent = token_entropy({"a": lp, "b": lp})
    assert abs(ent - math.log(2)) < 1e-9


def test_token_entropy_skewed():
    """One dominant token → low entropy."""
    lp_high = math.log(0.99)
    lp_low = math.log(0.01)
    ent = token_entropy({"a": lp_high, "b": lp_low})
    assert ent < math.log(2)
    assert ent > 0.0


def test_token_entropy_three_uniform():
    lp = math.log(1.0 / 3.0)
    ent = token_entropy({"a": lp, "b": lp, "c": lp})
    assert abs(ent - math.log(3)) < 1e-9


def test_token_entropy_normalizes():
    """Logprobs that don't sum to 1 are normalized."""
    ent = token_entropy({"a": -1.0, "b": -1.0})
    # Both equal → should still be ln(2)
    assert abs(ent - math.log(2)) < 1e-9


# ---------------------------------------------------------------------------
# sequence_entropy tests
# ---------------------------------------------------------------------------


def test_sequence_entropy_empty():
    assert sequence_entropy([]) == []


def test_sequence_entropy_basic():
    lps = [
        {"a": math.log(0.5), "b": math.log(0.5)},
        {"a": math.log(0.9), "b": math.log(0.1)},
    ]
    result = sequence_entropy(lps)
    assert len(result) == 2
    assert abs(result[0] - math.log(2)) < 1e-9
    assert result[1] < result[0]


# ---------------------------------------------------------------------------
# entropy_stats tests
# ---------------------------------------------------------------------------


def test_entropy_stats_empty():
    stats = entropy_stats([])
    assert stats.count == 0
    assert stats.mean == 0.0


def test_entropy_stats_single():
    stats = entropy_stats([1.5])
    assert stats.count == 1
    assert stats.min == 1.5
    assert stats.max == 1.5
    assert stats.mean == 1.5


def test_entropy_stats_multiple():
    vals = [0.1, 0.5, 0.3, 0.9, 0.2]
    stats = entropy_stats(vals)
    assert stats.count == 5
    assert stats.min == 0.1
    assert stats.max == 0.9


def test_entropy_stats_to_dict():
    stats = EntropyStats(min=0.1, max=0.9, mean=0.5, median=0.4, p95=0.8, count=10)
    d = stats.to_dict()
    assert d["count"] == 10
    assert d["min"] == 0.1


# ---------------------------------------------------------------------------
# entropy_at_divergence tests
# ---------------------------------------------------------------------------


def test_entropy_at_divergence_basic():
    bl = [{"a": math.log(0.5), "b": math.log(0.5)}] * 10
    tg = [{"a": math.log(0.9), "b": math.log(0.1)}] * 10
    comp = entropy_at_divergence(bl, tg, 5, context_window=2)
    assert comp.divergence_index == 5
    assert comp.baseline_entropy > comp.target_entropy
    assert comp.delta < 0
    assert comp.context_start == 3
    assert len(comp.context_baseline) == 5  # positions 3,4,5,6,7


def test_entropy_at_divergence_start():
    bl = [{"a": math.log(0.5), "b": math.log(0.5)}] * 3
    tg = [{"a": math.log(0.5), "b": math.log(0.5)}] * 3
    comp = entropy_at_divergence(bl, tg, 0, context_window=5)
    assert comp.context_start == 0
    assert len(comp.context_baseline) == 3


def test_entropy_comparison_to_dict():
    comp = EntropyComparison(
        divergence_index=5,
        baseline_entropy=0.5,
        target_entropy=0.3,
        delta=-0.2,
        context_baseline=[0.5, 0.5, 0.5],
        context_target=[0.3, 0.3, 0.3],
        context_start=3,
    )
    d = comp.to_dict()
    assert d["divergence_index"] == 5
    assert d["delta"] == -0.2


# ---------------------------------------------------------------------------
# Format tests
# ---------------------------------------------------------------------------


def test_format_entropy_stats():
    stats = EntropyStats(min=0.1, max=0.9, mean=0.5, median=0.4, p95=0.8, count=10)
    text = format_entropy_stats(stats)
    assert "0.1000" in text
    assert "Count" in text


def test_format_entropy_comparison():
    comp = EntropyComparison(
        divergence_index=5, baseline_entropy=0.5, target_entropy=0.3,
        delta=-0.2, context_baseline=[0.5], context_target=[0.3],
        context_start=5,
    )
    text = format_entropy_comparison(comp)
    assert "index 5" in text
    assert "Delta" in text


# ---------------------------------------------------------------------------
# File loading tests
# ---------------------------------------------------------------------------


def test_load_logprobs_file(tmp_path: Path):
    data = [{"a": -0.5, "b": -1.2}, {"c": -0.1}]
    f = tmp_path / "lp.json"
    f.write_text(json.dumps(data))
    loaded = load_logprobs_file(str(f))
    assert len(loaded) == 2
    assert loaded[0]["a"] == -0.5


def test_load_logprobs_file_invalid(tmp_path: Path):
    f = tmp_path / "bad.json"
    f.write_text('{"not": "a list"}')
    with pytest.raises(ValueError, match="Expected JSON array"):
        load_logprobs_file(str(f))


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def test_cli_entropy_help(capsys):
    from xpyd_acc.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["entropy", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "entropy" in captured.out or "baseline-logprobs" in captured.out


def test_cli_entropy_baseline_only(tmp_path: Path, capsys):
    from xpyd_acc.cli import main

    data = [{"a": math.log(0.5), "b": math.log(0.5)}] * 5
    f = tmp_path / "bl.json"
    f.write_text(json.dumps(data))

    main(["entropy", "--baseline-logprobs", str(f)])
    captured = capsys.readouterr()
    assert "Entropy Statistics" in captured.out


def test_cli_entropy_with_target_and_json(tmp_path: Path, capsys):
    from xpyd_acc.cli import main

    bl = [{"a": math.log(0.5), "b": math.log(0.5)}] * 5
    tg = [{"a": math.log(0.9), "b": math.log(0.1)}] * 5
    bf = tmp_path / "bl.json"
    tf = tmp_path / "tg.json"
    bf.write_text(json.dumps(bl))
    tf.write_text(json.dumps(tg))

    out = tmp_path / "result.json"
    main([
        "entropy",
        "--baseline-logprobs", str(bf),
        "--target-logprobs", str(tf),
        "--divergence-index", "2",
        "--json", str(out),
    ])

    captured = capsys.readouterr()
    assert "Baseline" in captured.out
    assert "Target" in captured.out
    assert "divergence" in captured.out.lower()

    result = json.loads(out.read_text())
    assert "baseline_stats" in result
    assert "target_stats" in result
    assert "comparison" in result
