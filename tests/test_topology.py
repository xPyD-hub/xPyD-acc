"""Tests for PD topology-aware testing."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from xpyd_acc.topology import (
    NodePairResult,
    TopologyNode,
    TopologyReport,
    build_pair_matrix,
    format_topology,
    parse_topology,
    scan_topology,
)

# --- TopologyNode ---


class TestTopologyNode:
    def test_to_dict(self):
        node = TopologyNode(node_id="p1", url="http://p1:8000", role="prefill", model="llama-7b")
        d = node.to_dict()
        assert d["node_id"] == "p1"
        assert d["role"] == "prefill"
        assert d["model"] == "llama-7b"

    def test_default_metadata(self):
        node = TopologyNode(node_id="d1", url="http://d1:8000", role="decode")
        assert node.metadata == {}
        assert node.model is None


# --- NodePairResult ---


class TestNodePairResult:
    def test_divergence_rate_zero(self):
        r = NodePairResult(
            prefill_node="p1", decode_node="d1",
            samples_tested=10, divergent_count=0,
        )
        assert r.divergence_rate == 0.0
        assert r.verdict == "clean"

    def test_divergence_rate_low(self):
        r = NodePairResult(
            prefill_node="p1", decode_node="d1",
            samples_tested=100, divergent_count=3,
        )
        assert r.divergence_rate == pytest.approx(0.03)
        assert r.verdict == "low"

    def test_divergence_rate_moderate(self):
        r = NodePairResult(
            prefill_node="p1", decode_node="d1",
            samples_tested=100, divergent_count=10,
        )
        assert r.divergence_rate == pytest.approx(0.1)
        assert r.verdict == "moderate"

    def test_divergence_rate_high(self):
        r = NodePairResult(
            prefill_node="p1", decode_node="d1",
            samples_tested=10, divergent_count=5,
        )
        assert r.divergence_rate == pytest.approx(0.5)
        assert r.verdict == "high"

    def test_zero_samples(self):
        r = NodePairResult(prefill_node="p1", decode_node="d1", samples_tested=0, divergent_count=0)
        assert r.divergence_rate == 0.0

    def test_to_dict(self):
        r = NodePairResult(
            prefill_node="p1", decode_node="d1",
            samples_tested=20, divergent_count=4,
            avg_logprob_gap=0.123456, first_divergence_indices=[3, 7],
        )
        d = r.to_dict()
        assert d["divergence_rate"] == 0.2
        assert d["avg_logprob_gap"] == 0.123456
        assert d["verdict"] == "high"
        assert d["first_divergence_indices"] == [3, 7]


# --- parse_topology ---


class TestParseTopology:
    def test_parse_instances(self):
        data = {
            "instances": [
                {"id": "p1", "url": "http://p1:8000", "role": "prefill", "model": "llama"},
                {"id": "d1", "url": "http://d1:8000", "role": "decode"},
                {"id": "d2", "url": "http://d2:8000", "role": "decode"},
            ]
        }
        prefill, decode = parse_topology(data)
        assert len(prefill) == 1
        assert len(decode) == 2
        assert prefill[0].node_id == "p1"
        assert prefill[0].model == "llama"

    def test_parse_nodes_key(self):
        data = {
            "nodes": [
                {"node_id": "px", "url": "http://px:8000", "role": "prefill"},
            ]
        }
        prefill, decode = parse_topology(data)
        assert len(prefill) == 1
        assert prefill[0].node_id == "px"

    def test_empty(self):
        prefill, decode = parse_topology({})
        assert prefill == []
        assert decode == []

    def test_extra_metadata(self):
        data = {
            "instances": [
                {"id": "p1", "url": "http://p1:8000", "role": "prefill", "gpu": "A100"},
            ]
        }
        prefill, _ = parse_topology(data)
        assert prefill[0].metadata == {"gpu": "A100"}


# --- build_pair_matrix ---


class TestBuildPairMatrix:
    def test_cartesian_product(self):
        p = [TopologyNode("p1", "http://p1", "prefill"), TopologyNode("p2", "http://p2", "prefill")]
        d = [TopologyNode("d1", "http://d1", "decode"), TopologyNode("d2", "http://d2", "decode")]
        pairs = build_pair_matrix(p, d)
        assert len(pairs) == 4

    def test_empty(self):
        assert build_pair_matrix([], []) == []


# --- scan_topology ---


class TestScanTopology:
    def _make_test_fn(self, divergent_pairs: set[tuple[str, str]] | None = None):
        divergent_pairs = divergent_pairs or set()

        def test_fn(p_node: TopologyNode, d_node: TopologyNode) -> NodePairResult:
            is_div = (p_node.node_id, d_node.node_id) in divergent_pairs
            return NodePairResult(
                prefill_node=p_node.node_id,
                decode_node=d_node.node_id,
                samples_tested=10,
                divergent_count=5 if is_div else 0,
                avg_logprob_gap=0.05 if is_div else None,
            )

        return test_fn

    def test_all_clean(self):
        p = [TopologyNode("p1", "http://p1", "prefill")]
        d = [TopologyNode("d1", "http://d1", "decode")]
        report = scan_topology(p, d, self._make_test_fn(), proxy_url="http://proxy")
        assert report.total_pairs == 1
        assert report.clean_pairs == 1
        assert len(report.problematic_pairs) == 0

    def test_one_bad_pair(self):
        p = [TopologyNode("p1", "http://p1", "prefill")]
        d = [TopologyNode("d1", "http://d1", "decode"), TopologyNode("d2", "http://d2", "decode")]
        report = scan_topology(
            p, d,
            self._make_test_fn(divergent_pairs={("p1", "d2")}),
            proxy_url="http://proxy",
        )
        assert report.total_pairs == 2
        assert report.clean_pairs == 1
        assert len(report.problematic_pairs) == 1
        assert report.problematic_pairs[0].decode_node == "d2"


# --- TopologyReport ---


class TestTopologyReport:
    def test_to_dict(self):
        report = TopologyReport(
            proxy_url="http://proxy",
            prefill_nodes=[TopologyNode("p1", "http://p1", "prefill")],
            decode_nodes=[TopologyNode("d1", "http://d1", "decode")],
            pair_results=[
                NodePairResult("p1", "d1", samples_tested=10, divergent_count=0),
            ],
            total_pairs=1,
        )
        d = report.to_dict()
        assert d["total_pairs"] == 1
        assert d["clean_pairs"] == 1
        assert d["problematic_pairs_count"] == 0

    def test_to_json(self):
        report = TopologyReport(
            proxy_url="http://proxy",
            prefill_nodes=[],
            decode_nodes=[],
            pair_results=[],
            total_pairs=0,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        report.to_json(path)
        data = json.loads(Path(path).read_text())
        assert data["proxy_url"] == "http://proxy"
        Path(path).unlink()


# --- format_topology ---


class TestFormatTopology:
    def test_empty(self):
        report = TopologyReport("http://proxy", [], [], [], 0)
        text = format_topology(report)
        assert "No node pairs" in text

    def test_with_results(self):
        report = TopologyReport(
            proxy_url="http://proxy",
            prefill_nodes=[TopologyNode("p1", "http://p1", "prefill")],
            decode_nodes=[TopologyNode("d1", "http://d1", "decode")],
            pair_results=[
                NodePairResult(
                    "p1", "d1", samples_tested=10,
                    divergent_count=3, avg_logprob_gap=0.05,
                ),
            ],
            total_pairs=1,
        )
        text = format_topology(report)
        assert "p1" in text
        assert "d1" in text
        assert "Problematic" in text

    def test_clean_no_warning(self):
        report = TopologyReport(
            proxy_url="http://proxy",
            prefill_nodes=[TopologyNode("p1", "http://p1", "prefill")],
            decode_nodes=[TopologyNode("d1", "http://d1", "decode")],
            pair_results=[
                NodePairResult("p1", "d1", samples_tested=10, divergent_count=0),
            ],
            total_pairs=1,
        )
        text = format_topology(report)
        assert "Problematic pairs" not in text


# --- CLI integration ---


class TestTopologyCLI:
    def test_topology_scan_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "xpyd_acc.cli", "topology-scan", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--proxy" in result.stdout

    def test_compare_prefill_decode_help(self):
        """Verify --prefill-node and --decode-node flags exist on compare-logprobs."""
        result = subprocess.run(
            [sys.executable, "-m", "xpyd_acc.cli", "compare-logprobs", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--prefill-node" in result.stdout
        assert "--decode-node" in result.stdout
