"""PD Topology-Aware Testing — discover and test prefill/decode node pairs.

Auto-discovers PD topology from xPyD-proxy and tests all prefill/decode node
pairs to identify which specific combination shows divergence. Critical for
production clusters where overall divergence may be caused by a single bad node.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TopologyNode:
    """A single node in the PD topology."""

    node_id: str
    url: str
    role: str  # "prefill" or "decode"
    model: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "url": self.url,
            "role": self.role,
            "model": self.model,
            "metadata": self.metadata,
        }


@dataclass
class NodePairResult:
    """Result of testing a specific prefill/decode node pair."""

    prefill_node: str
    decode_node: str
    samples_tested: int
    divergent_count: int
    avg_logprob_gap: float | None = None
    first_divergence_indices: list[int] = field(default_factory=list)

    @property
    def divergence_rate(self) -> float:
        if self.samples_tested == 0:
            return 0.0
        return self.divergent_count / self.samples_tested

    @property
    def verdict(self) -> str:
        rate = self.divergence_rate
        if rate == 0.0:
            return "clean"
        if rate < 0.05:
            return "low"
        if rate < 0.2:
            return "moderate"
        return "high"

    def to_dict(self) -> dict:
        return {
            "prefill_node": self.prefill_node,
            "decode_node": self.decode_node,
            "samples_tested": self.samples_tested,
            "divergent_count": self.divergent_count,
            "divergence_rate": round(self.divergence_rate, 4),
            "avg_logprob_gap": (
                round(self.avg_logprob_gap, 6) if self.avg_logprob_gap is not None else None
            ),
            "first_divergence_indices": self.first_divergence_indices,
            "verdict": self.verdict,
        }


@dataclass
class TopologyReport:
    """Full topology scan result."""

    proxy_url: str
    prefill_nodes: list[TopologyNode]
    decode_nodes: list[TopologyNode]
    pair_results: list[NodePairResult]
    total_pairs: int

    @property
    def clean_pairs(self) -> int:
        return sum(1 for r in self.pair_results if r.verdict == "clean")

    @property
    def problematic_pairs(self) -> list[NodePairResult]:
        return [r for r in self.pair_results if r.verdict != "clean"]

    def to_dict(self) -> dict:
        return {
            "proxy_url": self.proxy_url,
            "prefill_nodes": [n.to_dict() for n in self.prefill_nodes],
            "decode_nodes": [n.to_dict() for n in self.decode_nodes],
            "total_pairs": self.total_pairs,
            "clean_pairs": self.clean_pairs,
            "problematic_pairs_count": len(self.problematic_pairs),
            "pair_results": [r.to_dict() for r in self.pair_results],
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def parse_topology(
    data: dict, proxy_url: str = "",
) -> tuple[list[TopologyNode], list[TopologyNode]]:
    """Parse topology response from proxy into node lists.

    Expects a dict with 'instances' or 'nodes' key containing a list
    of objects with at least 'id'/'node_id', 'url', and 'role' fields.
    """
    nodes_data = data.get("instances") or data.get("nodes") or []
    prefill: list[TopologyNode] = []
    decode: list[TopologyNode] = []

    for item in nodes_data:
        node_id = item.get("id") or item.get("node_id") or ""
        url = item.get("url", "")
        role = item.get("role", "").lower()
        model = item.get("model")
        meta = {k: v for k, v in item.items() if k not in ("id", "node_id", "url", "role", "model")}

        node = TopologyNode(
            node_id=str(node_id),
            url=url,
            role=role,
            model=model,
            metadata=meta,
        )

        if role == "prefill":
            prefill.append(node)
        elif role == "decode":
            decode.append(node)

    return prefill, decode


def build_pair_matrix(
    prefill_nodes: list[TopologyNode],
    decode_nodes: list[TopologyNode],
) -> list[tuple[TopologyNode, TopologyNode]]:
    """Generate all prefill × decode node pairs."""
    return [(p, d) for p in prefill_nodes for d in decode_nodes]


def scan_topology(
    prefill_nodes: list[TopologyNode],
    decode_nodes: list[TopologyNode],
    test_fn: callable,
    proxy_url: str = "",
) -> TopologyReport:
    """Scan all node pairs using the provided test function.

    Args:
        prefill_nodes: discovered prefill nodes
        decode_nodes: discovered decode nodes
        test_fn: callable(prefill_node, decode_node) -> NodePairResult
        proxy_url: original proxy URL for reporting
    """
    pairs = build_pair_matrix(prefill_nodes, decode_nodes)
    results: list[NodePairResult] = []

    for p_node, d_node in pairs:
        result = test_fn(p_node, d_node)
        results.append(result)

    return TopologyReport(
        proxy_url=proxy_url,
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
        pair_results=results,
        total_pairs=len(pairs),
    )


def format_topology(report: TopologyReport) -> str:
    """Format topology scan result as terminal-friendly text."""
    lines: list[str] = []
    lines.append(f"Topology Scan: {report.proxy_url}")
    lines.append(
        f"Nodes: {len(report.prefill_nodes)} prefill, "
        f"{len(report.decode_nodes)} decode"
    )
    lines.append(
        f"Pairs tested: {report.total_pairs} | "
        f"Clean: {report.clean_pairs} | "
        f"Problematic: {len(report.problematic_pairs)}"
    )
    lines.append("")

    if not report.pair_results:
        lines.append("No node pairs to test.")
        return "\n".join(lines)

    # Node pair matrix
    hdr = f"{'Prefill → Decode':<30} {'Samples':>8} {'Divergent':>10} {'Rate':>8} {'Verdict':>10}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in report.pair_results:
        pair_label = f"{r.prefill_node} → {r.decode_node}"
        rate_str = f"{r.divergence_rate:.1%}"
        icon = "✅" if r.verdict == "clean" else "⚠️" if r.verdict == "low" else "❌"
        lines.append(
            f"{pair_label:<30} {r.samples_tested:>8} {r.divergent_count:>10} "
            f"{rate_str:>8} {icon} {r.verdict:>7}"
        )

    if report.problematic_pairs:
        lines.append("")
        lines.append("⚠ Problematic pairs:")
        for r in report.problematic_pairs:
            gap_str = f", avg gap={r.avg_logprob_gap:.4f}" if r.avg_logprob_gap is not None else ""
            lines.append(
                f"  {r.prefill_node} → {r.decode_node}: "
                f"{r.divergence_rate:.1%} divergence "
                f"({r.divergent_count}/{r.samples_tested}){gap_str}"
            )

    return "\n".join(lines)
