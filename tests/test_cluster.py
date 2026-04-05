"""Tests for divergence pattern clustering (M49)."""

from __future__ import annotations

import json

from xpyd_acc.cluster import ClusterResult, DivergenceCluster, cluster_divergences


def _make_report(samples: list[dict]) -> dict:
    return {"results": samples}


def _make_sample(
    sid: str, match: bool = False, div_index: int = 0,
    logprob_gap: float = 0.0, context_length: int = 100,
    baseline_output: str = "hello", target_output: str = "world",
) -> dict:
    return {
        "sample_id": sid,
        "match": match,
        "divergence_index": div_index,
        "logprob_gap": logprob_gap,
        "context_length": context_length,
        "baseline_output": baseline_output,
        "target_output": target_output,
    }


class TestClusterDivergences:
    def test_empty_report(self):
        result = cluster_divergences({"results": []})
        assert result.total_divergent == 0
        assert result.clusters == []
        assert result.k == 0

    def test_all_matched(self):
        report = _make_report([
            _make_sample("s1", match=True),
            _make_sample("s2", match=True),
        ])
        result = cluster_divergences(report)
        assert result.total_divergent == 0
        assert result.excluded_matched == 2

    def test_single_divergent(self):
        report = _make_report([
            _make_sample("s1", match=False, div_index=5, logprob_gap=0.1),
            _make_sample("s2", match=True),
        ])
        result = cluster_divergences(report)
        assert result.total_divergent == 1
        assert result.k == 1
        assert len(result.clusters) == 1
        assert result.clusters[0].sample_ids == ["s1"]
        assert result.silhouette_score is None

    def test_two_divergent_auto_k(self):
        report = _make_report([
            _make_sample("s1", match=False, div_index=5, logprob_gap=0.1, context_length=100),
            _make_sample("s2", match=False, div_index=50, logprob_gap=0.9, context_length=1000),
        ])
        result = cluster_divergences(report)
        assert result.total_divergent == 2
        # With only 2 divergent, max_k = min(10, 2-1) = 1, so k=1
        assert result.k == 1

    def test_multiple_clusters_manual_k(self):
        samples = []
        # Group A: early divergence, low gap
        for i in range(5):
            samples.append(_make_sample(f"a{i}", match=False, div_index=2 + i,
                                        logprob_gap=0.01, context_length=50))
        # Group B: late divergence, high gap
        for i in range(5):
            samples.append(_make_sample(f"b{i}", match=False, div_index=100 + i,
                                        logprob_gap=0.9, context_length=2000))
        report = _make_report(samples)
        result = cluster_divergences(report, k=2)
        assert result.k == 2
        assert len(result.clusters) == 2
        assert result.total_divergent == 10
        # Each cluster should have 5 members
        sizes = sorted(c.size for c in result.clusters)
        assert sizes == [5, 5]

    def test_auto_k_selection(self):
        samples = []
        for i in range(6):
            samples.append(_make_sample(f"a{i}", match=False, div_index=5,
                                        logprob_gap=0.01, context_length=100))
        for i in range(6):
            samples.append(_make_sample(f"b{i}", match=False, div_index=200,
                                        logprob_gap=0.95, context_length=5000))
        report = _make_report(samples)
        result = cluster_divergences(report)
        assert result.total_divergent == 12
        assert result.k >= 2
        assert result.silhouette_score is not None
        assert result.silhouette_score > 0

    def test_cluster_result_to_dict(self):
        cluster = DivergenceCluster(
            cluster_id=0, sample_ids=["s1"], size=1,
            centroid=[0.5, 0.5, 0.5, 0.5],
            avg_divergence_index=10.0, avg_logprob_gap=0.5,
            avg_context_length=200.0, representative_sample_id="s1",
        )
        result = ClusterResult(
            clusters=[cluster], k=1, silhouette_score=0.8,
            total_divergent=1, excluded_matched=0,
        )
        d = result.to_dict()
        assert d["k"] == 1
        assert d["silhouette_score"] == 0.8
        assert len(d["clusters"]) == 1
        assert d["clusters"][0]["sample_ids"] == ["s1"]

    def test_json_export(self, tmp_path):
        report = _make_report([
            _make_sample("s1", match=False, div_index=10, logprob_gap=0.5),
        ])
        result = cluster_divergences(report)
        out = tmp_path / "clusters.json"
        result.to_json(out)
        data = json.loads(out.read_text())
        assert data["total_divergent"] == 1
        assert len(data["clusters"]) == 1

    def test_representative_sample(self):
        samples = [
            _make_sample("s1", match=False, div_index=10, logprob_gap=0.5, context_length=100),
            _make_sample("s2", match=False, div_index=12, logprob_gap=0.5, context_length=100),
            _make_sample("s3", match=False, div_index=11, logprob_gap=0.5, context_length=100),
        ]
        report = _make_report(samples)
        result = cluster_divergences(report, k=1)
        assert result.clusters[0].representative_sample_id in ["s1", "s2", "s3"]

    def test_mixed_matched_and_divergent(self):
        samples = [
            _make_sample("m1", match=True),
            _make_sample("m2", match=True),
            _make_sample("d1", match=False, div_index=5),
            _make_sample("d2", match=False, div_index=50),
            _make_sample("d3", match=False, div_index=100),
        ]
        report = _make_report(samples)
        result = cluster_divergences(report)
        assert result.excluded_matched == 2
        assert result.total_divergent == 3

    def test_k_larger_than_samples(self):
        samples = [
            _make_sample("s1", match=False, div_index=5),
            _make_sample("s2", match=False, div_index=10),
        ]
        report = _make_report(samples)
        result = cluster_divergences(report, k=10)
        # k capped at n_samples
        assert result.k == 2


class TestClusterCLI:
    def test_cluster_cli(self, tmp_path):
        """Test cluster CLI integration."""
        from xpyd_acc.cli import main

        report = _make_report([
            _make_sample("s1", match=False, div_index=5, logprob_gap=0.1),
            _make_sample("s2", match=False, div_index=50, logprob_gap=0.9),
            _make_sample("s3", match=False, div_index=5, logprob_gap=0.15),
        ])
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report))
        json_out = tmp_path / "clusters.json"

        main(["cluster", "--input", str(report_path), "--json", str(json_out)])
        assert json_out.exists()
        data = json.loads(json_out.read_text())
        assert data["total_divergent"] == 3
