"""Divergence pattern clustering for batch reports."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DivergenceCluster:
    """A cluster of divergent samples sharing similar divergence patterns."""

    cluster_id: int
    sample_ids: list[str]
    size: int
    centroid: list[float]
    avg_divergence_index: float
    avg_logprob_gap: float
    avg_context_length: float
    representative_sample_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "sample_ids": self.sample_ids,
            "size": self.size,
            "centroid": self.centroid,
            "avg_divergence_index": self.avg_divergence_index,
            "avg_logprob_gap": self.avg_logprob_gap,
            "avg_context_length": self.avg_context_length,
            "representative_sample_id": self.representative_sample_id,
        }


@dataclass
class ClusterResult:
    """Result of divergence pattern clustering."""

    clusters: list[DivergenceCluster]
    k: int
    silhouette_score: float | None
    total_divergent: int
    excluded_matched: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "clusters": [c.to_dict() for c in self.clusters],
            "k": self.k,
            "silhouette_score": self.silhouette_score,
            "total_divergent": self.total_divergent,
            "excluded_matched": self.excluded_matched,
        }

    def to_json(self, path: str | Path) -> None:
        """Export cluster result to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _mean_vec(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    n = len(vectors)
    d = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(d)]


def _kmeans(
    features: list[list[float]], k: int, max_iter: int = 100, seed: int = 42
) -> tuple[list[int], list[list[float]]]:
    """Simple k-means clustering. Returns (assignments, centroids)."""
    rng = random.Random(seed)
    n = len(features)
    # k-means++ init
    centroids: list[list[float]] = [features[rng.randint(0, n - 1)][:]]
    for _ in range(1, k):
        dists = [min(_euclidean(f, c) ** 2 for c in centroids) for f in features]
        total = sum(dists)
        if total == 0:
            centroids.append(features[rng.randint(0, n - 1)][:])
            continue
        r = rng.random() * total
        cumulative = 0.0
        for i, d in enumerate(dists):
            cumulative += d
            if cumulative >= r:
                centroids.append(features[i][:])
                break
        else:
            centroids.append(features[-1][:])

    assignments = [0] * n
    for _ in range(max_iter):
        # assign
        new_assignments = []
        for f in features:
            dists = [_euclidean(f, c) for c in centroids]
            new_assignments.append(dists.index(min(dists)))
        if new_assignments == assignments and _ > 0:
            break
        assignments = new_assignments
        # update centroids
        for ci in range(k):
            members = [features[j] for j in range(n) if assignments[j] == ci]
            if members:
                centroids[ci] = _mean_vec(members)

    return assignments, centroids


def _silhouette_score(features: list[list[float]], assignments: list[int], k: int) -> float:
    """Compute mean silhouette score."""
    n = len(features)
    if n < 2 or k < 2:
        return 0.0

    scores = []
    for i in range(n):
        ci = assignments[i]
        # intra-cluster distance
        same = [j for j in range(n) if assignments[j] == ci and j != i]
        if not same:
            scores.append(0.0)
            continue
        a = sum(_euclidean(features[i], features[j]) for j in same) / len(same)
        # nearest other cluster distance
        b = float("inf")
        for ck in range(k):
            if ck == ci:
                continue
            others = [j for j in range(n) if assignments[j] == ck]
            if others:
                avg_d = sum(_euclidean(features[i], features[j]) for j in others) / len(others)
                b = min(b, avg_d)
        if b == float("inf"):
            scores.append(0.0)
            continue
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _extract_features(sample: dict[str, Any]) -> tuple[str, list[float]]:
    """Extract feature vector from a sample result dict."""
    sample_id = sample.get("sample_id", sample.get("id", "unknown"))
    div_index = sample.get("divergence_index") or 0
    logprob_gap = sample.get("logprob_gap") or 0.0
    context_length = sample.get("context_length") or 0
    baseline_len = len(sample.get("baseline_output", "") or "")
    target_len = len(sample.get("target_output", "") or "")
    length_ratio = target_len / baseline_len if baseline_len > 0 else 1.0
    return str(sample_id), [
        float(div_index), float(logprob_gap), float(context_length), length_ratio,
    ]


def _normalize_features(features: list[list[float]]) -> list[list[float]]:
    """Min-max normalize each dimension."""
    if not features:
        return features
    d = len(features[0])
    mins = [min(f[i] for f in features) for i in range(d)]
    maxs = [max(f[i] for f in features) for i in range(d)]
    result = []
    for f in features:
        row = []
        for i in range(d):
            rng = maxs[i] - mins[i]
            row.append((f[i] - mins[i]) / rng if rng > 0 else 0.0)
        result.append(row)
    return result


def cluster_divergences(
    report: dict[str, Any],
    k: int | None = None,
) -> ClusterResult:
    """Cluster divergent samples from a batch report.

    Args:
        report: Parsed batch report JSON (must have "results" list).
        k: Number of clusters. If None, auto-select via silhouette score.

    Returns:
        ClusterResult with cluster assignments and statistics.
    """
    results = report.get("results", [])
    matched = 0
    divergent_samples: list[dict[str, Any]] = []
    for s in results:
        if s.get("match", False):
            matched += 1
        else:
            divergent_samples.append(s)

    total_div = len(divergent_samples)

    # Edge cases
    if total_div == 0:
        return ClusterResult(
            clusters=[], k=0, silhouette_score=None,
            total_divergent=0, excluded_matched=matched,
        )
    if total_div == 1:
        sid, feat = _extract_features(divergent_samples[0])
        cluster = DivergenceCluster(
            cluster_id=0, sample_ids=[sid], size=1, centroid=feat,
            avg_divergence_index=feat[0], avg_logprob_gap=feat[1],
            avg_context_length=feat[2], representative_sample_id=sid,
        )
        return ClusterResult(
            clusters=[cluster], k=1, silhouette_score=None,
            total_divergent=1, excluded_matched=matched,
        )

    # Extract and normalize features
    ids_and_feats = [_extract_features(s) for s in divergent_samples]
    sample_ids = [x[0] for x in ids_and_feats]
    raw_features = [x[1] for x in ids_and_feats]
    features = _normalize_features(raw_features)

    # Select K
    if k is not None:
        best_k = min(k, total_div)
        assignments, centroids = _kmeans(features, best_k)
        sil = _silhouette_score(features, assignments, best_k) if best_k >= 2 else None
    else:
        max_k = min(10, total_div - 1)
        if max_k < 2:
            best_k = 1
            assignments = [0] * total_div
            centroids = [_mean_vec(features)]
            sil = None
        else:
            best_k = 2
            best_sil = -1.0
            best_assign: list[int] = []
            best_cent: list[list[float]] = []
            for try_k in range(2, max_k + 1):
                a, c = _kmeans(features, try_k)
                s = _silhouette_score(features, a, try_k)
                if s > best_sil:
                    best_sil = s
                    best_k = try_k
                    best_assign = a
                    best_cent = c
            assignments = best_assign
            centroids = best_cent
            sil = best_sil

    # Build clusters
    clusters: list[DivergenceCluster] = []
    for ci in range(best_k):
        members = [i for i in range(total_div) if assignments[i] == ci]
        if not members:
            continue
        member_ids = [sample_ids[i] for i in members]
        member_raw = [raw_features[i] for i in members]
        avg_div = sum(f[0] for f in member_raw) / len(member_raw)
        avg_gap = sum(f[1] for f in member_raw) / len(member_raw)
        avg_ctx = sum(f[2] for f in member_raw) / len(member_raw)
        # Representative: closest to centroid in normalized space
        member_feats = [features[i] for i in members]
        dists = [_euclidean(member_feats[j], centroids[ci]) for j in range(len(members))]
        rep_idx = members[dists.index(min(dists))]
        clusters.append(DivergenceCluster(
            cluster_id=ci,
            sample_ids=member_ids,
            size=len(members),
            centroid=centroids[ci],
            avg_divergence_index=avg_div,
            avg_logprob_gap=avg_gap,
            avg_context_length=avg_ctx,
            representative_sample_id=sample_ids[rep_idx],
        ))

    return ClusterResult(
        clusters=clusters, k=best_k, silhouette_score=sil,
        total_divergent=total_div, excluded_matched=matched,
    )
