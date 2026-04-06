"""Tests for hw_baseline module — Hardware Precision Baseline Library."""

from __future__ import annotations

import json
from pathlib import Path

from xpyd_acc.hw_baseline import (
    BaselineDB,
    ClassificationReport,
    DifferenceVerdict,
    HardwareProfile,
    PrecisionRange,
    classify_difference,
    format_classification,
    format_profile,
    format_profile_list,
)

# --- PrecisionRange tests ---


class TestPrecisionRange:
    def test_contains_within(self):
        r = PrecisionRange("max_abs_diff", 0.0, 0.01)
        assert r.contains(0.005)

    def test_contains_at_boundary(self):
        r = PrecisionRange("max_abs_diff", 0.0, 0.01)
        assert r.contains(0.0)
        assert r.contains(0.01)

    def test_contains_outside(self):
        r = PrecisionRange("max_abs_diff", 0.0, 0.01)
        assert not r.contains(0.02)

    def test_round_trip(self):
        r = PrecisionRange("cosine_sim", 0.99, 1.0, "test desc")
        d = r.to_dict()
        r2 = PrecisionRange.from_dict(d)
        assert r2.metric == r.metric
        assert r2.expected_min == r.expected_min
        assert r2.expected_max == r.expected_max
        assert r2.description == r.description


# --- HardwareProfile tests ---


class TestHardwareProfile:
    def test_round_trip(self):
        p = HardwareProfile(
            name="test-profile",
            gpu_arch="A100",
            precision_mode="BF16",
            attention_impl="FlashAttention-v2",
            tp_degree=2,
            ranges=[PrecisionRange("max_abs_diff", 0.0, 0.005)],
            metadata={"note": "test"},
        )
        d = p.to_dict()
        p2 = HardwareProfile.from_dict(d)
        assert p2.name == p.name
        assert p2.gpu_arch == p.gpu_arch
        assert p2.tp_degree == p.tp_degree
        assert len(p2.ranges) == 1
        assert p2.metadata == {"note": "test"}

    def test_get_range_found(self):
        p = HardwareProfile(
            name="t", gpu_arch="A100", precision_mode="BF16",
            attention_impl="FA", tp_degree=1,
            ranges=[PrecisionRange("max_abs_diff", 0.0, 0.01)],
        )
        r = p.get_range("max_abs_diff")
        assert r is not None
        assert r.expected_max == 0.01

    def test_get_range_not_found(self):
        p = HardwareProfile(
            name="t", gpu_arch="A100", precision_mode="BF16",
            attention_impl="FA", tp_degree=1,
        )
        assert p.get_range("nonexistent") is None


# --- classify_difference tests ---


class TestClassifyDifference:
    def _make_profile(self):
        return HardwareProfile(
            name="test",
            gpu_arch="A100",
            precision_mode="BF16",
            attention_impl="FA",
            tp_degree=1,
            ranges=[
                PrecisionRange("max_abs_diff", 0.0, 0.002),
                PrecisionRange("mean_abs_diff", 0.0, 0.0005),
                PrecisionRange("cosine_sim", 0.9995, 1.0),
            ],
        )

    def test_all_expected(self):
        profile = self._make_profile()
        report = classify_difference(profile, {
            "max_abs_diff": 0.001,
            "mean_abs_diff": 0.0003,
            "cosine_sim": 0.9998,
        })
        assert report.overall_classification == "expected"
        assert all(v.classification == "expected" for v in report.verdicts)

    def test_suspicious_diff(self):
        profile = self._make_profile()
        report = classify_difference(profile, {
            "max_abs_diff": 0.003,  # slightly above 0.002 max, ratio < 2
        })
        assert report.verdicts[0].classification == "suspicious"

    def test_likely_bug_diff(self):
        profile = self._make_profile()
        report = classify_difference(profile, {
            "max_abs_diff": 0.01,  # 5x the max -> likely_bug
        })
        assert report.verdicts[0].classification == "likely_bug"

    def test_cosine_sim_below_min(self):
        profile = self._make_profile()
        report = classify_difference(profile, {
            "cosine_sim": 0.98,  # far below 0.9995
        })
        assert report.verdicts[0].classification == "likely_bug"

    def test_cosine_sim_suspicious(self):
        profile = self._make_profile()
        report = classify_difference(profile, {
            "cosine_sim": 0.998,  # below 0.9995 but gap < 0.01
        })
        assert report.verdicts[0].classification == "suspicious"

    def test_unknown_metric(self):
        profile = self._make_profile()
        report = classify_difference(profile, {"unknown_metric": 42.0})
        assert report.verdicts[0].classification == "unknown"

    def test_empty_observations(self):
        profile = self._make_profile()
        report = classify_difference(profile, {})
        assert report.overall_classification == "unknown"
        assert len(report.verdicts) == 0

    def test_overall_worst(self):
        profile = self._make_profile()
        report = classify_difference(profile, {
            "max_abs_diff": 0.001,  # expected
            "cosine_sim": 0.98,  # likely_bug
        })
        assert report.overall_classification == "likely_bug"

    def test_below_min_diff_is_expected(self):
        """Values below min for diff metrics (less error) should be fine."""
        profile = HardwareProfile(
            name="t", gpu_arch="A100", precision_mode="BF16",
            attention_impl="FA", tp_degree=1,
            ranges=[PrecisionRange("max_abs_diff", 0.001, 0.01)],
        )
        report = classify_difference(profile, {"max_abs_diff": 0.0005})
        # Below min for a diff metric → classified as expected (less error is fine)
        assert report.verdicts[0].classification == "expected"


# --- BaselineDB tests ---


class TestBaselineDB:
    def test_builtin_profiles_loaded(self):
        db = BaselineDB()
        names = db.list_profiles()
        assert len(names) >= 6
        assert "a100-bf16-tp1" in names
        assert "h100-fp8-tp4" in names

    def test_get_profile(self):
        db = BaselineDB()
        p = db.get_profile("a100-bf16-tp1")
        assert p is not None
        assert p.gpu_arch == "A100"

    def test_get_profile_not_found(self):
        db = BaselineDB()
        assert db.get_profile("nonexistent") is None

    def test_add_and_remove(self):
        db = BaselineDB()
        custom = HardwareProfile(
            name="custom-test", gpu_arch="Custom", precision_mode="FP32",
            attention_impl="Naive", tp_degree=1,
        )
        db.add_profile(custom)
        assert "custom-test" in db.list_profiles()
        assert db.remove_profile("custom-test") is True
        assert "custom-test" not in db.list_profiles()

    def test_remove_nonexistent(self):
        db = BaselineDB()
        assert db.remove_profile("nonexistent") is False

    def test_export_import_json(self, tmp_path: Path):
        db = BaselineDB()
        path = tmp_path / "profiles.json"
        db.export_json(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert len(data["profiles"]) >= 6

        # Import into fresh DB
        db2 = BaselineDB()
        db2._profiles.clear()
        count = db2.import_json(path)
        assert count >= 6
        assert "a100-bf16-tp1" in db2.list_profiles()

    def test_find_profiles_by_gpu(self):
        db = BaselineDB()
        results = db.find_profiles(gpu_arch="A100")
        assert len(results) >= 2
        assert all(p.gpu_arch == "A100" for p in results)

    def test_find_profiles_by_precision(self):
        db = BaselineDB()
        results = db.find_profiles(precision_mode="FP8")
        assert len(results) >= 1
        assert all(p.precision_mode == "FP8" for p in results)

    def test_find_profiles_by_tp(self):
        db = BaselineDB()
        results = db.find_profiles(tp_degree=4)
        assert len(results) >= 1
        assert all(p.tp_degree == 4 for p in results)

    def test_find_profiles_no_match(self):
        db = BaselineDB()
        results = db.find_profiles(gpu_arch="NonexistentGPU")
        assert results == []


# --- Format tests ---


class TestFormatting:
    def test_format_profile(self):
        p = HardwareProfile(
            name="test", gpu_arch="A100", precision_mode="BF16",
            attention_impl="FA-v2", tp_degree=1,
            ranges=[PrecisionRange("max_abs_diff", 0.0, 0.01, "test range")],
        )
        text = format_profile(p)
        assert "test" in text
        assert "A100" in text
        assert "BF16" in text
        assert "max_abs_diff" in text

    def test_format_profile_list(self):
        db = BaselineDB()
        text = format_profile_list(db)
        assert "a100-bf16-tp1" in text
        assert "GPU" in text

    def test_format_profile_list_empty(self):
        db = BaselineDB()
        db._profiles.clear()
        text = format_profile_list(db)
        assert "No profiles" in text

    def test_format_classification_expected(self):
        report = ClassificationReport(
            profile_name="test",
            verdicts=[DifferenceVerdict(
                metric="max_abs_diff",
                observed_value=0.001,
                expected_range=PrecisionRange("max_abs_diff", 0.0, 0.01),
                classification="expected",
                reasoning="within range",
            )],
        )
        text = format_classification(report)
        assert "✅" in text
        assert "expected" in text

    def test_format_classification_bug(self):
        report = ClassificationReport(
            profile_name="test",
            verdicts=[DifferenceVerdict(
                metric="max_abs_diff",
                observed_value=0.1,
                expected_range=PrecisionRange("max_abs_diff", 0.0, 0.01),
                classification="likely_bug",
                reasoning="outside range",
            )],
        )
        text = format_classification(report)
        assert "❌" in text
        assert "likely_bug" in text


# --- Serialization round-trip ---


class TestSerialization:
    def test_classification_report_to_dict(self):
        report = ClassificationReport(
            profile_name="test",
            verdicts=[DifferenceVerdict(
                metric="m", observed_value=0.5,
                expected_range=PrecisionRange("m", 0.0, 1.0),
                classification="expected", reasoning="ok",
            )],
        )
        d = report.to_dict()
        assert d["profile_name"] == "test"
        assert d["overall_classification"] == "expected"
        assert len(d["verdicts"]) == 1

    def test_verdict_to_dict_no_range(self):
        v = DifferenceVerdict(
            metric="x", observed_value=1.0,
            expected_range=None,
            classification="unknown", reasoning="no range",
        )
        d = v.to_dict()
        assert d["expected_range"] is None
