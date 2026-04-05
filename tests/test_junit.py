"""Tests for JUnit XML export."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from xpyd_acc.batch_compare import BatchReport, MultiTargetBatchReport, SampleResult
from xpyd_acc.junit import multi_report_to_junit, report_to_junit


def _make_result(
    sample_id: str = "s1",
    match: bool = True,
    divergence_index: int | None = None,
    logprob_gap: float | None = None,
    baseline_output: str = "hello",
    target_output: str = "hello",
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        prompt="test prompt",
        baseline_output=baseline_output,
        target_output=target_output,
        exact_match=match,
        first_divergence_index=divergence_index,
        baseline_logprob_at_divergence=None,
        target_logprob_at_divergence=None,
        logprob_gap=logprob_gap,
        classification="match" if match else "likely_bug",
        context_length=10,
    )


def _make_report(results: list[SampleResult] | None = None) -> BatchReport:
    if results is None:
        results = [_make_result()]
    divergent = sum(1 for r in results if not r.exact_match)
    return BatchReport(
        total_samples=len(results),
        divergent_samples=divergent,
        match_samples=len(results) - divergent,
        divergence_rate=divergent / len(results) if results else 0.0,
        results=results,
    )


class TestReportToJunit:
    """Tests for single-target JUnit export."""

    def test_all_passing(self) -> None:
        report = _make_report([_make_result("s1"), _make_result("s2")])
        xml_str = report_to_junit(report, timestamp="2026-01-01T00:00:00Z")
        root = ET.fromstring(xml_str)
        assert root.tag == "testsuite"
        assert root.get("tests") == "2"
        assert root.get("failures") == "0"
        cases = root.findall("testcase")
        assert len(cases) == 2
        for tc in cases:
            assert tc.find("failure") is None

    def test_with_failures(self) -> None:
        results = [
            _make_result("pass1"),
            _make_result("fail1", match=False, divergence_index=5, logprob_gap=0.42,
                         baseline_output="abc", target_output="xyz"),
        ]
        report = _make_report(results)
        xml_str = report_to_junit(report, timestamp="2026-01-01T00:00:00Z")
        root = ET.fromstring(xml_str)
        assert root.get("tests") == "2"
        assert root.get("failures") == "1"
        fail_tc = root.findall("testcase")[1]
        failure = fail_tc.find("failure")
        assert failure is not None
        assert "token index 5" in failure.get("message", "")
        assert "Logprob gap: 0.4200" in failure.text
        assert "Baseline: abc" in failure.text
        assert "Target: xyz" in failure.text

    def test_empty_report(self) -> None:
        report = _make_report([])
        report.total_samples = 0
        report.divergence_rate = 0.0
        xml_str = report_to_junit(report, timestamp="2026-01-01T00:00:00Z")
        root = ET.fromstring(xml_str)
        assert root.get("tests") == "0"
        assert root.get("failures") == "0"
        assert len(root.findall("testcase")) == 0

    def test_all_failures(self) -> None:
        results = [
            _make_result("f1", match=False, divergence_index=0, logprob_gap=1.0),
            _make_result("f2", match=False, divergence_index=3, logprob_gap=0.5),
        ]
        report = _make_report(results)
        xml_str = report_to_junit(report, timestamp="2026-01-01T00:00:00Z")
        root = ET.fromstring(xml_str)
        assert root.get("tests") == "2"
        assert root.get("failures") == "2"
        for tc in root.findall("testcase"):
            assert tc.find("failure") is not None

    def test_xml_declaration(self) -> None:
        report = _make_report()
        xml_str = report_to_junit(report)
        assert xml_str.startswith("<?xml")

    def test_suite_name(self) -> None:
        report = _make_report()
        xml_str = report_to_junit(report, suite_name="custom-suite")
        root = ET.fromstring(xml_str)
        assert root.get("name") == "custom-suite"

    def test_timestamp_in_output(self) -> None:
        report = _make_report()
        xml_str = report_to_junit(report, timestamp="2026-04-05T12:00:00Z")
        root = ET.fromstring(xml_str)
        assert root.get("timestamp") == "2026-04-05T12:00:00Z"

    def test_default_timestamp(self) -> None:
        report = _make_report()
        xml_str = report_to_junit(report)
        root = ET.fromstring(xml_str)
        assert root.get("timestamp") is not None

    def test_truncated_output_in_failure(self) -> None:
        long_output = "x" * 500
        result = _make_result("long", match=False, divergence_index=1,
                              baseline_output=long_output, target_output="short")
        report = _make_report([result])
        xml_str = report_to_junit(report)
        root = ET.fromstring(xml_str)
        failure = root.find(".//failure")
        assert failure is not None
        # Truncated to 200 chars
        assert len(failure.text.split("Baseline: ")[1].split("\n")[0]) == 200

    def test_failure_no_divergence_index(self) -> None:
        result = _make_result("no_idx", match=False, divergence_index=None, logprob_gap=None)
        report = _make_report([result])
        xml_str = report_to_junit(report)
        root = ET.fromstring(xml_str)
        failure = root.find(".//failure")
        assert failure is not None
        assert "Baseline:" in failure.get("message", "")


class TestMultiReportToJunit:
    """Tests for multi-target JUnit export."""

    def test_multi_target(self) -> None:
        r1 = _make_report([_make_result("s1")])
        r2 = _make_report([_make_result("s1", match=False, divergence_index=2)])
        multi = MultiTargetBatchReport(
            baseline_url="http://baseline",
            target_urls=["http://t1", "http://t2"],
            per_target={"http://t1": r1, "http://t2": r2},
            agreement_matrix={},
            total_samples=1,
        )
        xml_str = multi_report_to_junit(multi, timestamp="2026-01-01T00:00:00Z")
        root = ET.fromstring(xml_str)
        assert root.tag == "testsuites"
        suites = root.findall("testsuite")
        assert len(suites) == 2
        assert suites[0].get("name") == "xpyd-acc:http://t1"
        assert suites[0].get("failures") == "0"
        assert suites[1].get("name") == "xpyd-acc:http://t2"
        assert suites[1].get("failures") == "1"

    def test_multi_target_xml_declaration(self) -> None:
        r1 = _make_report()
        multi = MultiTargetBatchReport(
            baseline_url="http://baseline",
            target_urls=["http://t1"],
            per_target={"http://t1": r1},
            agreement_matrix={},
            total_samples=1,
        )
        xml_str = multi_report_to_junit(multi)
        assert xml_str.startswith("<?xml")


class TestCliIntegration:
    """Test --junit flag parsing (unit level)."""

    def test_junit_arg_default_none(self) -> None:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--junit", default=None)
        args = parser.parse_args([])
        assert args.junit is None

    def test_junit_arg_with_value(self) -> None:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--junit", default=None)
        args = parser.parse_args(["--junit", "results.xml"])
        assert args.junit == "results.xml"
