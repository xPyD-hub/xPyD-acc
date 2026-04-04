"""Automated diagnostic pipeline for PD disaggregation accuracy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum

from xpyd_acc.kvcache import KVCacheComparator, KVCacheLoader
from xpyd_acc.logprobs import (
    ComparisonReport,
    LogprobsCollector,
    LogprobsComparator,
)


class StepStatus(str, Enum):
    """Status of a diagnostic step."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class DiagnosticStep:
    """Result of a single diagnostic step."""

    name: str
    description: str
    status: StepStatus
    detail: str = ""
    data: dict | None = None


@dataclass
class DiagnosticReport:
    """Full diagnostic report across all steps."""

    steps: list[DiagnosticStep] = field(default_factory=list)
    overall_pass: bool = True

    def to_json(self) -> str:
        """Serialize report to JSON string."""
        return json.dumps(asdict(self), indent=2, default=str)


class DiagnosticPipeline:
    """Run all diagnostic checks in sequence."""

    def __init__(
        self,
        baseline_url: str,
        target_url: str,
        prompt: str,
        *,
        model: str = "default",
        api_key: str = "no-key",
        max_tokens: int = 64,
        kv_baseline_path: str | None = None,
        kv_target_path: str | None = None,
        kv_max_abs_threshold: float = 1e-3,
        kv_cosine_threshold: float = 0.999,
        sampling_params: object | None = None,
    ) -> None:
        self.baseline_url = baseline_url
        self.target_url = target_url
        self.prompt = prompt
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.kv_baseline_path = kv_baseline_path
        self.kv_target_path = kv_target_path
        self.kv_max_abs_threshold = kv_max_abs_threshold
        self.kv_cosine_threshold = kv_cosine_threshold
        self.sampling_params = sampling_params

    async def run(self) -> DiagnosticReport:
        """Execute all diagnostic steps and return the report."""
        report = DiagnosticReport()

        # Step 1: First-token logprobs comparison
        step1 = await self._step_first_token()
        report.steps.append(step1)
        if step1.status == StepStatus.FAIL:
            report.overall_pass = False

        # Step 2: KV cache comparison (if dumps provided)
        step2 = self._step_kv_cache()
        report.steps.append(step2)
        if step2.status == StepStatus.FAIL:
            report.overall_pass = False

        # Step 3: Full sequence logprobs comparison
        step3 = await self._step_full_sequence()
        report.steps.append(step3)
        if step3.status == StepStatus.FAIL:
            report.overall_pass = False

        return report

    async def _step_first_token(self) -> DiagnosticStep:
        """Step 1: Compare first token between baseline and target."""
        try:
            comparison = await self._collect_and_compare(max_tokens=1)
        except Exception as e:
            return DiagnosticStep(
                name="first_token",
                description="Baseline vs target: first token match",
                status=StepStatus.FAIL,
                detail=f"Error: {e}",
            )

        if comparison.match:
            return DiagnosticStep(
                name="first_token",
                description="Baseline vs target: first token match",
                status=StepStatus.PASS,
                detail="First token matches between endpoints",
            )

        d = comparison.divergence
        detail = (
            f"First token diverges: expected {d.expected_token!r}, "
            f"got {d.actual_token!r} (logprob diff: {d.prob_diff:.6f})"
        ) if d else "No tokens generated"

        return DiagnosticStep(
            name="first_token",
            description="Baseline vs target: first token match",
            status=StepStatus.FAIL,
            detail=detail,
        )

    def _step_kv_cache(self) -> DiagnosticStep:
        """Step 2: Compare KV cache dumps if available."""
        if not self.kv_baseline_path or not self.kv_target_path:
            return DiagnosticStep(
                name="kv_cache",
                description="KV cache numerical accuracy",
                status=StepStatus.SKIP,
                detail="No KV cache dumps provided (use --kv-baseline and --kv-target)",
            )

        try:
            baseline = KVCacheLoader.load(self.kv_baseline_path)
            target = KVCacheLoader.load(self.kv_target_path)
        except Exception as e:
            return DiagnosticStep(
                name="kv_cache",
                description="KV cache numerical accuracy",
                status=StepStatus.FAIL,
                detail=f"Error loading KV cache: {e}",
            )

        comparator = KVCacheComparator(
            max_abs_threshold=self.kv_max_abs_threshold,
            cosine_threshold=self.kv_cosine_threshold,
        )
        kv_report = comparator.compare(
            baseline, target,
            baseline_path=self.kv_baseline_path,
            target_path=self.kv_target_path,
        )

        if kv_report.match:
            return DiagnosticStep(
                name="kv_cache",
                description="KV cache numerical accuracy",
                status=StepStatus.PASS,
                detail=f"All {len(kv_report.layers)} layers within tolerance",
                data={"divergent_layers": []},
            )

        return DiagnosticStep(
            name="kv_cache",
            description="KV cache numerical accuracy",
            status=StepStatus.FAIL,
            detail=(
                f"{len(kv_report.divergent_layers)} divergent layer(s): "
                + ", ".join(kv_report.divergent_layers)
            ),
            data={"divergent_layers": kv_report.divergent_layers},
        )

    async def _step_full_sequence(self) -> DiagnosticStep:
        """Step 3: Full sequence logprobs comparison."""
        try:
            comparison = await self._collect_and_compare(max_tokens=self.max_tokens)
        except Exception as e:
            return DiagnosticStep(
                name="full_sequence",
                description="Baseline vs target: full sequence match",
                status=StepStatus.FAIL,
                detail=f"Error: {e}",
            )

        if comparison.match:
            return DiagnosticStep(
                name="full_sequence",
                description="Baseline vs target: full sequence match",
                status=StepStatus.PASS,
                detail=f"All {comparison.total_tokens_compared} tokens match",
            )

        d = comparison.divergence
        detail = (
            f"Divergence at token {d.token_index}: "
            f"expected {d.expected_token!r}, got {d.actual_token!r} "
            f"(logprob diff: {d.prob_diff:.6f})"
        ) if d else "Comparison failed"

        return DiagnosticStep(
            name="full_sequence",
            description="Baseline vs target: full sequence match",
            status=StepStatus.FAIL,
            detail=detail,
            data={
                "divergence_index": d.token_index if d else None,
                "total_tokens": comparison.total_tokens_compared,
            },
        )

    async def _collect_and_compare(self, max_tokens: int) -> ComparisonReport:
        """Collect logprobs from both endpoints and compare."""
        baseline_collector = LogprobsCollector(
            self.baseline_url, api_key=self.api_key, model=self.model,
        )
        target_collector = LogprobsCollector(
            self.target_url, api_key=self.api_key, model=self.model,
        )

        baseline_result = await baseline_collector.collect(
            self.prompt, max_tokens=max_tokens, sampling_params=self.sampling_params,
        )
        target_result = await target_collector.collect(
            self.prompt, max_tokens=max_tokens, sampling_params=self.sampling_params,
        )

        comparator = LogprobsComparator()
        return comparator.compare(baseline_result, target_result)


def format_rich_report(report: DiagnosticReport) -> str:
    """Format diagnostic report with rich terminal symbols."""
    status_icons = {
        StepStatus.PASS: "✅",
        StepStatus.FAIL: "❌",
        StepStatus.SKIP: "⏭️",
    }

    lines = [
        "╔══════════════════════════════════════╗",
        "║    xPyD-acc Diagnostic Report        ║",
        "╚══════════════════════════════════════╝",
        "",
    ]

    for i, step in enumerate(report.steps, 1):
        icon = status_icons[step.status]
        lines.append(f"  Step {i}: {step.description}")
        lines.append(f"    {icon} {step.status.value.upper()} — {step.detail}")
        lines.append("")

    lines.append("─" * 40)
    if report.overall_pass:
        lines.append("✅ ALL CHECKS PASSED")
    else:
        failed = sum(1 for s in report.steps if s.status == StepStatus.FAIL)
        lines.append(f"❌ {failed} CHECK(S) FAILED")

    return "\n".join(lines)
