"""Prompt sensitivity analysis — test if divergence persists across prompt perturbations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from xpyd_acc.logprobs import LogprobsCollector


def generate_perturbations(prompt: str, count: int = 5) -> list[str]:
    """Generate *count* minor perturbations of *prompt*.

    Perturbation strategies (cycled to reach *count*):
    1. Leading space
    2. Trailing newline
    3. Double trailing newline
    4. Leading newline
    5. Trailing space
    6. Leading + trailing space
    7. Replace first double-space with single (or add double-space)
    """
    strategies: list[Callable[[str], str]] = [
        lambda p: " " + p,
        lambda p: p + "\n",
        lambda p: p + "\n\n",
        lambda p: "\n" + p,
        lambda p: p + " ",
        lambda p: " " + p + " ",
        lambda p: p.replace("  ", " ", 1) if "  " in p else p + "  ",
    ]
    results: list[str] = []
    seen: set[str] = {prompt}
    idx = 0
    while len(results) < count:
        candidate = strategies[idx % len(strategies)](prompt)
        if candidate not in seen:
            results.append(candidate)
            seen.add(candidate)
        idx += 1
        # Safety: if we've cycled through all strategies without new candidates, stop
        if idx > count + len(strategies) * 2:
            break
    return results


@dataclass
class PerturbationResult:
    """Result for a single perturbation."""

    perturbation: str
    diverges: bool
    first_divergence_index: int | None
    total_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "perturbation": self.perturbation,
            "diverges": self.diverges,
            "first_divergence_index": self.first_divergence_index,
            "total_tokens": self.total_tokens,
        }


@dataclass
class SensitivityResult:
    """Aggregate sensitivity analysis result."""

    original_prompt: str
    original_diverges: bool
    perturbation_count: int
    divergent_count: int
    classification: str  # "systematic", "sensitive", "robust"
    perturbation_results: list[PerturbationResult] = field(default_factory=list)

    @property
    def divergence_rate(self) -> float:
        total = self.perturbation_count + 1  # include original
        divergent = self.divergent_count + (1 if self.original_diverges else 0)
        return divergent / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "original_diverges": self.original_diverges,
            "perturbation_count": self.perturbation_count,
            "divergent_count": self.divergent_count,
            "classification": self.classification,
            "divergence_rate": self.divergence_rate,
            "perturbation_results": [r.to_dict() for r in self.perturbation_results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _classify(original_diverges: bool, perturbation_results: list[PerturbationResult]) -> str:
    """Classify sensitivity: systematic, sensitive, or robust."""
    all_diverge = original_diverges and all(r.diverges for r in perturbation_results)
    none_diverge = not original_diverges and all(not r.diverges for r in perturbation_results)
    if not perturbation_results:
        return "systematic" if original_diverges else "robust"
    if all_diverge:
        return "systematic"
    if none_diverge:
        return "robust"
    return "sensitive"


async def run_sensitivity(
    baseline_url: str,
    target_url: str,
    prompt: str,
    *,
    model: str = "default",
    max_tokens: int = 64,
    api_key: str | None = None,
    perturbation_count: int = 5,
    retries: int = 3,
    retry_delay: float = 1.0,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    on_result: Callable[[int, PerturbationResult], None] | None = None,
) -> SensitivityResult:
    """Run sensitivity analysis by testing prompt perturbations."""
    collector = LogprobsCollector(
        baseline_url=baseline_url,
        target_url=target_url,
        model=model,
        max_tokens=max_tokens,
        api_key=api_key,
        retries=retries,
        retry_delay=retry_delay,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    # Test original prompt
    original_result = await collector.collect(prompt)
    original_diverges = original_result.first_divergence_index is not None

    # Generate and test perturbations
    perturbations = generate_perturbations(prompt, count=perturbation_count)
    perturbation_results: list[PerturbationResult] = []

    for i, perturbed in enumerate(perturbations):
        result = await collector.collect(perturbed)
        diverges = result.first_divergence_index is not None
        pr = PerturbationResult(
            perturbation=perturbed,
            diverges=diverges,
            first_divergence_index=result.first_divergence_index,
            total_tokens=len(result.baseline_tokens),
        )
        perturbation_results.append(pr)
        if on_result:
            on_result(i, pr)

    classification = _classify(original_diverges, perturbation_results)
    divergent_count = sum(1 for r in perturbation_results if r.diverges)

    return SensitivityResult(
        original_prompt=prompt,
        original_diverges=original_diverges,
        perturbation_count=len(perturbation_results),
        divergent_count=divergent_count,
        classification=classification,
        perturbation_results=perturbation_results,
    )


def format_sensitivity(result: SensitivityResult) -> str:
    """Format sensitivity result for terminal output."""
    lines: list[str] = []
    lines.append("=== Prompt Sensitivity Analysis ===")
    lines.append("")
    lines.append(f"Original prompt: {result.original_prompt!r}")
    lines.append(f"Original diverges: {'Yes' if result.original_diverges else 'No'}")
    lines.append(f"Perturbations tested: {result.perturbation_count}")
    lines.append(f"Perturbations divergent: {result.divergent_count}")
    lines.append(f"Overall divergence rate: {result.divergence_rate:.1%}")
    lines.append(f"Classification: {result.classification.upper()}")
    lines.append("")

    if result.perturbation_results:
        lines.append("--- Per-Perturbation Results ---")
        for i, pr in enumerate(result.perturbation_results, 1):
            status = "DIVERGE" if pr.diverges else "MATCH"
            div_info = f" (index {pr.first_divergence_index})" if pr.diverges else ""
            lines.append(f"  [{i}] {status}{div_info} | tokens={pr.total_tokens}")
    lines.append("")

    if result.classification == "systematic":
        lines.append("Verdict: Divergence is SYSTEMATIC — persists across all perturbations.")
    elif result.classification == "sensitive":
        lines.append("Verdict: Divergence is SENSITIVE — depends on exact prompt wording.")
    else:
        lines.append("Verdict: Output is ROBUST — no divergence detected.")

    return "\n".join(lines)
