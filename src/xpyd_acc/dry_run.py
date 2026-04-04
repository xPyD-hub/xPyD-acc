"""Dry run mode for batch comparison: validate setup without API calls."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DryRunResult:
    """Result of a dry run validation."""

    valid: bool
    sample_count: int
    estimated_prompt_tokens: int
    resolved_config: dict[str, Any] = field(default_factory=dict)
    template_name: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    healthcheck_passed: bool | None = None  # None if skipped

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def validate_dataset(dataset_path: str | Path) -> tuple[int, int, list[str]]:
    """Validate dataset file, return (sample_count, estimated_tokens, errors)."""
    from xpyd_acc.batch_compare import load_dataset

    errors: list[str] = []
    path = Path(dataset_path)

    if not path.exists():
        return 0, 0, [f"Dataset file not found: {path}"]

    try:
        samples = load_dataset(path)
    except Exception as e:
        return 0, 0, [f"Failed to load dataset: {e}"]

    if not samples:
        return 0, 0, ["Dataset is empty (0 samples)"]

    total_tokens = sum(_estimate_tokens(s.prompt) for s in samples)
    return len(samples), total_tokens, errors


def validate_template(
    template_spec: str,
    sample_prompts: list[str],
    sample_metadata: list[dict[str, Any]],
) -> tuple[str | None, list[str]]:
    """Validate template rendering. Returns (template_name, errors)."""
    from xpyd_acc.templates import resolve_template

    errors: list[str] = []
    try:
        template = resolve_template(template_spec)
    except Exception as e:
        return None, [f"Failed to resolve template '{template_spec}': {e}"]

    # Try rendering first sample
    if sample_prompts:
        variables = {"prompt": sample_prompts[0]}
        if sample_metadata:
            variables.update(sample_metadata[0])
        try:
            rendered = template.render(variables)
            if not rendered.strip():
                errors.append("Template rendered to empty string for first sample")
        except Exception as e:
            errors.append(f"Template render failed on first sample: {e}")

    return template.name, errors


async def run_dry_run(
    dataset_path: str | Path,
    baseline_url: str,
    target_url: str,
    *,
    template: str | None = None,
    skip_healthcheck: bool = False,
    model: str | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    concurrency: int | None = None,
    retries: int | None = None,
    retry_delay: float | None = None,
) -> DryRunResult:
    """Run dry validation of batch-compare setup."""
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Validate dataset
    sample_count, estimated_tokens, ds_errors = validate_dataset(dataset_path)
    errors.extend(ds_errors)

    # 2. Validate template if specified
    template_name: str | None = None
    if template and not ds_errors:
        from xpyd_acc.batch_compare import load_dataset

        samples = load_dataset(dataset_path)
        prompts = [s.prompt for s in samples]
        metadata = [s.metadata for s in samples]
        template_name, tpl_errors = validate_template(template, prompts, metadata)
        errors.extend(tpl_errors)

        if not tpl_errors:
            # Re-estimate tokens with rendered prompts
            from xpyd_acc.templates import resolve_template

            tpl = resolve_template(template)
            total = 0
            for s in samples:
                variables = {"prompt": s.prompt, **s.metadata}
                rendered = tpl.render(variables)
                total += _estimate_tokens(rendered)
            estimated_tokens = total

    # 3. Healthcheck
    healthcheck_passed: bool | None = None
    if not skip_healthcheck and not errors:
        from xpyd_acc.healthcheck import check_endpoint

        healthcheck_passed = True
        for url in [baseline_url, target_url]:
            try:
                result = await check_endpoint(url, api_key=api_key)
                if not result.reachable:
                    errors.append(f"Endpoint unreachable: {url}")
                    healthcheck_passed = False
            except Exception as e:
                errors.append(f"Healthcheck failed for {url}: {e}")
                healthcheck_passed = False

    # 4. Build resolved config
    resolved_config = {
        "baseline": baseline_url,
        "target": target_url,
        "dataset": str(dataset_path),
        "model": model or "default",
        "max_tokens": max_tokens or 64,
        "concurrency": concurrency or 5,
        "retries": retries or 3,
        "retry_delay": retry_delay or 1.0,
    }

    if sample_count > 1000:
        warnings.append(f"Large dataset ({sample_count} samples) — batch run may take a while")

    return DryRunResult(
        valid=len(errors) == 0,
        sample_count=sample_count,
        estimated_prompt_tokens=estimated_tokens,
        resolved_config=resolved_config,
        template_name=template_name,
        errors=errors,
        warnings=warnings,
        healthcheck_passed=healthcheck_passed,
    )


def format_dry_run(result: DryRunResult) -> str:
    """Format dry run result for terminal output."""
    lines: list[str] = []
    status = "✅ PASS" if result.valid else "❌ FAIL"
    lines.append(f"Dry Run Validation: {status}")
    lines.append("")

    lines.append(f"  Samples:          {result.sample_count}")
    lines.append(f"  Est. tokens:      {result.estimated_prompt_tokens:,}")
    if result.template_name:
        lines.append(f"  Template:         {result.template_name}")

    if result.healthcheck_passed is not None:
        hc = "✅" if result.healthcheck_passed else "❌"
        lines.append(f"  Healthcheck:      {hc}")
    else:
        lines.append("  Healthcheck:      skipped")

    lines.append("")
    lines.append("  Resolved config:")
    for k, v in result.resolved_config.items():
        lines.append(f"    {k}: {v}")

    if result.warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w in result.warnings:
            lines.append(f"    ⚠️  {w}")

    if result.errors:
        lines.append("")
        lines.append("  Errors:")
        for e in result.errors:
            lines.append(f"    ❌ {e}")

    return "\n".join(lines)
