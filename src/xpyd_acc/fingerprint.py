"""Model fingerprinting: deterministic probes to identify endpoint behavior."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from xpyd_acc.log import get_logger

logger = get_logger("fingerprint")

# Fixed probes designed to elicit deterministic, model-characteristic responses.
# Temperature=0, seed=42, max_tokens=16 to keep outputs short and stable.
DEFAULT_PROBES = [
    "What is 2+2?",
    "Complete: The capital of France is",
    "Translate to French: hello",
    "What comes after 1, 1, 2, 3, 5?",
    "Repeat exactly: alpha bravo charlie",
]


@dataclass
class ProbeResult:
    """Result of a single fingerprint probe."""

    prompt: str
    output: str
    tokens: list[str] = field(default_factory=list)


@dataclass
class Fingerprint:
    """A model fingerprint built from deterministic probes."""

    endpoint: str
    model: str
    hash: str
    probes: list[ProbeResult]
    probe_count: int

    def matches(self, other: Fingerprint) -> bool:
        """Check if two fingerprints are identical."""
        return self.hash == other.hash

    def diff(self, other: Fingerprint) -> list[dict[str, Any]]:
        """Return per-probe diffs where outputs differ."""
        diffs: list[dict[str, Any]] = []
        for i, (a, b) in enumerate(zip(self.probes, other.probes)):
            if a.output != b.output:
                diffs.append({
                    "probe_index": i,
                    "prompt": a.prompt,
                    "output_a": a.output,
                    "output_b": b.output,
                })
        return diffs

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "hash": self.hash,
            "probe_count": self.probe_count,
            "probes": [
                {"prompt": p.prompt, "output": p.output, "tokens": p.tokens}
                for p in self.probes
            ],
        }


@dataclass
class FingerprintComparison:
    """Result of comparing two fingerprints."""

    match: bool
    hash_a: str
    hash_b: str
    endpoint_a: str
    endpoint_b: str
    differing_probes: list[dict[str, Any]]
    total_probes: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "match": self.match,
            "hash_a": self.hash_a,
            "hash_b": self.hash_b,
            "endpoint_a": self.endpoint_a,
            "endpoint_b": self.endpoint_b,
            "differing_probes": self.differing_probes,
            "total_probes": self.total_probes,
        }


def _compute_hash(probes: list[ProbeResult]) -> str:
    """Compute SHA-256 hash from probe outputs."""
    hasher = hashlib.sha256()
    for p in probes:
        hasher.update(p.output.encode("utf-8"))
    return hasher.hexdigest()[:16]


async def collect_fingerprint(
    endpoint: str,
    *,
    api_key: str = "no-key",
    model: str = "default",
    probes: list[str] | None = None,
    max_tokens: int = 16,
    timeout: float = 30.0,
    retries: int = 3,
    retry_delay: float = 1.0,
) -> Fingerprint:
    """Send deterministic probes to an endpoint and build a fingerprint."""
    from xpyd_acc.logprobs import LogprobsCollector
    from xpyd_acc.sampling import SamplingParams

    probe_prompts = probes if probes is not None else DEFAULT_PROBES
    collector = LogprobsCollector(endpoint, api_key=api_key, model=model)
    sampling = SamplingParams(temperature=0, seed=42)

    results: list[ProbeResult] = []
    for prompt in probe_prompts:
        logger.info("Probing: %s", prompt[:50])
        lr = await collector.collect(
            prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            sampling_params=sampling,
        )
        tokens = [t.token for t in lr.tokens]
        output = "".join(tokens)
        results.append(ProbeResult(prompt=prompt, output=output, tokens=tokens))

    fp_hash = _compute_hash(results)
    logger.info("Fingerprint for %s: %s", endpoint, fp_hash)
    return Fingerprint(
        endpoint=endpoint,
        model=lr.model if results else model,
        hash=fp_hash,
        probes=results,
        probe_count=len(results),
    )


def compare_fingerprints(a: Fingerprint, b: Fingerprint) -> FingerprintComparison:
    """Compare two fingerprints and return a comparison result."""
    diffs = a.diff(b)
    return FingerprintComparison(
        match=a.matches(b),
        hash_a=a.hash,
        hash_b=b.hash,
        endpoint_a=a.endpoint,
        endpoint_b=b.endpoint,
        differing_probes=diffs,
        total_probes=min(a.probe_count, b.probe_count),
    )
