"""Endpoint health check for OpenAI-compatible API servers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from xpyd_acc.log import get_logger

logger = get_logger("healthcheck")


@dataclass
class HealthCheckResult:
    """Result of a single endpoint health check."""

    url: str
    reachable: bool
    response_time_ms: float | None
    models_available: list[str]
    error: str | None

    @property
    def healthy(self) -> bool:
        """Return True if endpoint is reachable and responded."""
        return self.reachable and self.error is None


async def check_endpoint(
    url: str,
    *,
    api_key: str = "no-key",
    timeout: float = 10.0,
) -> HealthCheckResult:
    """Check if an OpenAI-compatible endpoint is healthy.

    Performs:
    1. TCP connectivity check via /v1/models
    2. Response time measurement
    3. Model list extraction

    Args:
        url: Base URL of the endpoint (e.g. http://localhost:8000).
        api_key: API key for authentication.
        timeout: HTTP timeout in seconds.

    Returns:
        HealthCheckResult with check details.
    """
    base_url = url.rstrip("/")
    models_url = f"{base_url}/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    logger.info("Checking endpoint health: %s", models_url)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            start = time.monotonic()
            resp = await client.get(models_url, headers=headers)
            elapsed_ms = (time.monotonic() - start) * 1000

            resp.raise_for_status()
            data = resp.json()
            models = [
                m.get("id", "unknown")
                for m in data.get("data", [])
            ]

            return HealthCheckResult(
                url=base_url,
                reachable=True,
                response_time_ms=round(elapsed_ms, 1),
                models_available=models,
                error=None,
            )
    except httpx.ConnectError as exc:
        return HealthCheckResult(
            url=base_url,
            reachable=False,
            response_time_ms=None,
            models_available=[],
            error=f"Connection failed: {exc}",
        )
    except httpx.TimeoutException:
        return HealthCheckResult(
            url=base_url,
            reachable=False,
            response_time_ms=None,
            models_available=[],
            error=f"Timeout after {timeout}s",
        )
    except httpx.HTTPStatusError as exc:
        elapsed_ms = (time.monotonic() - start) * 1000  # type: ignore[possibly-undefined]
        return HealthCheckResult(
            url=base_url,
            reachable=True,
            response_time_ms=round(elapsed_ms, 1),
            models_available=[],
            error=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
        )
    except Exception as exc:
        return HealthCheckResult(
            url=base_url,
            reachable=False,
            response_time_ms=None,
            models_available=[],
            error=str(exc),
        )


async def check_endpoints(
    urls: list[str],
    *,
    api_key: str = "no-key",
    timeout: float = 10.0,
) -> list[HealthCheckResult]:
    """Check multiple endpoints sequentially.

    Args:
        urls: List of endpoint URLs to check.
        api_key: API key for authentication.
        timeout: HTTP timeout per endpoint.

    Returns:
        List of HealthCheckResult, one per URL.
    """
    results = []
    for url in urls:
        result = await check_endpoint(url, api_key=api_key, timeout=timeout)
        results.append(result)
    return results


def format_healthcheck(results: list[HealthCheckResult]) -> str:
    """Format health check results for terminal display.

    Args:
        results: List of HealthCheckResult to format.

    Returns:
        Human-readable string with ✅/❌ indicators.
    """
    lines = ["Endpoint Health Check", "=" * 40]

    for r in results:
        status = "✅" if r.healthy else "❌"
        lines.append(f"\n{status} {r.url}")

        if r.reachable:
            lines.append(f"  Reachable: ✅ ({r.response_time_ms}ms)")
        else:
            lines.append("  Reachable: ❌")

        if r.models_available:
            models_str = ", ".join(r.models_available)
            lines.append(f"  Models: {models_str}")
        elif r.reachable:
            lines.append("  Models: none listed")

        if r.error:
            lines.append(f"  Error: {r.error}")

    all_healthy = all(r.healthy for r in results)
    lines.append("")
    overall = "✅ All endpoints healthy" if all_healthy else "❌ Some endpoints unhealthy"
    lines.append(f"Overall: {overall}")

    return "\n".join(lines)
