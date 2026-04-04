"""Integration with xPyD ecosystem: endpoint detection and proxy config helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx


class EndpointType(str, Enum):
    """Type of an xPyD endpoint."""

    AGGREGATED = "aggregated"
    DISAGGREGATED = "disaggregated"
    UNKNOWN = "unknown"


@dataclass
class EndpointInfo:
    """Information about a detected endpoint."""

    url: str
    endpoint_type: EndpointType = EndpointType.UNKNOWN
    model: str | None = None
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_aggregated(self) -> bool:
        return self.endpoint_type == EndpointType.AGGREGATED

    @property
    def is_disaggregated(self) -> bool:
        return self.endpoint_type == EndpointType.DISAGGREGATED


async def detect_endpoint_type(
    url: str,
    *,
    timeout: float = 10.0,
) -> EndpointInfo:
    """Probe an endpoint and classify it as aggregated, disaggregated, or unknown.

    Detection strategy:
    1. Try GET /health or /v1/models to confirm endpoint is alive.
    2. Try GET /info or /v1/info for xPyD-specific metadata.
    3. Look for disaggregation markers in the response (e.g. "mode", "type",
       "disaggregated", "prefill", "decode" keys).
    """
    info = EndpointInfo(url=url)
    base = url.rstrip("/")

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Step 1: health check
        alive = False
        for path in ("/health", "/v1/models"):
            try:
                resp = await client.get(f"{base}{path}")
                if resp.status_code < 500:
                    alive = True
                    if path == "/v1/models":
                        _parse_models_response(resp, info)
                    break
            except httpx.HTTPError:
                continue

        if not alive:
            return info

        # Step 2: probe info endpoints for xPyD metadata
        for path in ("/info", "/v1/info", "/v1/config"):
            try:
                resp = await client.get(f"{base}{path}")
                if resp.status_code == 200:
                    data = resp.json()
                    info.metadata.update(data)
                    _classify_from_metadata(data, info)
                    if info.endpoint_type != EndpointType.UNKNOWN:
                        return info
            except (httpx.HTTPError, json.JSONDecodeError, ValueError):
                continue

        # Step 3: heuristic — if /health is up but no info endpoint,
        # treat as aggregated (standard OpenAI-compatible server)
        if alive and info.endpoint_type == EndpointType.UNKNOWN:
            info.endpoint_type = EndpointType.AGGREGATED

    return info


def _parse_models_response(resp: httpx.Response, info: EndpointInfo) -> None:
    """Extract model name from /v1/models response."""
    try:
        data = resp.json()
        models = data.get("data", [])
        if models and isinstance(models, list):
            info.model = models[0].get("id")
    except (json.JSONDecodeError, ValueError):
        pass


def _classify_from_metadata(data: dict[str, Any], info: EndpointInfo) -> None:
    """Classify endpoint type from metadata dict."""
    if data.get("version"):
        info.version = str(data["version"])

    # Look for explicit type/mode fields
    for key in ("type", "mode", "endpoint_type", "server_type"):
        val = str(data.get(key, "")).lower()
        if "disagg" in val or "pd" in val:
            info.endpoint_type = EndpointType.DISAGGREGATED
            return
        if "agg" in val and "disagg" not in val:
            info.endpoint_type = EndpointType.AGGREGATED
            return

    # Look for disaggregation markers (prefill/decode split)
    all_text = json.dumps(data).lower()
    disagg_markers = ("prefill_url", "decode_url", "prefill_endpoint", "decode_endpoint")
    if any(marker in all_text for marker in disagg_markers):
        info.endpoint_type = EndpointType.DISAGGREGATED
        return


@dataclass
class ProxyEndpoint:
    """A single endpoint extracted from proxy config."""

    name: str
    url: str
    endpoint_type: EndpointType = EndpointType.UNKNOWN


@dataclass
class ProxyConfig:
    """Helper to parse xPyD-proxy configuration and extract endpoints."""

    endpoints: list[ProxyEndpoint] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> ProxyConfig:
        """Load proxy config from a JSON or YAML file."""
        p = Path(path)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            # Lightweight YAML subset: only support JSON-compatible YAML
            # For full YAML, users should install PyYAML
            try:
                import yaml  # type: ignore[import-untyped]

                data = yaml.safe_load(text)
            except ImportError:
                msg = "PyYAML is required to load YAML config files"
                raise ImportError(msg) from None
        else:
            data = json.loads(text)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProxyConfig:
        """Parse proxy config from a dict."""
        config = cls(raw=data)

        # Support multiple config formats
        # Format 1: {"endpoints": [{"name": ..., "url": ...}, ...]}
        if "endpoints" in data and isinstance(data["endpoints"], list):
            for ep in data["endpoints"]:
                if isinstance(ep, dict) and "url" in ep:
                    config.endpoints.append(
                        ProxyEndpoint(
                            name=ep.get("name", ep["url"]),
                            url=ep["url"],
                            endpoint_type=_parse_type(ep.get("type", "")),
                        )
                    )

        # Format 2: {"prefill_url": ..., "decode_url": ...}
        if "prefill_url" in data and "decode_url" in data:
            config.endpoints.append(
                ProxyEndpoint(
                    name="prefill",
                    url=data["prefill_url"],
                    endpoint_type=EndpointType.DISAGGREGATED,
                )
            )
            config.endpoints.append(
                ProxyEndpoint(
                    name="decode",
                    url=data["decode_url"],
                    endpoint_type=EndpointType.DISAGGREGATED,
                )
            )

        # Format 3: {"upstream": "http://..."}
        if "upstream" in data and isinstance(data["upstream"], str):
            config.endpoints.append(
                ProxyEndpoint(
                    name="upstream",
                    url=data["upstream"],
                )
            )

        return config

    def get_baseline(self) -> ProxyEndpoint | None:
        """Get the aggregated/baseline endpoint, if any."""
        for ep in self.endpoints:
            if ep.endpoint_type == EndpointType.AGGREGATED:
                return ep
            if ep.name in ("baseline", "aggregated", "upstream"):
                return ep
        return self.endpoints[0] if self.endpoints else None

    def get_target(self) -> ProxyEndpoint | None:
        """Get the disaggregated/target endpoint, if any."""
        for ep in self.endpoints:
            if ep.endpoint_type == EndpointType.DISAGGREGATED:
                return ep
            if ep.name in ("target", "disaggregated", "pd"):
                return ep
        return self.endpoints[1] if len(self.endpoints) > 1 else None


def _parse_type(val: str) -> EndpointType:
    """Parse an endpoint type string."""
    v = val.lower()
    if "disagg" in v or "pd" in v:
        return EndpointType.DISAGGREGATED
    if "agg" in v and "disagg" not in v:
        return EndpointType.AGGREGATED
    return EndpointType.UNKNOWN


def format_detect_report(info: EndpointInfo) -> str:
    """Format endpoint detection result for terminal output."""
    lines = [
        "═" * 60,
        "  xPyD Endpoint Detection Report",
        "═" * 60,
        f"  URL:    {info.url}",
        f"  Type:   {info.endpoint_type.value}",
    ]
    if info.model:
        lines.append(f"  Model:  {info.model}")
    if info.version:
        lines.append(f"  Version: {info.version}")
    if info.metadata:
        lines.append(f"  Metadata keys: {', '.join(info.metadata.keys())}")
    lines.append("═" * 60)
    return "\n".join(lines)
