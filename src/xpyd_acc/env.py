"""Environment variable support for xpyd-acc configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Environment variable names
ENV_API_KEY = "XPYD_ACC_API_KEY"
ENV_BASELINE_URL = "XPYD_ACC_BASELINE_URL"
ENV_TARGET_URL = "XPYD_ACC_TARGET_URL"
ENV_MODEL = "XPYD_ACC_MODEL"


@dataclass
class EnvDefaults:
    """Defaults resolved from environment variables."""

    api_key: str | None = None
    baseline_url: str | None = None
    target_url: str | None = None
    model: str | None = None


def get_env_defaults() -> EnvDefaults:
    """Read configuration defaults from environment variables.

    Returns an EnvDefaults with non-None values only for variables that are set
    and non-empty in the environment.
    """
    return EnvDefaults(
        api_key=os.environ.get(ENV_API_KEY) or None,
        baseline_url=os.environ.get(ENV_BASELINE_URL) or None,
        target_url=os.environ.get(ENV_TARGET_URL) or None,
        model=os.environ.get(ENV_MODEL) or None,
    )


def apply_env_defaults(
    cli_value: str | None,
    env_value: str | None,
    config_value: str | None = None,
    default: str | None = None,
) -> str | None:
    """Resolve a value using priority: CLI > env > config > default."""
    if cli_value is not None:
        return cli_value
    if env_value is not None:
        return env_value
    if config_value is not None:
        return config_value
    return default
