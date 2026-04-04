"""Named profile (preset) support for xpyd-acc.

Profiles are named configuration presets that can be activated via
``--profile <name>`` to reduce repetitive CLI flags.

Built-in profiles are always available.  User-defined profiles live in
``[profiles.<name>]`` TOML sections and override built-ins of the same name.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class ProfileConfig:
    """A named configuration profile (preset).

    All fields are optional — only non-None values override the defaults.
    """

    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    retries: int | None = None
    retry_delay: float | None = None
    normalize_whitespace: bool | None = None
    ignore_case: bool | None = None
    numeric_tolerance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return non-None fields as a dict."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

BUILTIN_PROFILES: dict[str, ProfileConfig] = {
    "greedy": ProfileConfig(temperature=0.0, seed=42),
    "stochastic": ProfileConfig(temperature=0.7),
}


def parse_profiles(raw: dict[str, Any]) -> dict[str, ProfileConfig]:
    """Parse ``[profiles.*]`` TOML tables into ProfileConfig instances.

    Unknown keys inside a profile section are silently ignored.

    Args:
        raw: The top-level parsed TOML dict (or the ``profiles`` sub-dict).

    Returns:
        Mapping of profile name → ProfileConfig.
    """
    profiles_section = raw.get("profiles", raw) if "profiles" in raw else raw
    valid_fields = {f.name for f in ProfileConfig.__dataclass_fields__.values()}
    result: dict[str, ProfileConfig] = {}
    for name, values in profiles_section.items():
        if not isinstance(values, dict):
            continue
        filtered = {k: v for k, v in values.items() if k in valid_fields}
        result[name] = ProfileConfig(**filtered)
    return result


def resolve_profile(
    name: str,
    user_profiles: dict[str, ProfileConfig] | None = None,
) -> ProfileConfig:
    """Look up a profile by name.

    User-defined profiles take precedence over built-ins.

    Args:
        name: Profile name to look up.
        user_profiles: Profiles parsed from user config file.

    Returns:
        The resolved ProfileConfig.

    Raises:
        KeyError: If the profile name is not found.
    """
    if user_profiles and name in user_profiles:
        return user_profiles[name]
    if name in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name]
    available = sorted(set((user_profiles or {}).keys()) | set(BUILTIN_PROFILES.keys()))
    raise KeyError(
        f"Unknown profile '{name}'. Available profiles: {', '.join(available)}"
    )


def apply_profile(args: dict[str, Any], profile: ProfileConfig) -> dict[str, Any]:
    """Apply profile values to an args dict.

    Profile values fill in where the current value is None (or False for bools).
    CLI flags (non-None) always win.

    Args:
        args: Mutable argument dict.
        profile: Profile to apply.

    Returns:
        The same args dict, mutated in place.
    """
    for key, val in profile.to_dict().items():
        if key in args:
            current = args[key]
            if current is None or current is False:
                args[key] = val
    return args


def list_profiles(
    user_profiles: dict[str, ProfileConfig] | None = None,
) -> dict[str, ProfileConfig]:
    """Return all available profiles (built-in + user-defined).

    User-defined profiles override built-ins of the same name.
    """
    merged = dict(BUILTIN_PROFILES)
    if user_profiles:
        merged.update(user_profiles)
    return merged
