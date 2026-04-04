"""Tests for named profile (preset) support."""

from __future__ import annotations

import pytest

from xpyd_acc.profiles import (
    BUILTIN_PROFILES,
    ProfileConfig,
    apply_profile,
    list_profiles,
    parse_profiles,
    resolve_profile,
)

# ---------------------------------------------------------------------------
# ProfileConfig
# ---------------------------------------------------------------------------

class TestProfileConfig:
    def test_to_dict_all_none(self):
        p = ProfileConfig()
        assert p.to_dict() == {}

    def test_to_dict_partial(self):
        p = ProfileConfig(temperature=0.0, seed=42)
        assert p.to_dict() == {"temperature": 0.0, "seed": 42}

    def test_to_dict_full(self):
        p = ProfileConfig(
            model="gpt-4", temperature=0.5, top_p=0.9, seed=1,
            max_tokens=128, retries=5, retry_delay=2.0,
            normalize_whitespace=True, ignore_case=True, numeric_tolerance=0.01,
        )
        assert len(p.to_dict()) == 10


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

class TestBuiltinProfiles:
    def test_greedy_profile(self):
        p = BUILTIN_PROFILES["greedy"]
        assert p.temperature == 0.0
        assert p.seed == 42

    def test_stochastic_profile(self):
        p = BUILTIN_PROFILES["stochastic"]
        assert p.temperature == 0.7
        assert p.seed is None


# ---------------------------------------------------------------------------
# parse_profiles
# ---------------------------------------------------------------------------

class TestParseProfiles:
    def test_empty(self):
        assert parse_profiles({}) == {}

    def test_from_raw_toml(self):
        raw = {
            "profiles": {
                "fast": {"max_tokens": 32, "temperature": 0.0},
                "slow": {"max_tokens": 512, "retries": 10},
            }
        }
        result = parse_profiles(raw)
        assert "fast" in result
        assert result["fast"].max_tokens == 32
        assert result["fast"].temperature == 0.0
        assert result["slow"].retries == 10

    def test_unknown_keys_ignored(self):
        raw = {
            "profiles": {
                "x": {"temperature": 0.5, "bogus_key": 999},
            }
        }
        result = parse_profiles(raw)
        assert result["x"].temperature == 0.5

    def test_non_dict_values_skipped(self):
        raw = {"profiles": {"name": "not-a-dict"}}
        result = parse_profiles(raw)
        assert result == {}


# ---------------------------------------------------------------------------
# resolve_profile
# ---------------------------------------------------------------------------

class TestResolveProfile:
    def test_builtin(self):
        p = resolve_profile("greedy")
        assert p.temperature == 0.0

    def test_user_overrides_builtin(self):
        user = {"greedy": ProfileConfig(temperature=0.0, seed=99)}
        p = resolve_profile("greedy", user)
        assert p.seed == 99

    def test_user_only(self):
        user = {"custom": ProfileConfig(model="llama")}
        p = resolve_profile("custom", user)
        assert p.model == "llama"

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown profile 'nope'"):
            resolve_profile("nope")

    def test_error_lists_available(self):
        user = {"alpha": ProfileConfig()}
        with pytest.raises(KeyError, match="alpha"):
            resolve_profile("nope", user)


# ---------------------------------------------------------------------------
# apply_profile
# ---------------------------------------------------------------------------

class TestApplyProfile:
    def test_fills_none_values(self):
        args = {"temperature": None, "seed": None, "model": "gpt-4"}
        profile = ProfileConfig(temperature=0.0, seed=42, model="llama")
        apply_profile(args, profile)
        assert args["temperature"] == 0.0
        assert args["seed"] == 42
        assert args["model"] == "gpt-4"  # CLI wins

    def test_fills_false_bools(self):
        args = {"normalize_whitespace": False, "ignore_case": False}
        profile = ProfileConfig(normalize_whitespace=True, ignore_case=True)
        apply_profile(args, profile)
        assert args["normalize_whitespace"] is True
        assert args["ignore_case"] is True

    def test_no_effect_when_all_set(self):
        args = {"temperature": 0.9, "seed": 1}
        profile = ProfileConfig(temperature=0.0, seed=42)
        apply_profile(args, profile)
        assert args["temperature"] == 0.9
        assert args["seed"] == 1

    def test_missing_key_in_args_ignored(self):
        args = {"model": None}
        profile = ProfileConfig(temperature=0.5)
        apply_profile(args, profile)
        assert "temperature" not in args


# ---------------------------------------------------------------------------
# list_profiles
# ---------------------------------------------------------------------------

class TestListProfiles:
    def test_builtins_only(self):
        result = list_profiles()
        assert "greedy" in result
        assert "stochastic" in result

    def test_user_added(self):
        user = {"custom": ProfileConfig(model="x")}
        result = list_profiles(user)
        assert "custom" in result
        assert "greedy" in result

    def test_user_overrides(self):
        user = {"greedy": ProfileConfig(temperature=0.0, seed=99)}
        result = list_profiles(user)
        assert result["greedy"].seed == 99


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    def test_profiles_subcommand(self):
        """The 'profiles' subcommand should not error."""
        from xpyd_acc.cli import main

        # Just ensure it doesn't crash (no config file needed)
        main(["profiles"])

    def test_profile_flag_unknown(self):
        """--profile with unknown name should error."""
        from xpyd_acc.cli import main
        with pytest.raises(SystemExit):
            main(["compare-logprobs", "--baseline", "http://a", "--target", "http://b",
                   "--prompt", "hi", "--profile", "nonexistent_xyz"])
