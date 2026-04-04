"""Tests for tolerance-based matching (M22)."""

from __future__ import annotations

import pytest

from xpyd_acc.output_compare import MatchConfig, normalized_match


class TestMatchConfigDefaults:
    """MatchConfig should default to strict matching."""

    def test_default_config(self) -> None:
        cfg = MatchConfig()
        assert cfg.normalize_whitespace is False
        assert cfg.ignore_case is False
        assert cfg.numeric_tolerance is None

    def test_none_config_strict(self) -> None:
        assert normalized_match("hello", "hello", None) is True
        assert normalized_match("hello", "Hello", None) is False


class TestNormalizeWhitespace:
    """Test whitespace normalization."""

    def test_trailing_whitespace(self) -> None:
        cfg = MatchConfig(normalize_whitespace=True)
        assert normalized_match("hello ", "hello", cfg) is True

    def test_leading_whitespace(self) -> None:
        cfg = MatchConfig(normalize_whitespace=True)
        assert normalized_match("  hello", "hello", cfg) is True

    def test_multiple_spaces(self) -> None:
        cfg = MatchConfig(normalize_whitespace=True)
        assert normalized_match("hello   world", "hello world", cfg) is True

    def test_tabs_and_newlines(self) -> None:
        cfg = MatchConfig(normalize_whitespace=True)
        assert normalized_match("hello\t\nworld", "hello world", cfg) is True

    def test_different_content_still_fails(self) -> None:
        cfg = MatchConfig(normalize_whitespace=True)
        assert normalized_match("hello world", "hello earth", cfg) is False

    def test_disabled_whitespace(self) -> None:
        cfg = MatchConfig(normalize_whitespace=False)
        assert normalized_match("hello ", "hello", cfg) is False


class TestIgnoreCase:
    """Test case-insensitive matching."""

    def test_case_match(self) -> None:
        cfg = MatchConfig(ignore_case=True)
        assert normalized_match("Hello World", "hello world", cfg) is True

    def test_mixed_case(self) -> None:
        cfg = MatchConfig(ignore_case=True)
        assert normalized_match("ABC", "abc", cfg) is True

    def test_different_content(self) -> None:
        cfg = MatchConfig(ignore_case=True)
        assert normalized_match("abc", "xyz", cfg) is False

    def test_disabled_case(self) -> None:
        cfg = MatchConfig(ignore_case=False)
        assert normalized_match("Hello", "hello", cfg) is False


class TestNumericTolerance:
    """Test numeric tolerance matching."""

    def test_exact_numbers(self) -> None:
        cfg = MatchConfig(numeric_tolerance=0.01)
        assert normalized_match("result: 3.14", "result: 3.14", cfg) is True

    def test_within_tolerance(self) -> None:
        cfg = MatchConfig(numeric_tolerance=0.01)
        assert normalized_match("result: 3.14", "result: 3.145", cfg) is True

    def test_outside_tolerance(self) -> None:
        cfg = MatchConfig(numeric_tolerance=0.001)
        assert normalized_match("result: 3.14", "result: 3.20", cfg) is False

    def test_multiple_numbers(self) -> None:
        cfg = MatchConfig(numeric_tolerance=0.1)
        assert normalized_match("x=1.0, y=2.0", "x=1.05, y=2.08", cfg) is True

    def test_different_text_between_numbers(self) -> None:
        cfg = MatchConfig(numeric_tolerance=0.1)
        assert normalized_match("x=1.0 foo", "x=1.0 bar", cfg) is False

    def test_different_number_count(self) -> None:
        cfg = MatchConfig(numeric_tolerance=0.1)
        assert normalized_match("1.0 2.0", "1.0", cfg) is False

    def test_negative_numbers(self) -> None:
        cfg = MatchConfig(numeric_tolerance=0.01)
        assert normalized_match("value: -3.14", "value: -3.145", cfg) is True

    def test_scientific_notation(self) -> None:
        cfg = MatchConfig(numeric_tolerance=0.01)
        assert normalized_match("val: 1e-3", "val: 0.001", cfg) is True

    def test_integers(self) -> None:
        cfg = MatchConfig(numeric_tolerance=0.5)
        assert normalized_match("count: 10", "count: 10", cfg) is True


class TestCombinedTolerances:
    """Test combinations of tolerance modes."""

    def test_whitespace_and_case(self) -> None:
        cfg = MatchConfig(normalize_whitespace=True, ignore_case=True)
        assert normalized_match("  Hello  World  ", "hello world", cfg) is True

    def test_whitespace_and_numeric(self) -> None:
        cfg = MatchConfig(normalize_whitespace=True, numeric_tolerance=0.01)
        assert normalized_match("result:  3.14", "result: 3.145", cfg) is True

    def test_case_and_numeric(self) -> None:
        cfg = MatchConfig(ignore_case=True, numeric_tolerance=0.01)
        assert normalized_match("Result: 3.14", "result: 3.145", cfg) is True

    def test_all_three(self) -> None:
        cfg = MatchConfig(
            normalize_whitespace=True, ignore_case=True, numeric_tolerance=0.01,
        )
        assert normalized_match(
            "  Result:  3.14  ", "result: 3.145", cfg,
        ) is True

    def test_all_three_fail(self) -> None:
        cfg = MatchConfig(
            normalize_whitespace=True, ignore_case=True, numeric_tolerance=0.001,
        )
        assert normalized_match(
            "Result: 3.14", "result: 3.20", cfg,
        ) is False


class TestConfigToml:
    """Test TOML [matching] section parsing."""

    def test_matching_section(self, tmp_path: pytest.TempPathFactory) -> None:
        from xpyd_acc.config import load_config

        toml_file = tmp_path / "test.toml"  # type: ignore[operator]
        toml_file.write_text(
            "[matching]\n"
            "normalize_whitespace = true\n"
            "ignore_case = true\n"
            "numeric_tolerance = 0.01\n"
        )
        cfg = load_config(str(toml_file))
        assert cfg.matching.normalize_whitespace is True
        assert cfg.matching.ignore_case is True
        assert cfg.matching.numeric_tolerance == 0.01

    def test_empty_matching(self, tmp_path: pytest.TempPathFactory) -> None:
        from xpyd_acc.config import load_config

        toml_file = tmp_path / "test.toml"  # type: ignore[operator]
        toml_file.write_text("[defaults]\nmodel = 'test'\n")
        cfg = load_config(str(toml_file))
        assert cfg.matching.normalize_whitespace is False
        assert cfg.matching.ignore_case is False
        assert cfg.matching.numeric_tolerance is None

    def test_merge_cli_args(self) -> None:
        from xpyd_acc.config import AppConfig, MatchingConfig, merge_cli_args

        config = AppConfig(
            matching=MatchingConfig(
                normalize_whitespace=True,
                ignore_case=True,
                numeric_tolerance=0.05,
            ),
        )
        args = {
            "normalize_whitespace": False,
            "ignore_case": False,
            "numeric_tolerance": None,
        }
        merged = merge_cli_args(config, args, "batch-compare")
        assert merged["normalize_whitespace"] is True
        assert merged["ignore_case"] is True
        assert merged["numeric_tolerance"] == 0.05


class TestEdgeCases:
    """Edge cases for tolerance matching."""

    def test_empty_strings(self) -> None:
        cfg = MatchConfig(normalize_whitespace=True)
        assert normalized_match("", "", cfg) is True

    def test_whitespace_only(self) -> None:
        cfg = MatchConfig(normalize_whitespace=True)
        assert normalized_match("   ", "", cfg) is True

    def test_no_numbers_with_tolerance(self) -> None:
        cfg = MatchConfig(numeric_tolerance=0.1)
        assert normalized_match("hello", "hello", cfg) is True

    def test_no_numbers_different(self) -> None:
        cfg = MatchConfig(numeric_tolerance=0.1)
        assert normalized_match("hello", "world", cfg) is False
