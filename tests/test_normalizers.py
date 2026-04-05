"""Tests for custom output normalizers."""

from __future__ import annotations

import pytest

from xpyd_acc.normalizers import (
    apply_normalizers,
    load_normalizer,
    normalize_json,
    normalize_numbers,
    resolve_normalizers,
    strip_thinking_tags,
)


class TestStripThinkingTags:
    def test_removes_think_tag(self):
        text = "<think>some reasoning</think>The answer is 42."
        assert strip_thinking_tags(text) == "The answer is 42."

    def test_removes_thinking_tag(self):
        text = "<thinking>step by step</thinking> Result: 5"
        assert strip_thinking_tags(text) == "Result: 5"

    def test_removes_thought_tag(self):
        text = "<thought>hmm</thought>Yes"
        assert strip_thinking_tags(text) == "Yes"

    def test_removes_reasoning_tag(self):
        text = "<reasoning>logic</reasoning>Done"
        assert strip_thinking_tags(text) == "Done"

    def test_case_insensitive(self):
        text = "<THINK>stuff</THINK>answer"
        assert strip_thinking_tags(text) == "answer"

    def test_multiline(self):
        text = "<think>\nline1\nline2\n</think>\nResult"
        assert strip_thinking_tags(text) == "Result"

    def test_no_tags(self):
        text = "No tags here"
        assert strip_thinking_tags(text) == "No tags here"

    def test_multiple_tags(self):
        text = "<think>a</think>middle<think>b</think>end"
        assert strip_thinking_tags(text) == "middleend"


class TestNormalizeJson:
    def test_valid_json(self):
        text = '{"b": 2, "a": 1}'
        result = normalize_json(text)
        assert result == '{\n  "a": 1,\n  "b": 2\n}'

    def test_not_json(self):
        text = "just plain text"
        assert normalize_json(text) == text

    def test_json_array(self):
        text = "[3, 1, 2]"
        result = normalize_json(text)
        assert result == "[\n  3,\n  1,\n  2\n]"


class TestNormalizeNumbers:
    def test_rounds_floats(self):
        text = "value is 3.141592653589793"
        result = normalize_numbers(text, precision=2)
        assert "3.14" in result

    def test_leaves_integers(self):
        text = "count is 42"
        assert normalize_numbers(text) == "count is 42"

    def test_scientific_notation(self):
        text = "1.23e-4"
        result = normalize_numbers(text, precision=6)
        assert result == str(round(1.23e-4, 6))

    def test_no_numbers(self):
        text = "no numbers here"
        assert normalize_numbers(text) == "no numbers here"


class TestLoadNormalizer:
    def test_builtin_by_name(self):
        fn = load_normalizer("strip_thinking_tags")
        assert fn is strip_thinking_tags

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown normalizer"):
            load_normalizer("nonexistent")

    def test_module_function(self):
        fn = load_normalizer("json:dumps")
        import json
        assert fn is json.dumps

    def test_module_not_found(self):
        with pytest.raises(ValueError, match="Cannot import"):
            load_normalizer("nonexistent_module_xyz:func")

    def test_function_not_found(self):
        with pytest.raises(ValueError, match="has no attribute"):
            load_normalizer("json:nonexistent_func_xyz")


class TestResolveNormalizers:
    def test_resolves_list(self):
        fns = resolve_normalizers(["strip_thinking_tags", "normalize_json"])
        assert len(fns) == 2
        assert fns[0] is strip_thinking_tags
        assert fns[1] is normalize_json


class TestApplyNormalizers:
    def test_chain(self):
        text = "<think>reason</think>{\"b\":2,\"a\":1}"
        fns = [strip_thinking_tags, normalize_json]
        result = apply_normalizers(text, fns)
        assert result == '{\n  "a": 1,\n  "b": 2\n}'

    def test_empty_list(self):
        text = "unchanged"
        assert apply_normalizers(text, []) == "unchanged"


class TestNormalizedMatchIntegration:
    """Test normalizers integration with normalized_match."""

    def test_match_with_normalizer(self):
        from xpyd_acc.output_compare import normalized_match

        text1 = "<think>reasoning</think>answer"
        text2 = "answer"
        assert not normalized_match(text1, text2)
        assert normalized_match(text1, text2, normalizers=[strip_thinking_tags])

    def test_match_with_config_and_normalizer(self):
        from xpyd_acc.output_compare import MatchConfig, normalized_match

        text1 = "<think>x</think>  HELLO  "
        text2 = "hello"
        config = MatchConfig(normalize_whitespace=True, ignore_case=True)
        assert normalized_match(text1, text2, config, normalizers=[strip_thinking_tags])
