"""Tests for output comparison utilities."""

from __future__ import annotations

from xpyd_acc.output_compare import (
    OutputComparator,
    levenshtein_distance,
    reassemble_streaming_chunks,
    tokenize_simple,
)


class TestLevenshteinDistance:
    def test_identical(self) -> None:
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty(self) -> None:
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_single_edit(self) -> None:
        assert levenshtein_distance("cat", "hat") == 1
        assert levenshtein_distance("cat", "cats") == 1
        assert levenshtein_distance("cats", "cat") == 1

    def test_multiple_edits(self) -> None:
        assert levenshtein_distance("kitten", "sitting") == 3


class TestTokenizeSimple:
    def test_basic(self) -> None:
        assert tokenize_simple("hello world") == ["hello", "world"]

    def test_empty(self) -> None:
        assert tokenize_simple("") == []

    def test_punctuation(self) -> None:
        assert tokenize_simple("hello, world!") == ["hello,", "world!"]


class TestReassembleStreamingChunks:
    def test_basic(self) -> None:
        assert reassemble_streaming_chunks(["hel", "lo ", "world"]) == "hello world"

    def test_empty(self) -> None:
        assert reassemble_streaming_chunks([]) == ""

    def test_single(self) -> None:
        assert reassemble_streaming_chunks(["full text"]) == "full text"


class TestOutputComparator:
    def test_exact_match(self) -> None:
        comp = OutputComparator()
        report = comp.compare("hello world", "hello world")
        assert report.exact_match is True
        assert report.edit_distance == 0
        assert report.normalized_edit_distance == 0.0

    def test_mismatch(self) -> None:
        comp = OutputComparator()
        report = comp.compare("the cat sat", "the dog sat")
        assert report.exact_match is False
        assert report.edit_distance > 0
        # Token diff should show "cat" → "dog"
        replace_diffs = [d for d in report.token_diffs if d.tag == "replace"]
        assert len(replace_diffs) == 1
        assert replace_diffs[0].baseline_tokens == ["cat"]
        assert replace_diffs[0].target_tokens == ["dog"]

    def test_empty_inputs(self) -> None:
        comp = OutputComparator()
        report = comp.compare("", "")
        assert report.exact_match is True
        assert report.edit_distance == 0

    def test_semantic_similarity(self) -> None:
        comp = OutputComparator()
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [1.0, 0.0, 0.0]
        report = comp.compare("a", "b", baseline_embeddings=emb_a, target_embeddings=emb_b)
        assert report.semantic_similarity is not None
        assert abs(report.semantic_similarity - 1.0) < 1e-6

    def test_semantic_similarity_orthogonal(self) -> None:
        comp = OutputComparator()
        emb_a = [1.0, 0.0]
        emb_b = [0.0, 1.0]
        report = comp.compare("a", "b", baseline_embeddings=emb_a, target_embeddings=emb_b)
        assert report.semantic_similarity is not None
        assert abs(report.semantic_similarity) < 1e-6

    def test_no_embeddings(self) -> None:
        comp = OutputComparator()
        report = comp.compare("a", "b")
        assert report.semantic_similarity is None

    def test_streaming_compare(self) -> None:
        comp = OutputComparator()
        report = comp.compare_streaming(
            ["hello ", "world"],
            ["hello ", "world"],
        )
        assert report.exact_match is True

    def test_streaming_mismatch(self) -> None:
        comp = OutputComparator()
        report = comp.compare_streaming(
            ["the ", "cat"],
            ["the ", "dog"],
        )
        assert report.exact_match is False

    def test_format_report_match(self) -> None:
        comp = OutputComparator()
        report = comp.compare("same text", "same text")
        text = OutputComparator.format_report(report)
        assert "Exact match: Yes" in text

    def test_format_report_mismatch(self) -> None:
        comp = OutputComparator()
        report = comp.compare("hello world", "hello mars")
        text = OutputComparator.format_report(report)
        assert "Exact match: No" in text
        assert "REPLACE" in text

    def test_summary(self) -> None:
        comp = OutputComparator()
        report = comp.compare("a", "a")
        assert "EXACT MATCH" in report.summary()
        report2 = comp.compare("a", "b")
        assert "MISMATCH" in report2.summary()
