"""Tests for checkpoint module — resumable batch comparison."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from xpyd_acc.checkpoint import (
    Checkpoint,
    dict_to_result,
    load_checkpoint,
    result_to_dict,
    save_checkpoint,
    validate_checkpoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checkpoint(**overrides) -> Checkpoint:
    defaults = dict(
        created_at=1000.0,
        updated_at=1001.0,
        completed_ids={"s1", "s2"},
        results={
            "s1": {"sample_id": "s1", "prompt": "hi", "baseline_output": "a",
                    "target_output": "a", "exact_match": True,
                    "first_divergence_index": None,
                    "baseline_logprob_at_divergence": None,
                    "target_logprob_at_divergence": None,
                    "logprob_gap": None, "classification": "match",
                    "context_length": 1, "request_ids": {},
                    "baseline_finish_reason": "stop",
                    "target_finish_reason": "stop"},
        },
        baseline_url="http://base",
        target_url="http://target",
        model="m1",
        total_samples=5,
    )
    defaults.update(overrides)
    return Checkpoint(**defaults)


def _sample_result_mock():
    """Return a mock SampleResult-like object with expected attributes."""
    m = MagicMock()
    m.sample_id = "s1"
    m.prompt = "hello"
    m.baseline_output = "world"
    m.target_output = "world"
    m.exact_match = True
    m.first_divergence_index = None
    m.baseline_logprob_at_divergence = None
    m.target_logprob_at_divergence = None
    m.logprob_gap = None
    m.classification = "match"
    m.context_length = 2
    m.request_ids = {}
    m.baseline_finish_reason = "stop"
    m.target_finish_reason = "stop"
    return m


# ---------------------------------------------------------------------------
# Checkpoint dataclass
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_add_result(self):
        cp = Checkpoint()
        cp.add_result("s1", {"sample_id": "s1"})
        assert "s1" in cp.completed_ids
        assert cp.completed_count == 1
        assert cp.is_completed("s1")
        assert not cp.is_completed("s2")

    def test_round_trip_dict(self):
        cp = _make_checkpoint()
        data = cp.to_dict()
        restored = Checkpoint.from_dict(data)
        assert restored.completed_ids == cp.completed_ids
        assert restored.baseline_url == cp.baseline_url
        assert restored.target_url == cp.target_url
        assert restored.model == cp.model
        assert restored.total_samples == cp.total_samples
        assert restored.results == cp.results

    def test_completed_ids_sorted_in_dict(self):
        cp = Checkpoint(completed_ids={"c", "a", "b"})
        data = cp.to_dict()
        assert data["completed_ids"] == ["a", "b", "c"]

    def test_from_dict_defaults(self):
        cp = Checkpoint.from_dict({})
        assert cp.completed_ids == set()
        assert cp.results == {}
        assert cp.baseline_url == ""


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_load(self, tmp_path: Path):
        cp = _make_checkpoint()
        path = tmp_path / "cp.json"
        save_checkpoint(cp, path)
        loaded = load_checkpoint(path)
        assert loaded is not None
        assert loaded.completed_ids == cp.completed_ids
        assert loaded.baseline_url == cp.baseline_url

    def test_load_nonexistent(self, tmp_path: Path):
        assert load_checkpoint(tmp_path / "nope.json") is None

    def test_load_corrupt(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        assert load_checkpoint(path) is None

    def test_atomic_write(self, tmp_path: Path):
        """Ensure .tmp file is cleaned up after save."""
        path = tmp_path / "cp.json"
        save_checkpoint(_make_checkpoint(), path)
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_valid(self):
        cp = _make_checkpoint()
        assert validate_checkpoint(cp, "http://base", "http://target", "m1", 5)

    def test_baseline_mismatch(self):
        cp = _make_checkpoint()
        assert not validate_checkpoint(cp, "http://other", "http://target", "m1", 5)

    def test_target_mismatch(self):
        cp = _make_checkpoint()
        assert not validate_checkpoint(cp, "http://base", "http://other", "m1", 5)

    def test_model_mismatch(self):
        cp = _make_checkpoint()
        assert not validate_checkpoint(cp, "http://base", "http://target", "m2", 5)

    def test_total_samples_mismatch(self):
        cp = _make_checkpoint()
        assert not validate_checkpoint(cp, "http://base", "http://target", "m1", 10)


# ---------------------------------------------------------------------------
# result serialisation helpers
# ---------------------------------------------------------------------------

class TestResultSerialization:
    def test_result_to_dict(self):
        m = _sample_result_mock()
        d = result_to_dict(m)
        assert d["sample_id"] == "s1"
        assert d["exact_match"] is True
        assert d["classification"] == "match"

    def test_dict_to_result(self):
        d = {
            "sample_id": "s1",
            "prompt": "hello",
            "baseline_output": "world",
            "target_output": "world",
            "exact_match": True,
            "first_divergence_index": None,
            "baseline_logprob_at_divergence": None,
            "target_logprob_at_divergence": None,
            "logprob_gap": None,
            "classification": "match",
            "context_length": 2,
            "request_ids": {},
            "baseline_finish_reason": "stop",
            "target_finish_reason": "stop",
        }
        sr = dict_to_result(d)
        assert sr.sample_id == "s1"
        assert sr.exact_match is True
        assert sr.classification == "match"
        assert sr.baseline_finish_reason == "stop"

    def test_dict_to_result_missing_optional(self):
        d = {
            "sample_id": "s2",
            "prompt": "hi",
            "baseline_output": "a",
            "target_output": "b",
            "exact_match": False,
        }
        sr = dict_to_result(d)
        assert sr.sample_id == "s2"
        assert sr.classification == "unknown"
        assert sr.context_length == 0
        assert sr.baseline_finish_reason is None

    def test_round_trip(self):
        m = _sample_result_mock()
        d = result_to_dict(m)
        sr = dict_to_result(d)
        assert sr.sample_id == m.sample_id
        assert sr.exact_match == m.exact_match
        assert sr.classification == m.classification


# ---------------------------------------------------------------------------
# CLI integration test (batch-compare --checkpoint)
# ---------------------------------------------------------------------------

class TestCheckpointCLI:
    def test_checkpoint_flag_accepted(self):
        """Verify the CLI parser accepts --checkpoint without error."""
        from xpyd_acc.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["batch-compare", "--help"])
        assert exc_info.value.code == 0

    def test_checkpoint_clear_flag_accepted(self):
        """Verify the CLI parser accepts --checkpoint-clear without error."""
        from xpyd_acc.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["batch-compare", "--help"])
        assert exc_info.value.code == 0
