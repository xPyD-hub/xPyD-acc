"""Tests for CSV, JSON array, and JSONL dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_acc.batch_compare import load_dataset

# ── JSONL (existing format) ──────────────────────────────────────────────


def test_load_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"prompt": "hello"}\n{"id": "x", "prompt": "world", "expected": "ok"}\n'
    )
    samples = load_dataset(p)
    assert len(samples) == 2
    assert samples[0].prompt == "hello"
    assert samples[0].id == "0"
    assert samples[1].id == "x"
    assert samples[1].expected == "ok"


def test_load_jsonl_missing_prompt(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"text": "no prompt"}\n')
    with pytest.raises(ValueError, match="missing 'prompt'"):
        load_dataset(p)


# ── JSON array ───────────────────────────────────────────────────────────


def test_load_json_array(tmp_path: Path) -> None:
    p = tmp_path / "data.json"
    data = [
        {"prompt": "q1", "expected": "a1"},
        {"id": "s2", "prompt": "q2", "meta": "v"},
    ]
    p.write_text(json.dumps(data))
    samples = load_dataset(p)
    assert len(samples) == 2
    assert samples[0].prompt == "q1"
    assert samples[0].expected == "a1"
    assert samples[1].id == "s2"
    assert samples[1].metadata == {"meta": "v"}


def test_load_json_array_not_a_list(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text('{"prompt": "oops"}')
    with pytest.raises(ValueError, match="expected a JSON array"):
        load_dataset(p)


def test_load_json_array_missing_prompt(tmp_path: Path) -> None:
    p = tmp_path / "bad2.json"
    p.write_text('[{"text": "no prompt"}]')
    with pytest.raises(ValueError, match="missing 'prompt'"):
        load_dataset(p)


def test_load_json_array_non_object_item(tmp_path: Path) -> None:
    p = tmp_path / "bad3.json"
    p.write_text('["just a string"]')
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_dataset(p)


# ── CSV ──────────────────────────────────────────────────────────────────


def test_load_csv(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text("id,prompt,expected\n1,hello,world\n2,foo,bar\n")
    samples = load_dataset(p)
    assert len(samples) == 2
    assert samples[0].id == "1"
    assert samples[0].prompt == "hello"
    assert samples[0].expected == "world"
    assert samples[1].prompt == "foo"


def test_load_csv_no_prompt_column(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("id,text\n1,hello\n")
    with pytest.raises(ValueError, match="must have a 'prompt' column"):
        load_dataset(p)


def test_load_csv_empty_prompt(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("id,prompt\n1,\n")
    with pytest.raises(ValueError, match="empty or missing 'prompt'"):
        load_dataset(p)


def test_load_csv_with_metadata(tmp_path: Path) -> None:
    p = tmp_path / "meta.csv"
    p.write_text("prompt,category,difficulty\nWhat is 2+2?,math,easy\n")
    samples = load_dataset(p)
    assert len(samples) == 1
    assert samples[0].metadata == {"category": "math", "difficulty": "easy"}


# ── Auto-detection ───────────────────────────────────────────────────────


def test_unknown_extension_falls_back_to_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "data.txt"
    p.write_text('{"prompt": "from txt"}\n')
    samples = load_dataset(p)
    assert len(samples) == 1
    assert samples[0].prompt == "from txt"
