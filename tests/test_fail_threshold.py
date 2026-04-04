"""Tests for --fail-threshold feature."""

from __future__ import annotations

import argparse

import pytest

from xpyd_acc.cli import _resolve_fail_threshold
from xpyd_acc.config import BatchConfig


class TestResolveFailThreshold:
    """Test fail threshold resolution priority chain."""

    def test_cli_flag_takes_priority(self) -> None:
        args = argparse.Namespace(fail_threshold=0.05)
        assert _resolve_fail_threshold(args, None) == 0.05

    def test_env_var_used_when_no_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XPYD_ACC_FAIL_THRESHOLD", "0.2")
        args = argparse.Namespace(fail_threshold=None)
        assert _resolve_fail_threshold(args, None) == 0.2

    def test_cli_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XPYD_ACC_FAIL_THRESHOLD", "0.2")
        args = argparse.Namespace(fail_threshold=0.05)
        assert _resolve_fail_threshold(args, None) == 0.05

    def test_none_when_nothing_set(self) -> None:
        args = argparse.Namespace(fail_threshold=None)
        assert _resolve_fail_threshold(args, None) is None

    def test_invalid_env_var_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XPYD_ACC_FAIL_THRESHOLD", "not_a_number")
        args = argparse.Namespace(fail_threshold=None)
        assert _resolve_fail_threshold(args, None) is None


class TestBatchConfigFailThreshold:
    """Test BatchConfig supports fail_threshold field."""

    def test_default_none(self) -> None:
        cfg = BatchConfig()
        assert cfg.fail_threshold is None

    def test_set_value(self) -> None:
        cfg = BatchConfig(fail_threshold=0.1)
        assert cfg.fail_threshold == 0.1
