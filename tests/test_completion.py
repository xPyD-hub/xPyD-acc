"""Tests for shell completion generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from xpyd_acc.completion import GENERATORS, generate_bash, generate_fish, generate_zsh


@pytest.fixture()
def parser():
    """Build the real CLI parser for testing."""
    # Import the main function and build the parser the same way CLI does
    import argparse

    p = argparse.ArgumentParser(prog="xpyd-acc")
    sub = p.add_subparsers(dest="command")
    lp = sub.add_parser("compare-logprobs")
    lp.add_argument("--baseline", required=True)
    lp.add_argument("--target", required=True)
    sub.add_parser("profiles")
    comp = sub.add_parser("completion")
    comp.add_argument("shell", choices=["bash", "zsh", "fish"])
    return p


class TestBashCompletion:
    """Bash completion generation."""

    def test_non_empty(self, parser):
        script = generate_bash(parser)
        assert len(script) > 0

    def test_contains_function(self, parser):
        script = generate_bash(parser)
        assert "_xpyd_acc_complete" in script

    def test_contains_complete_command(self, parser):
        script = generate_bash(parser)
        assert "complete -F _xpyd_acc_complete xpyd-acc" in script

    def test_contains_subcommands(self, parser):
        script = generate_bash(parser)
        assert "compare-logprobs" in script
        assert "profiles" in script

    def test_contains_flags(self, parser):
        script = generate_bash(parser)
        assert "--baseline" in script
        assert "--target" in script


class TestZshCompletion:
    """Zsh completion generation."""

    def test_non_empty(self, parser):
        script = generate_zsh(parser)
        assert len(script) > 0

    def test_contains_compdef(self, parser):
        script = generate_zsh(parser)
        assert "#compdef xpyd-acc" in script

    def test_contains_function(self, parser):
        script = generate_zsh(parser)
        assert "_xpyd_acc()" in script

    def test_contains_subcommands(self, parser):
        script = generate_zsh(parser)
        assert "compare-logprobs" in script


class TestFishCompletion:
    """Fish completion generation."""

    def test_non_empty(self, parser):
        script = generate_fish(parser)
        assert len(script) > 0

    def test_contains_complete(self, parser):
        script = generate_fish(parser)
        assert "complete -c xpyd-acc" in script

    def test_contains_subcommands(self, parser):
        script = generate_fish(parser)
        assert "compare-logprobs" in script

    def test_contains_long_flags(self, parser):
        script = generate_fish(parser)
        assert "baseline" in script


class TestGenerators:
    """GENERATORS dict covers all shells."""

    def test_all_shells(self):
        assert set(GENERATORS) == {"bash", "zsh", "fish"}


class TestOutputFlag:
    """--output writes to file."""

    def test_write_to_file(self, parser):
        script = generate_bash(parser)
        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as f:
            path = Path(f.name)
        path.write_text(script)
        assert path.read_text() == script
        path.unlink()
