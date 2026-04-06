"""Tests for Interactive REPL (M82)."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

from xpyd_acc.repl import (
    ReplCommand,
    ReplEntry,
    ReplSession,
    format_comparison,
    format_diff,
    format_history,
    parse_command,
    run_repl,
)


class TestParseCommand:
    def test_not_a_command(self):
        assert parse_command("hello world") is None

    def test_simple_command(self):
        cmd = parse_command(":quit")
        assert cmd == ReplCommand(name="quit", args="")

    def test_command_with_args(self):
        cmd = parse_command(":set temperature=0.5")
        assert cmd == ReplCommand(name="set", args="temperature=0.5")

    def test_export_with_path(self):
        cmd = parse_command(":export /tmp/out.json")
        assert cmd == ReplCommand(name="export", args="/tmp/out.json")

    def test_empty_colon(self):
        assert parse_command(":") is None

    def test_case_insensitive(self):
        cmd = parse_command(":QUIT")
        assert cmd is not None
        assert cmd.name == "quit"


class TestReplSession:
    def test_set_temperature(self):
        s = ReplSession(baseline_url="http://a", target_url="http://b", model="m")
        msg = s.set_param("temperature", "0.7")
        assert s.temperature == 0.7
        assert "0.7" in msg

    def test_set_top_p(self):
        s = ReplSession(baseline_url="http://a", target_url="http://b", model="m")
        s.set_param("top_p", "0.9")
        assert s.top_p == 0.9

    def test_set_seed(self):
        s = ReplSession(baseline_url="http://a", target_url="http://b", model="m")
        s.set_param("seed", "42")
        assert s.seed == 42

    def test_set_max_tokens(self):
        s = ReplSession(baseline_url="http://a", target_url="http://b", model="m")
        s.set_param("max_tokens", "128")
        assert s.max_tokens == 128

    def test_set_unknown(self):
        s = ReplSession(baseline_url="http://a", target_url="http://b", model="m")
        msg = s.set_param("unknown_param", "val")
        assert "Unknown" in msg

    def test_export_json_empty(self):
        s = ReplSession(baseline_url="http://a", target_url="http://b", model="m")
        data = s.export_json()
        assert data["baseline_url"] == "http://a"
        assert data["entries"] == []

    def test_export_json_with_entries(self):
        s = ReplSession(baseline_url="http://a", target_url="http://b", model="m")
        s.history.append(ReplEntry(
            prompt="hello", baseline_output="hi", target_output="hi",
            match=True, timestamp=1.0,
        ))
        data = s.export_json()
        assert len(data["entries"]) == 1
        assert data["entries"][0]["match"] is True


class TestFormatFunctions:
    def test_format_comparison_match(self):
        e = ReplEntry(prompt="p", baseline_output="out", target_output="out", match=True)
        text = format_comparison(e)
        assert "MATCH" in text

    def test_format_comparison_diverge(self):
        e = ReplEntry(prompt="p", baseline_output="a", target_output="b", match=False)
        text = format_comparison(e)
        assert "DIVERGE" in text

    def test_format_diff(self):
        e = ReplEntry(
            prompt="p", baseline_output="hello world",
            target_output="hello earth", match=False,
        )
        text = format_diff(e)
        assert "world" in text
        assert "earth" in text

    def test_format_diff_empty(self):
        e = ReplEntry(prompt="p", baseline_output="", target_output="", match=True)
        text = format_diff(e)
        assert "empty" in text

    def test_format_history_empty(self):
        s = ReplSession(baseline_url="http://a", target_url="http://b", model="m")
        text = format_history(s)
        assert "no history" in text

    def test_format_history_with_entries(self):
        s = ReplSession(baseline_url="http://a", target_url="http://b", model="m")
        s.history.append(ReplEntry(
            prompt="hello", baseline_output="a",
            target_output="a", match=True,
        ))
        s.history.append(ReplEntry(
            prompt="bye", baseline_output="a",
            target_output="b", match=False,
        ))
        text = format_history(s)
        assert "✅" in text
        assert "❌" in text


class TestReplLoop:
    def test_quit_command(self):
        inputs = iter([":quit"])
        output = []
        asyncio.run(run_repl(
            "http://a", "http://b", "m",
            print_fn=output.append,
            input_fn=lambda _: next(inputs),
        ))
        assert any("Bye" in line for line in output)

    def test_help_command(self):
        inputs = iter([":help", ":quit"])
        output = []
        asyncio.run(run_repl(
            "http://a", "http://b", "m",
            print_fn=output.append,
            input_fn=lambda _: next(inputs),
        ))
        assert any("logprobs" in line for line in output)

    def test_set_command(self):
        inputs = iter([":set temperature=0.3", ":quit"])
        output = []
        asyncio.run(run_repl(
            "http://a", "http://b", "m",
            print_fn=output.append,
            input_fn=lambda _: next(inputs),
        ))
        assert any("0.3" in line for line in output)

    def test_logprobs_toggle(self):
        inputs = iter([":logprobs", ":logprobs", ":quit"])
        output = []
        asyncio.run(run_repl(
            "http://a", "http://b", "m",
            print_fn=output.append,
            input_fn=lambda _: next(inputs),
        ))
        assert any("on" in line for line in output)
        assert any("off" in line for line in output)

    def test_diff_no_history(self):
        inputs = iter([":diff", ":quit"])
        output = []
        asyncio.run(run_repl(
            "http://a", "http://b", "m",
            print_fn=output.append,
            input_fn=lambda _: next(inputs),
        ))
        assert any("no results" in line for line in output)

    def test_history_command(self):
        inputs = iter([":history", ":quit"])
        output = []
        asyncio.run(run_repl(
            "http://a", "http://b", "m",
            print_fn=output.append,
            input_fn=lambda _: next(inputs),
        ))
        assert any("no history" in line for line in output)

    def test_export_command(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            inputs = iter([f":export {path}", ":quit"])
            output = []
            asyncio.run(run_repl(
                "http://a", "http://b", "m",
                print_fn=output.append,
                input_fn=lambda _: next(inputs),
            ))
            with open(path) as f:
                data = json.load(f)
            assert data["baseline_url"] == "http://a"
        finally:
            os.unlink(path)

    def test_eof_exits(self):
        output = []
        asyncio.run(run_repl(
            "http://a", "http://b", "m",
            print_fn=output.append,
            input_fn=lambda _: (_ for _ in ()).throw(EOFError),
        ))
        assert any("Bye" in line for line in output)

    def test_unknown_command(self):
        inputs = iter([":foobar", ":quit"])
        output = []
        asyncio.run(run_repl(
            "http://a", "http://b", "m",
            print_fn=output.append,
            input_fn=lambda _: next(inputs),
        ))
        assert any("Unknown command" in line for line in output)

    def test_empty_input_skipped(self):
        inputs = iter(["", "  ", ":quit"])
        output = []
        asyncio.run(run_repl(
            "http://a", "http://b", "m",
            print_fn=output.append,
            input_fn=lambda _: next(inputs),
        ))
        assert any("Bye" in line for line in output)

    def test_set_without_equals(self):
        inputs = iter([":set foo", ":quit"])
        output = []
        asyncio.run(run_repl(
            "http://a", "http://b", "m",
            print_fn=output.append,
            input_fn=lambda _: next(inputs),
        ))
        assert any("Usage" in line for line in output)

    def test_export_without_path(self):
        inputs = iter([":export", ":quit"])
        output = []
        asyncio.run(run_repl(
            "http://a", "http://b", "m",
            print_fn=output.append,
            input_fn=lambda _: next(inputs),
        ))
        assert any("Usage" in line for line in output)


class TestCLIIntegration:
    def test_repl_parser_exists(self):
        """Verify the repl subcommand is registered."""
        from xpyd_acc.cli import main

        # Just check the parser doesn't crash on --help
        try:
            main(["repl", "--help"])
        except SystemExit as e:
            assert e.code == 0
