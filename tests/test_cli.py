"""Smoke tests for CLI."""

import subprocess
import sys


def test_help():
    result = subprocess.run(
        [sys.executable, "-m", "xpyd_acc.cli"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
