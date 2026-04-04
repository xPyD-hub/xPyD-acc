"""Tests for xpyd_acc.log module."""

from __future__ import annotations

import logging

from xpyd_acc.log import get_logger, setup_logging


class TestSetupLogging:
    """Tests for setup_logging()."""

    def test_default_level_is_warning(self) -> None:
        setup_logging(0)
        logger = logging.getLogger("xpyd_acc")
        assert logger.level == logging.WARNING

    def test_verbose_sets_info(self) -> None:
        setup_logging(1)
        logger = logging.getLogger("xpyd_acc")
        assert logger.level == logging.INFO

    def test_double_verbose_sets_debug(self) -> None:
        setup_logging(2)
        logger = logging.getLogger("xpyd_acc")
        assert logger.level == logging.DEBUG

    def test_quiet_sets_error(self) -> None:
        setup_logging(-1)
        logger = logging.getLogger("xpyd_acc")
        assert logger.level == logging.ERROR

    def test_get_logger_returns_child(self) -> None:
        child = get_logger("test_child")
        assert child.name == "xpyd_acc.test_child"

    def test_handlers_not_duplicated(self) -> None:
        setup_logging(0)
        setup_logging(0)
        logger = logging.getLogger("xpyd_acc")
        assert len(logger.handlers) == 1


class TestCLIVerbosity:
    """Tests for --verbose / --quiet CLI integration."""

    def test_verbose_flag_accepted(self) -> None:
        from xpyd_acc.cli import main

        # Just verify -v doesn't crash before subcommand check
        # (will exit due to no subcommand, but should parse -v)
        try:
            main(["-v", "--version"])
        except SystemExit:
            pass  # --version causes SystemExit(0)

    def test_quiet_flag_accepted(self) -> None:
        from xpyd_acc.cli import main
        try:
            main(["-q", "--version"])
        except SystemExit:
            pass

    def test_verbose_and_quiet_mutually_exclusive(self) -> None:
        import pytest

        from xpyd_acc.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["-v", "-q"])
        assert exc_info.value.code != 0
