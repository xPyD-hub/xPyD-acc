"""Tests for M39: Configuration Validation & Init Command."""

from __future__ import annotations

from pathlib import Path

import pytest

from xpyd_acc.cli import main
from xpyd_acc.config_validate import (
    generate_starter_config,
    validate_config,
)


class TestGenerateStarterConfig:
    """Tests for the init/generate functionality."""

    def test_creates_file(self, tmp_path: Path) -> None:
        output = tmp_path / "xpyd-acc.toml"
        result = generate_starter_config(output)
        assert output.exists()
        assert result == str(output)
        content = output.read_text()
        assert "[defaults]" in content
        assert "[batch]" in content
        assert "[matching]" in content

    def test_refuses_overwrite(self, tmp_path: Path) -> None:
        output = tmp_path / "xpyd-acc.toml"
        output.write_text("existing")
        with pytest.raises(FileExistsError, match="already exists"):
            generate_starter_config(output)

    def test_force_overwrite(self, tmp_path: Path) -> None:
        output = tmp_path / "xpyd-acc.toml"
        output.write_text("existing")
        generate_starter_config(output, force=True)
        assert "[defaults]" in output.read_text()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        output = tmp_path / "sub" / "dir" / "xpyd-acc.toml"
        generate_starter_config(output)
        assert output.exists()

    def test_starter_config_is_valid_toml(self, tmp_path: Path) -> None:
        output = tmp_path / "xpyd-acc.toml"
        generate_starter_config(output)
        issues = validate_config(output)
        assert issues == []


class TestValidateConfig:
    """Tests for the config validation functionality."""

    def test_valid_config(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.toml"
        cfg.write_text(
            '[defaults]\nmodel = "gpt-4"\nmax_tokens = 128\n\n'
            "[batch]\nconcurrency = 10\n"
        )
        issues = validate_config(cfg)
        assert issues == []

    def test_unknown_section(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.toml"
        cfg.write_text("[unknown_section]\nfoo = 1\n")
        issues = validate_config(cfg)
        assert any("unknown section" in i for i in issues)

    def test_unknown_key(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.toml"
        cfg.write_text("[defaults]\nnonexistent_key = 42\n")
        issues = validate_config(cfg)
        assert any("unknown key" in i for i in issues)

    def test_type_mismatch(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.toml"
        cfg.write_text("[defaults]\nmax_tokens = 'not_a_number'\n")
        issues = validate_config(cfg)
        assert any("error:" in i and "max_tokens" in i for i in issues)

    def test_int_allowed_for_float(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.toml"
        cfg.write_text("[defaults]\nretry_delay = 2\n")
        issues = validate_config(cfg)
        assert issues == []

    def test_profiles_section_freeform(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.toml"
        cfg.write_text("[profiles.greedy]\ntemperature = 0.0\nseed = 42\n")
        issues = validate_config(cfg)
        assert issues == []

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_config(tmp_path / "nonexistent.toml")

    def test_invalid_toml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad.toml"
        cfg.write_text("[defaults\nbroken")
        issues = validate_config(cfg)
        assert any("TOML parse error" in i for i in issues)

    def test_bool_type_check(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.toml"
        cfg.write_text('[matching]\nnormalize_whitespace = "yes"\n')
        issues = validate_config(cfg)
        assert any("error:" in i and "normalize_whitespace" in i for i in issues)


class TestCLIInit:
    """Tests for the xpyd-acc init CLI command."""

    def test_init_creates_file(self, tmp_path: Path) -> None:
        target = str(tmp_path / "out.toml")
        main(["init", "--output", target])
        assert Path(target).exists()

    def test_init_refuses_overwrite(self, tmp_path: Path) -> None:
        target = tmp_path / "out.toml"
        target.write_text("existing")
        with pytest.raises(SystemExit, match="1"):
            main(["init", "--output", str(target)])

    def test_init_force_overwrite(self, tmp_path: Path) -> None:
        target = tmp_path / "out.toml"
        target.write_text("existing")
        main(["init", "--output", str(target), "--force"])
        assert "[defaults]" in target.read_text()


class TestCLIConfigValidate:
    """Tests for the xpyd-acc config validate CLI command."""

    def test_validate_valid(self, tmp_path: Path) -> None:
        cfg = tmp_path / "good.toml"
        cfg.write_text('[defaults]\nmodel = "test"\n')
        with pytest.raises(SystemExit, match="0"):
            main(["config", "validate", str(cfg)])

    def test_validate_with_errors(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad.toml"
        cfg.write_text("[defaults]\nmax_tokens = 'wrong'\n")
        with pytest.raises(SystemExit, match="1"):
            main(["config", "validate", str(cfg)])

    def test_validate_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="1"):
            main(["config", "validate", "/tmp/no_such_file_xyz.toml"])
