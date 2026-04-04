"""Tests for prompt template module."""

from __future__ import annotations

import pytest

from xpyd_acc.templates import (
    BUILTIN_TEMPLATES,
    PromptTemplate,
    get_builtin_template,
    load_template,
    load_template_toml,
    load_template_yaml,
    resolve_template,
)


class TestPromptTemplate:
    """Test PromptTemplate dataclass."""

    def test_render_simple(self) -> None:
        t = PromptTemplate(name="test", template="Hello {name}!")
        assert t.render({"name": "World"}) == "Hello World!"

    def test_render_multiple_variables(self) -> None:
        t = PromptTemplate(name="test", template="Q: {question}\nA: {answer}")
        result = t.render({"question": "What?", "answer": "Yes"})
        assert result == "Q: What?\nA: Yes"

    def test_render_missing_variable(self) -> None:
        t = PromptTemplate(name="test", template="Hello {name} from {place}!")
        with pytest.raises(ValueError, match="missing required variables.*place"):
            t.render({"name": "World"})

    def test_auto_detect_variables(self) -> None:
        t = PromptTemplate(name="test", template="{a} and {b} and {a}")
        assert t.variables == ["a", "b"]

    def test_explicit_variables(self) -> None:
        t = PromptTemplate(name="test", template="{x}", variables=["x"])
        assert t.variables == ["x"]

    def test_validate_empty_template(self) -> None:
        t = PromptTemplate(name="test", template="")
        issues = t.validate()
        assert any("empty" in i.lower() for i in issues)

    def test_validate_empty_name(self) -> None:
        t = PromptTemplate(name="", template="hello")
        issues = t.validate()
        assert any("name" in i.lower() for i in issues)

    def test_validate_ok(self) -> None:
        t = PromptTemplate(name="test", template="hello {x}")
        assert t.validate() == []

    def test_render_extra_variables_ignored(self) -> None:
        t = PromptTemplate(name="test", template="Hi {name}")
        result = t.render({"name": "Bob", "extra": "ignored"})
        assert result == "Hi Bob"


class TestBuiltinTemplates:
    """Test built-in templates."""

    def test_gsm8k(self) -> None:
        t = get_builtin_template("gsm8k")
        result = t.render({"question": "What is 2+2?"})
        assert "What is 2+2?" in result
        assert "step by step" in result

    def test_mmlu(self) -> None:
        t = get_builtin_template("mmlu")
        result = t.render({
            "subject": "math",
            "question": "What is pi?",
            "choice_a": "3.14",
            "choice_b": "2.71",
            "choice_c": "1.41",
            "choice_d": "1.61",
        })
        assert "A. 3.14" in result
        assert "math" in result

    def test_simple(self) -> None:
        t = get_builtin_template("simple")
        assert t.render({"prompt": "hello"}) == "hello"

    def test_unknown_builtin(self) -> None:
        with pytest.raises(KeyError, match="Unknown built-in"):
            get_builtin_template("nonexistent")

    def test_all_builtins_valid(self) -> None:
        for name, t in BUILTIN_TEMPLATES.items():
            assert t.validate() == [], f"Built-in '{name}' has validation issues"


class TestLoadTemplate:
    """Test template file loading."""

    def test_load_yaml(self, tmp_path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text("name: my-template\ntemplate: 'Q: {question}'\n")
        t = load_template_yaml(f)
        assert t.name == "my-template"
        assert t.render({"question": "hi"}) == "Q: hi"

    def test_load_toml(self, tmp_path) -> None:
        f = tmp_path / "test.toml"
        f.write_text('[template]\nname = "my-template"\ntemplate = "Q: {question}"\n')
        t = load_template_toml(f)
        assert t.name == "my-template"
        assert t.render({"question": "hi"}) == "Q: hi"

    def test_load_toml_toplevel(self, tmp_path) -> None:
        f = tmp_path / "test.toml"
        f.write_text('name = "flat"\ntemplate = "Hello {name}"\n')
        t = load_template_toml(f)
        assert t.name == "flat"

    def test_load_auto_yaml(self, tmp_path) -> None:
        f = tmp_path / "test.yml"
        f.write_text("name: auto\ntemplate: '{prompt}'\n")
        t = load_template(f)
        assert t.name == "auto"

    def test_load_auto_toml(self, tmp_path) -> None:
        f = tmp_path / "test.toml"
        f.write_text('name = "auto"\ntemplate = "{prompt}"\n')
        t = load_template(f)
        assert t.name == "auto"

    def test_load_unsupported_format(self, tmp_path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported template format"):
            load_template(f)

    def test_load_missing_template_field(self, tmp_path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text("name: broken\n")
        with pytest.raises(ValueError, match="missing required 'template'"):
            load_template_yaml(f)

    def test_load_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_template_yaml("/nonexistent/path.yaml")


class TestResolveTemplate:
    """Test resolve_template function."""

    def test_resolve_builtin(self) -> None:
        t = resolve_template("gsm8k")
        assert t.name == "gsm8k"

    def test_resolve_file(self, tmp_path) -> None:
        f = tmp_path / "custom.yaml"
        f.write_text("name: custom\ntemplate: '{prompt}'\n")
        t = resolve_template(str(f))
        assert t.name == "custom"
