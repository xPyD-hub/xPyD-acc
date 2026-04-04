"""Prompt template support for batch comparisons."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import yaml  # type: ignore[import-untyped]


@dataclass
class PromptTemplate:
    """A reusable prompt template with variable substitution."""

    name: str
    template: str
    description: str = ""
    variables: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Auto-detect variables from template if not provided."""
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> list[str]:
        """Extract {variable} placeholders from template string."""
        return sorted(set(re.findall(r"\{(\w+)\}", self.template)))

    def render(self, variables: dict[str, str]) -> str:
        """Render the template with the given variables.

        Args:
            variables: Mapping of variable names to values.

        Returns:
            Rendered prompt string.

        Raises:
            ValueError: If required variables are missing.
        """
        missing = [v for v in self.variables if v not in variables]
        if missing:
            msg = (
                f"Template '{self.name}' missing required variables: {', '.join(missing)}"
            )
            raise ValueError(msg)
        result = self.template
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", value)
        return result

    def validate(self) -> list[str]:
        """Validate the template and return a list of issues (empty if valid)."""
        issues: list[str] = []
        if not self.template.strip():
            issues.append("Template string is empty")
        if not self.name.strip():
            issues.append("Template name is empty")
        return issues


# Built-in templates for common evaluation formats
BUILTIN_TEMPLATES: dict[str, PromptTemplate] = {
    "gsm8k": PromptTemplate(
        name="gsm8k",
        template="Q: {question}\nA: Let's think step by step.",
        description="GSM8K math reasoning format",
    ),
    "mmlu": PromptTemplate(
        name="mmlu",
        template=(
            "The following is a multiple choice question about {subject}.\n\n"
            "{question}\n\n"
            "A. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}\n\n"
            "Answer:"
        ),
        description="MMLU multiple-choice format",
    ),
    "humaneval": PromptTemplate(
        name="humaneval",
        template=(
            "Complete the following Python function:\n\n{prompt}\n"
        ),
        description="HumanEval code generation format",
    ),
    "mt-bench": PromptTemplate(
        name="mt-bench",
        template=(
            "[Instruction]\n{instruction}\n\n"
            "[Response]"
        ),
        description="MT-Bench single-turn format",
    ),
    "simple": PromptTemplate(
        name="simple",
        template="{prompt}",
        description="Pass-through template",
    ),
}


def load_template_yaml(path: str | Path) -> PromptTemplate:
    """Load a prompt template from a YAML file.

    Expected format:
        name: my-template
        template: "Q: {question}\\nA:"
        description: Optional description
        variables: [question]  # optional, auto-detected if omitted

    Args:
        path: Path to the YAML template file.

    Returns:
        Loaded PromptTemplate.

    Raises:
        ValueError: If the file is missing required fields.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    return _parse_template_dict(data, str(path))


def load_template_toml(path: str | Path) -> PromptTemplate:
    """Load a prompt template from a TOML file.

    Expected format:
        [template]
        name = "my-template"
        template = "Q: {question}\\nA:"
        description = "Optional description"
        variables = ["question"]

    Args:
        path: Path to the TOML template file.

    Returns:
        Loaded PromptTemplate.

    Raises:
        ValueError: If the file is missing required fields.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    # Support both top-level and [template] section
    if "template" in data and isinstance(data["template"], dict):
        return _parse_template_dict(data["template"], str(path))
    return _parse_template_dict(data, str(path))


def load_template(path: str | Path) -> PromptTemplate:
    """Load a prompt template from a YAML or TOML file, auto-detecting format.

    Args:
        path: Path to the template file.

    Returns:
        Loaded PromptTemplate.

    Raises:
        ValueError: If the file format is unsupported or missing required fields.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return load_template_yaml(path)
    if suffix == ".toml":
        return load_template_toml(path)
    msg = f"Unsupported template format: {suffix} (use .yaml, .yml, or .toml)"
    raise ValueError(msg)


def get_builtin_template(name: str) -> PromptTemplate:
    """Get a built-in template by name.

    Args:
        name: Template name (e.g., 'gsm8k', 'mmlu').

    Returns:
        The built-in PromptTemplate.

    Raises:
        KeyError: If no built-in template with that name exists.
    """
    if name not in BUILTIN_TEMPLATES:
        available = ", ".join(sorted(BUILTIN_TEMPLATES))
        msg = f"Unknown built-in template '{name}'. Available: {available}"
        raise KeyError(msg)
    return BUILTIN_TEMPLATES[name]


def resolve_template(name_or_path: str) -> PromptTemplate:
    """Resolve a template by built-in name or file path.

    First checks built-in templates, then tries loading from file.

    Args:
        name_or_path: Built-in template name or path to template file.

    Returns:
        Resolved PromptTemplate.
    """
    if name_or_path in BUILTIN_TEMPLATES:
        return BUILTIN_TEMPLATES[name_or_path]
    return load_template(name_or_path)


def _parse_template_dict(data: dict[str, Any], source: str) -> PromptTemplate:
    """Parse a template from a dictionary."""
    if not isinstance(data, dict):
        msg = f"Template file {source}: expected a mapping, got {type(data).__name__}"
        raise ValueError(msg)
    if "template" not in data:
        msg = f"Template file {source}: missing required 'template' field"
        raise ValueError(msg)
    name = data.get("name", Path(source).stem)
    template_str = data["template"]
    if not isinstance(template_str, str):
        msg = f"Template file {source}: 'template' must be a string"
        raise ValueError(msg)
    description = data.get("description", "")
    variables = data.get("variables", [])
    return PromptTemplate(
        name=name,
        template=template_str,
        description=description,
        variables=variables if variables else [],
    )
