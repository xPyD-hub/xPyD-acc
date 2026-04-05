"""Custom output normalizers for pre-comparison text transformation.

Normalizers are functions that take a string and return a normalized string.
They are applied before comparison (after existing tolerance matching in
MatchConfig).

Built-in normalizers:
- strip_thinking_tags: Remove <think>...</think> and similar tags
- normalize_json: Pretty-print JSON strings for canonical comparison
- normalize_numbers: Round floats to configurable decimal places

Custom normalizers can be loaded from Python modules using the
``module:function`` syntax.
"""

from __future__ import annotations

import importlib
import json
import re
from typing import Callable

from xpyd_acc.log import get_logger

logger = get_logger("normalizers")

# Type alias for normalizer functions
Normalizer = Callable[[str], str]

# Regex for <think>...</think>, <thinking>...</thinking>, etc.
_THINKING_TAG_RE = re.compile(
    r"<(think|thinking|thought|reasoning)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

# Default decimal places for normalize_numbers
_DEFAULT_PRECISION = 6


def strip_thinking_tags(text: str) -> str:
    """Remove ``<think>...</think>`` and similar reasoning tags.

    Handles ``<think>``, ``<thinking>``, ``<thought>``, ``<reasoning>``
    (case-insensitive). Strips leading/trailing whitespace after removal.
    """
    result = _THINKING_TAG_RE.sub("", text)
    return result.strip()


def normalize_json(text: str) -> str:
    """Pretty-print JSON strings for canonical comparison.

    If the text is valid JSON, returns a canonically formatted version
    (sorted keys, 2-space indent). If not valid JSON, returns the text
    unchanged.
    """
    try:
        parsed = json.loads(text)
        return json.dumps(parsed, sort_keys=True, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return text


def normalize_numbers(text: str, *, precision: int = _DEFAULT_PRECISION) -> str:
    """Round floating-point numbers in the text to *precision* decimal places.

    Integer-like numbers (no decimal point) are left unchanged.
    """
    def _round_match(m: re.Match[str]) -> str:
        s = m.group(0)
        if "." not in s and "e" not in s.lower():
            return s  # integer, leave as-is
        try:
            return str(round(float(s), precision))
        except ValueError:
            return s

    return re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?").sub(_round_match, text)


# Registry of built-in normalizers
BUILTIN_NORMALIZERS: dict[str, Normalizer] = {
    "strip_thinking_tags": strip_thinking_tags,
    "normalize_json": normalize_json,
    "normalize_numbers": normalize_numbers,
}


def load_normalizer(spec: str) -> Normalizer:
    """Load a normalizer by name or ``module:function`` spec.

    Built-in names are resolved first. If *spec* contains a colon, it is
    treated as ``module:function`` and imported dynamically.
    """
    if spec in BUILTIN_NORMALIZERS:
        return BUILTIN_NORMALIZERS[spec]

    if ":" in spec:
        module_path, func_name = spec.rsplit(":", 1)
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            msg = f"Cannot import normalizer module '{module_path}': {exc}"
            raise ValueError(msg) from exc
        func = getattr(mod, func_name, None)
        if func is None:
            msg = f"Module '{module_path}' has no attribute '{func_name}'"
            raise ValueError(msg)
        if not callable(func):
            msg = f"'{module_path}:{func_name}' is not callable"
            raise ValueError(msg)
        return func

    msg = (
        f"Unknown normalizer '{spec}'. "
        f"Built-in: {', '.join(sorted(BUILTIN_NORMALIZERS))}. "
        f"Or use 'module:function' for custom normalizers."
    )
    raise ValueError(msg)


def resolve_normalizers(specs: list[str]) -> list[Normalizer]:
    """Resolve a list of normalizer specs into callable normalizers."""
    return [load_normalizer(s) for s in specs]


def apply_normalizers(text: str, normalizers: list[Normalizer]) -> str:
    """Apply a chain of normalizers to *text* in order."""
    result = text
    for norm in normalizers:
        result = norm(result)
    return result
