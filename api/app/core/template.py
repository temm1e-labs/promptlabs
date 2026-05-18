"""Mustache-style {{var}} substitution for prompt templates.

Design notes:
- Variable names follow Python identifier rules: [a-zA-Z_][a-zA-Z0-9_]*
- Surrounding whitespace inside braces is allowed: {{ name }} == {{name}}
- Missing variables render as empty string (lenient by default) or raise (strict)
- No escaping, conditionals, or loops — keep it simple. Switch to Jinja later if needed.
"""

from __future__ import annotations

import re
from typing import Any

VAR_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class MissingVariableError(KeyError):
    """Raised by render(..., strict=True) when a template variable has no value."""


def extract_variables(template: str) -> list[str]:
    """Return variable names in first-appearance order, deduplicated."""
    seen: set[str] = set()
    result: list[str] = []
    for match in VAR_PATTERN.finditer(template):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def render(template: str, variables: dict[str, Any], strict: bool = False) -> str:
    """Substitute {{var}} placeholders with values from ``variables``.

    Missing variables render as empty string unless ``strict=True``,
    in which case ``MissingVariableError`` is raised.
    """

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in variables:
            value = variables[name]
            return str(value) if value is not None else ""
        if strict:
            raise MissingVariableError(name)
        return ""

    return VAR_PATTERN.sub(_sub, template)


def validate(template: str, variables: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (missing, extra) variable name lists, sorted.

    - ``missing``: variables the template references but not present in ``variables``
    - ``extra``: keys in ``variables`` that the template doesn't reference
    """
    needed = set(extract_variables(template))
    provided = set(variables.keys())
    return sorted(needed - provided), sorted(provided - needed)


def has_variables(template: str) -> bool:
    return VAR_PATTERN.search(template) is not None
