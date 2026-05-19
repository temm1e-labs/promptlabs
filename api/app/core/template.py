"""Multi-syntax variable detection and rendering for prompt templates.

Auto-detects the dominant placeholder syntax per template and renders
accordingly. Supports:

  - ``{{var}}``    Jinja2 / Mustache / Handlebars
  - ``{var}``      Python ``str.format()``
  - ``${var}``     POSIX shell / JS template literal
  - ``%(var)s``    Python old-style percent formatting

Disambiguation is the hard part:
  * ``{{ "json": "value" }}`` is a sample-output block, NOT a Jinja2 variable.
    Detected by classifying the contents of every ``{{...}}`` pair — only
    single-identifier (optional Jinja2 filter chain) contents count as
    variables. JSON-ish content, multi-line content, anything with quotes
    or colons → marked as a sample-output zone and rendered as literal
    single-braced text.
  * ``{var}`` inside ``{{...}}`` is treated as Python's ``.format()``
    escape semantics — only counted as a placeholder in FORMAT mode and
    only when outside every ``{{...}}`` span.
  * Fenced code blocks and ``{% raw %}...{% endraw %}`` regions are
    masked entirely so example-code-in-docs doesn't get parsed.

Public API:
  ``detect(template)`` → ``DetectionResult``
  ``extract_variables(template)`` → ``list[str]``    (legacy)
  ``render(template, vars, ...)`` → ``str``
  ``validate(template, vars)`` → ``(missing, extra)``
  ``has_variables(template)`` → ``bool``
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ─── module-level regexes (compiled once) ────────────────────────────────

_IDENT = r"[a-zA-Z_][a-zA-Z0-9_]*"

# A {{...}} pair (non-greedy, DOTALL). Used both for detection and rendering.
_JINJA_BLOCK = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)

# A {{...}} content that's a single var, optionally followed by Jinja2 filter
# chain like ``name | upper`` or ``items | join(', ')``.
_JINJA_INNER_VAR = re.compile(
    rf"^\s*({_IDENT})(?:\s*\|\s*{_IDENT}(?:\([^)]*\))?)*\s*$"
)

# A {var} placeholder (Python .format style).
_FORMAT_VAR = re.compile(rf"\{{({_IDENT})\}}")

# ${var} shell / JS template style.
_SHELL_VAR = re.compile(rf"\$\{{({_IDENT})\}}")

# %(var)s Python old-style.
_PERCENT_VAR = re.compile(rf"%\(({_IDENT})\)s")

# Fenced code block — anything inside ```...``` is treated as literal.
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)

# Explicit "do not interpolate" markers.
_JINJA_RAW = re.compile(r"\{%\s*raw\s*%\}.*?\{%\s*endraw\s*%\}", re.DOTALL)

# Common prose tokens that look like {NAME} but aren't real variables.
_NOISE_IDENTS = frozenset({"TODO", "FIXME", "XXX", "NA", "None", "null", "NULL"})


# ─── public types ────────────────────────────────────────────────────────


class Syntax(str, Enum):
    JINJA2 = "jinja2"
    FORMAT = "format"
    SHELL = "shell"
    PERCENT = "percent"
    NONE = "none"


class MissingVariableError(KeyError):
    """Raised by ``render(..., strict=True)`` when a variable has no value."""


@dataclass(frozen=True)
class Zone:
    """A span of the template that should NOT be substituted."""

    start: int
    end: int
    reason: str  # "code_fence" | "jinja_raw" | "sample_output"


@dataclass(frozen=True)
class DetectionResult:
    syntax: Syntax
    variables: list[str]
    skip_zones: list[Zone] = field(default_factory=list)
    confidence: float = 1.0
    scores: dict[Syntax, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ─── core helpers ────────────────────────────────────────────────────────


def _span_inside_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    for s, e in spans:
        if start >= s and end <= e:
            return True
    return False


def _find_protected_spans(template: str) -> list[tuple[int, int, str]]:
    """Spans excluded from all variable detection (code blocks + raw regions)."""
    out: list[tuple[int, int, str]] = []
    for m in _CODE_FENCE.finditer(template):
        out.append((m.start(), m.end(), "code_fence"))
    for m in _JINJA_RAW.finditer(template):
        out.append((m.start(), m.end(), "jinja_raw"))
    return out


def _is_likely_variable_content(content: str) -> bool:
    """Classify a ``{{...}}`` block's content as variable vs sample-output.

    Variable: single identifier, optionally with Jinja2 filter chain.
    Sample output: JSON-ish, multi-line, quoted strings, etc.
    """
    inner = content.strip()
    if not inner:
        return False
    if "\n" in inner:
        # multi-line content is overwhelmingly sample output, not a var expression
        return False
    if len(inner) > 80:
        # giant single-line "expressions" are usually inline JSON
        return False
    return _JINJA_INNER_VAR.fullmatch(inner) is not None


def _scan_jinja_blocks(
    template: str, protected: list[tuple[int, int]]
) -> tuple[list[tuple[int, int, str]], list[tuple[int, int]]]:
    """Return (variable_matches, sample_output_spans).

    Each variable match: ``(start, end, identifier)``.
    Each sample-output span: ``(start, end)`` for a ``{{...}}`` block whose
    content isn't a valid variable expression.
    """
    var_matches: list[tuple[int, int, str]] = []
    sample_spans: list[tuple[int, int]] = []
    for m in _JINJA_BLOCK.finditer(template):
        if _span_inside_any(m.start(), m.end(), protected):
            continue
        content = m.group(1)
        if _is_likely_variable_content(content):
            inner_m = _JINJA_INNER_VAR.fullmatch(content.strip())
            assert inner_m is not None  # guaranteed by _is_likely_variable_content
            var_matches.append((m.start(), m.end(), inner_m.group(1)))
        else:
            sample_spans.append((m.start(), m.end()))
    return var_matches, sample_spans


def _scan_format_vars(
    template: str,
    protected: list[tuple[int, int]],
    jinja_spans: list[tuple[int, int]],
) -> list[tuple[int, int, str]]:
    """Find ``{var}`` matches outside protected zones and outside ``{{...}}`` blocks.

    ``{var}`` inside a ``{{...}}`` pair is part of Python's ``.format()``
    escape semantics — when in FORMAT mode it's still substituted (Python's
    actual behavior), but the OPENING ``{`` of the escape doesn't itself
    start a placeholder. We exclude the literal ``{{`` and ``}}`` characters
    by skipping any match whose span overlaps a ``{{`` or ``}}`` token.
    """
    matches: list[tuple[int, int, str]] = []
    exclude = protected[:]
    for m in _FORMAT_VAR.finditer(template):
        # Skip if this match IS a {{ or }} pair (would mis-fire on something
        # like "{{x}}" where the regex could greedily match "{x}" if the
        # outer braces are also single — but our regex is `\{ident\}` so it
        # only matches single braces, not doubles. Still belt-and-suspenders:)
        s, e = m.start(), m.end()
        # If preceded by `{` or followed by `}`, it's part of a {{...}} pair.
        if s > 0 and template[s - 1] == "{":
            continue
        if e < len(template) and template[e] == "}":
            continue
        if _span_inside_any(s, e, exclude):
            continue
        ident = m.group(1)
        if ident in _NOISE_IDENTS:
            continue
        matches.append((s, e, ident))
    return matches


def _scan_simple_pattern(
    pattern: re.Pattern[str],
    template: str,
    protected: list[tuple[int, int]],
) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for m in pattern.finditer(template):
        if _span_inside_any(m.start(), m.end(), protected):
            continue
        ident = m.group(1)
        if ident in _NOISE_IDENTS:
            continue
        out.append((m.start(), m.end(), ident))
    return out


def _dedup_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _score_matches(matches: list[tuple[int, int, str]]) -> int:
    """Score: count + repeat-bonus (same ident used 2+ times = strong signal)."""
    if not matches:
        return 0
    idents = [m[2] for m in matches]
    return len(matches) + (len(matches) - len(set(idents)))


# ─── detection ───────────────────────────────────────────────────────────


def detect(template: str) -> DetectionResult:
    """Detect dominant syntax and extract variables + sample-output zones.

    See module docstring for the disambiguation strategy.
    """
    if not template:
        return DetectionResult(Syntax.NONE, [], [], 1.0, {}, ["empty template"])

    protected_full = _find_protected_spans(template)
    protected = [(s, e) for s, e, _ in protected_full]

    jinja_vars, jinja_samples = _scan_jinja_blocks(template, protected)
    all_jinja_spans = [(s, e) for s, e, _ in jinja_vars] + jinja_samples

    fmt_vars = _scan_format_vars(template, protected, all_jinja_spans)
    shell_vars = _scan_simple_pattern(_SHELL_VAR, template, protected)
    pct_vars = _scan_simple_pattern(_PERCENT_VAR, template, protected)

    scores: dict[Syntax, int] = {
        Syntax.JINJA2: _score_matches(jinja_vars),
        Syntax.FORMAT: _score_matches(fmt_vars),
        Syntax.SHELL: _score_matches(shell_vars),
        Syntax.PERCENT: _score_matches(pct_vars),
    }

    # Pick winner with deterministic tiebreaker.
    # Priority order when scores tie: shell > percent > jinja2 > format.
    # Shell and percent are syntactically unambiguous; jinja2 is more
    # explicit than format; format last because single braces are easiest
    # to misuse in prose.
    priority = {Syntax.SHELL: 0, Syntax.PERCENT: 1, Syntax.JINJA2: 2, Syntax.FORMAT: 3}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], priority[kv[0]]))
    winner_syntax, winner_score = ranked[0]

    # If no syntax found anything, the template has no real variables.
    # Still expose any {{...}} sample-output blocks as zones so rendering
    # can de-escape them.
    if winner_score == 0:
        zones = [Zone(s, e, r) for s, e, r in protected_full]
        for s, e in jinja_samples:
            zones.append(Zone(s, e, "sample_output"))
        return DetectionResult(
            Syntax.NONE, [], zones, 1.0, scores, ["no variables detected"]
        )

    matches_by_syntax = {
        Syntax.JINJA2: jinja_vars,
        Syntax.FORMAT: fmt_vars,
        Syntax.SHELL: shell_vars,
        Syntax.PERCENT: pct_vars,
    }
    chosen_matches = matches_by_syntax[winner_syntax]
    variables = _dedup_preserving_order([m[2] for m in chosen_matches])

    # Build skip zones. Always: protected (code fences, raw blocks).
    # If winner == JINJA2 → only non-var {{...}} blocks become sample zones.
    # If winner != JINJA2 → ALL {{...}} blocks become sample zones (they're
    #   not part of the winning syntax, so render as literal single-braced).
    zones: list[Zone] = [Zone(s, e, r) for s, e, r in protected_full]
    if winner_syntax == Syntax.JINJA2:
        for s, e in jinja_samples:
            zones.append(Zone(s, e, "sample_output"))
    else:
        for s, e in all_jinja_spans:
            zones.append(Zone(s, e, "sample_output"))

    # Confidence: how dominant is the winner relative to runner-up?
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    if runner_up == 0:
        confidence = 1.0
    else:
        confidence = max(0.5, 1.0 - runner_up / winner_score)

    notes = []
    if any(s for k, s in scores.items() if k != winner_syntax):
        other = {k.value: s for k, s in scores.items() if s and k != winner_syntax}
        notes.append(f"other syntaxes also matched (treated as literal): {other}")

    return DetectionResult(winner_syntax, variables, zones, confidence, scores, notes)


# ─── legacy public API ───────────────────────────────────────────────────


def extract_variables(template: str) -> list[str]:
    """Return variable names in first-appearance order, deduplicated.

    Auto-detects syntax — supports ``{{var}}``, ``{var}``, ``${var}``,
    ``%(var)s``.
    """
    return detect(template).variables


def has_variables(template: str) -> bool:
    """True iff the template references at least one variable."""
    return len(detect(template).variables) > 0


def validate(
    template: str, variables: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Return ``(missing, extra)`` lists, sorted.

    ``missing``: vars referenced by the template but not present in ``variables``.
    ``extra``: keys in ``variables`` that the template doesn't reference.
    """
    needed = set(detect(template).variables)
    provided = set(variables.keys())
    return sorted(needed - provided), sorted(provided - needed)


# ─── rendering ───────────────────────────────────────────────────────────


def render(
    template: str, variables: dict[str, Any], strict: bool = False
) -> str:
    """Substitute placeholders with values; auto-detects template syntax.

    Missing variables render as empty string unless ``strict=True``, in which
    case ``MissingVariableError`` is raised. Sample-output ``{{...}}`` blocks
    are unwrapped to single braces, matching Python ``.format()`` semantics
    for ``{{`` → ``{`` and ``}}`` → ``}``.
    """
    if not template:
        return template

    detection = detect(template)
    syntax = detection.syntax
    if syntax == Syntax.JINJA2:
        return _render_jinja(template, variables, strict, detection)
    if syntax == Syntax.SHELL:
        return _render_with_pattern(template, variables, strict, _SHELL_VAR)
    if syntax == Syntax.PERCENT:
        return _render_with_pattern(template, variables, strict, _PERCENT_VAR)
    # FORMAT or NONE: both use Python .format() escape semantics for {{ }}.
    # FORMAT additionally substitutes {var}; NONE has no vars to substitute.
    return _render_format(template, variables, strict)


def _value_for(name: str, variables: dict[str, Any], strict: bool) -> str:
    if name in variables:
        val = variables[name]
        return "" if val is None else str(val)
    if strict:
        raise MissingVariableError(name)
    return ""


def _render_jinja(
    template: str,
    variables: dict[str, Any],
    strict: bool,
    detection: DetectionResult,
) -> str:
    """Render Jinja2 mode: ``{{var}}`` → value, ``{{ non-var }}`` → ``{ non-var }``."""
    sample_spans = {
        (z.start, z.end) for z in detection.skip_zones if z.reason == "sample_output"
    }
    out: list[str] = []
    cursor = 0
    for m in _JINJA_BLOCK.finditer(template):
        s, e = m.start(), m.end()
        out.append(template[cursor:s])
        if (s, e) in sample_spans:
            # de-escape: render with single braces
            out.append("{" + m.group(1) + "}")
        else:
            content = m.group(1).strip()
            inner_m = _JINJA_INNER_VAR.fullmatch(content)
            if inner_m is None:
                # belt-and-suspenders: treat as literal
                out.append("{" + m.group(1) + "}")
            else:
                ident = inner_m.group(1)
                out.append(_value_for(ident, variables, strict))
        cursor = e
    out.append(template[cursor:])
    return "".join(out)


def _render_format(
    template: str, variables: dict[str, Any], strict: bool
) -> str:
    """Render Python ``.format()`` mode using string.Formatter semantics.

    ``{{`` → ``{``, ``}}`` → ``}``, ``{var}`` → value. We use Formatter so
    nested escape semantics behave exactly like Python's str.format().
    """

    class _Lenient(string.Formatter):
        def get_value(  # type: ignore[override]
            self, key: Any, args: Any, kwargs: dict[str, Any]
        ) -> Any:
            if isinstance(key, str):
                if key in kwargs:
                    val = kwargs[key]
                    return "" if val is None else val
                if strict:
                    raise MissingVariableError(key)
                return ""
            # positional placeholders aren't expected; fall back to empty
            return ""

        def format_field(self, value: Any, format_spec: str) -> str:  # type: ignore[override]
            try:
                return super().format_field(value, format_spec)
            except (ValueError, TypeError):
                return str(value)

    formatter = _Lenient()
    try:
        return formatter.vformat(template, (), dict(variables))
    except (IndexError, KeyError, ValueError):
        # Last-resort fallback: manual {{ }} de-escape + {var} substitution.
        # Keeps us functional on templates that contain things Python's
        # Formatter chokes on (e.g. unmatched `}` in a JSON example without
        # the corresponding escape).
        return _render_format_manual(template, variables, strict)


def _render_format_manual(
    template: str, variables: dict[str, Any], strict: bool
) -> str:
    """Hand-rolled FORMAT renderer for templates that break string.Formatter.

    Tokenizes left-to-right: ``{{`` → ``{``, ``}}`` → ``}``, ``{ident}`` →
    value lookup, ``{`` or ``}`` alone → emitted as-is (lenient — real
    .format() would raise).
    """
    out: list[str] = []
    i = 0
    n = len(template)
    while i < n:
        ch = template[i]
        if ch == "{" and i + 1 < n and template[i + 1] == "{":
            out.append("{")
            i += 2
            continue
        if ch == "}" and i + 1 < n and template[i + 1] == "}":
            out.append("}")
            i += 2
            continue
        if ch == "{":
            # Look for }
            end = template.find("}", i + 1)
            if end == -1:
                out.append(ch)
                i += 1
                continue
            inner = template[i + 1 : end]
            ident_m = re.fullmatch(_IDENT, inner)
            if ident_m is None:
                # Not a valid placeholder; emit char and continue
                out.append(ch)
                i += 1
                continue
            out.append(_value_for(inner, variables, strict))
            i = end + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _render_with_pattern(
    template: str,
    variables: dict[str, Any],
    strict: bool,
    pattern: re.Pattern[str],
) -> str:
    """Generic single-pattern renderer used for SHELL and PERCENT modes."""

    def sub(m: re.Match[str]) -> str:
        return _value_for(m.group(1), variables, strict)

    return pattern.sub(sub, template)
