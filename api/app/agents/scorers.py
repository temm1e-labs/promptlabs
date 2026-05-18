"""Deterministic scorers — exact_match, regex, json_valid, length_within.

These are stateless utility functions. The orchestrator can plug them into a rubric
when a criterion has a deterministic ground truth; they return 0.0 or 1.0 (or a
gradient where appropriate, e.g. length_within).
"""

from __future__ import annotations

import json
import re


def exact_match(actual: str, expected: str, case_sensitive: bool = False) -> float:
    """1.0 iff actual.strip() == expected.strip()."""
    if case_sensitive:
        return float(actual.strip() == expected.strip())
    return float(actual.strip().lower() == expected.strip().lower())


def regex_match(actual: str, pattern: str, flags: int = re.IGNORECASE) -> float:
    """1.0 if regex matches anywhere in `actual`."""
    return float(re.search(pattern, actual, flags) is not None)


def json_valid(actual: str) -> float:
    """1.0 if `actual` parses as JSON."""
    try:
        json.loads(actual)
        return 1.0
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0


def length_within(actual: str, min_chars: int = 0, max_chars: int | None = None) -> float:
    """1.0 if len(actual) is within [min_chars, max_chars]; 0.0 otherwise.

    Pass ``max_chars=None`` for an open upper bound.
    """
    n = len(actual)
    if n < min_chars:
        return 0.0
    if max_chars is not None and n > max_chars:
        return 0.0
    return 1.0


def contains_all(actual: str, substrings: list[str], case_sensitive: bool = False) -> float:
    """1.0 iff every substring is found in actual."""
    haystack = actual if case_sensitive else actual.lower()
    needles = substrings if case_sensitive else [s.lower() for s in substrings]
    return float(all(n in haystack for n in needles))


def contains_none(actual: str, substrings: list[str], case_sensitive: bool = False) -> float:
    """1.0 iff none of the substrings are found in actual (useful for safety: no PII, etc.)."""
    haystack = actual if case_sensitive else actual.lower()
    needles = substrings if case_sensitive else [s.lower() for s in substrings]
    return float(not any(n in haystack for n in needles))
