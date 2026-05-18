import pytest

from app.core.template import (
    MissingVariableError,
    extract_variables,
    has_variables,
    render,
    validate,
)


class TestExtract:
    def test_simple(self) -> None:
        assert extract_variables("hello {{name}}") == ["name"]

    def test_whitespace_tolerant(self) -> None:
        assert extract_variables("{{ a }} {{b}} {{   c   }}") == ["a", "b", "c"]

    def test_dedupe_preserves_first_appearance(self) -> None:
        assert extract_variables("{{x}} {{y}} {{x}}") == ["x", "y"]

    def test_none(self) -> None:
        assert extract_variables("no vars here") == []

    def test_underscore_and_digits(self) -> None:
        assert extract_variables("{{user_id_42}}") == ["user_id_42"]

    def test_invalid_name_ignored(self) -> None:
        # leading digit not a valid identifier
        assert extract_variables("{{42name}}") == []
        # spaces inside name not allowed
        assert extract_variables("{{user name}}") == []


class TestRender:
    def test_substitution(self) -> None:
        assert render("hi {{name}}", {"name": "world"}) == "hi world"

    def test_multiple(self) -> None:
        assert render("{{a}} + {{b}}", {"a": "1", "b": "2"}) == "1 + 2"

    def test_missing_renders_empty_by_default(self) -> None:
        assert render("hi {{name}}!", {}) == "hi !"

    def test_missing_raises_in_strict_mode(self) -> None:
        with pytest.raises(MissingVariableError) as exc:
            render("hi {{name}}", {}, strict=True)
        assert "name" in str(exc.value)

    def test_extra_keys_ignored(self) -> None:
        assert render("hi", {"unused": "x"}) == "hi"

    def test_non_string_values_coerced(self) -> None:
        assert render("count = {{n}}", {"n": 42}) == "count = 42"

    def test_none_values_render_empty(self) -> None:
        assert render("[{{x}}]", {"x": None}) == "[]"


class TestValidate:
    def test_all_present(self) -> None:
        missing, extra = validate("hi {{name}}", {"name": "w"})
        assert missing == []
        assert extra == []

    def test_missing(self) -> None:
        missing, extra = validate("{{a}} {{b}}", {"a": "1"})
        assert missing == ["b"]
        assert extra == []

    def test_extra(self) -> None:
        missing, extra = validate("{{a}}", {"a": "1", "z": "2"})
        assert missing == []
        assert extra == ["z"]


class TestHasVariables:
    def test_true(self) -> None:
        assert has_variables("{{x}}") is True

    def test_false(self) -> None:
        assert has_variables("plain") is False
