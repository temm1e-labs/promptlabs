import pytest

from app.core.template import (
    DetectionResult,
    MissingVariableError,
    Syntax,
    detect,
    extract_variables,
    has_variables,
    render,
    validate,
)


# ─── legacy Jinja2-style tests (backwards-compat) ────────────────────────


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
        assert extract_variables("{{42name}}") == []
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


# ─── new syntax detection ────────────────────────────────────────────────


class TestDetectSyntax:
    def test_pure_jinja2(self) -> None:
        d = detect("Hello {{name}}, welcome to {{place}}.")
        assert d.syntax == Syntax.JINJA2
        assert d.variables == ["name", "place"]

    def test_pure_python_format(self) -> None:
        d = detect("Hello {name}, welcome to {place}.")
        assert d.syntax == Syntax.FORMAT
        assert d.variables == ["name", "place"]

    def test_shell_style(self) -> None:
        d = detect("export PATH=${path} for ${user}")
        assert d.syntax == Syntax.SHELL
        assert d.variables == ["path", "user"]

    def test_python_percent_style(self) -> None:
        d = detect("Hello %(name)s, you are %(age)s years old")
        assert d.syntax == Syntax.PERCENT
        assert d.variables == ["name", "age"]

    def test_empty_template(self) -> None:
        d = detect("")
        assert d.syntax == Syntax.NONE
        assert d.variables == []

    def test_no_variables(self) -> None:
        d = detect("This is just a plain prompt with no placeholders.")
        assert d.syntax == Syntax.NONE
        assert d.variables == []


class TestSampleOutputDisambiguation:
    """``{{...}}`` blocks that are JSON-ish are NOT variables."""

    def test_jinja_with_json_sample(self) -> None:
        """Real Jinja2 vars coexist with JSON sample output."""
        template = (
            "Hello {{name}}. Output format:\n"
            "{{\n"
            '  "key": "value",\n'
            '  "score": 0.95\n'
            "}}\n"
        )
        d = detect(template)
        assert d.syntax == Syntax.JINJA2
        assert d.variables == ["name"]
        # the JSON block should be marked as a sample-output zone
        assert any(z.reason == "sample_output" for z in d.skip_zones)

    def test_python_format_with_double_brace_json(self) -> None:
        """User's reported case: {var} placeholders with {{ }} JSON output example."""
        template = (
            "Group: {group}\n"
            "Description: {group_description}\n\n"
            "## OUTPUT FORMAT:\n"
            "{{\n"
            '  "filled_subject": "<value>",\n'
            '  "filled_fields": {{}}\n'
            "}}\n"
        )
        d = detect(template)
        assert d.syntax == Syntax.FORMAT
        assert "group" in d.variables
        assert "group_description" in d.variables
        # Should NOT pick up the JSON example as variables
        assert "filled_subject" not in d.variables
        assert "filled_fields" not in d.variables

    def test_jinja_filter_chain(self) -> None:
        d = detect("Hello {{ name | upper }} and {{ age | default('?') }}")
        assert d.syntax == Syntax.JINJA2
        assert d.variables == ["name", "age"]

    def test_multiline_double_brace_is_sample(self) -> None:
        """Multi-line content in {{...}} is always sample output, never variable."""
        template = "{{\n  some_content\n  more_content\n}}"
        d = detect(template)
        # No real variables, just a sample zone
        assert d.variables == []


class TestRenderingNewSyntaxes:
    def test_format_mode_substitutes(self) -> None:
        assert render("Hello {name}", {"name": "World"}) == "Hello World"

    def test_format_mode_with_json_sample(self) -> None:
        """The user's exact case: render {var} and unwrap {{ }} JSON sample."""
        template = (
            "Group: {group}\n"
            "Output:\n"
            "{{\n"
            '  "field": "value"\n'
            "}}\n"
        )
        rendered = render(template, {"group": "X"})
        assert "Group: X" in rendered
        # The {{ }} should be unwrapped to { }
        assert '{\n  "field": "value"\n}' in rendered
        assert "{{" not in rendered
        assert "}}" not in rendered

    def test_shell_mode(self) -> None:
        assert render("hi ${name}", {"name": "world"}) == "hi world"

    def test_percent_mode(self) -> None:
        assert render("hi %(name)s", {"name": "world"}) == "hi world"

    def test_format_mode_missing_var_strict(self) -> None:
        with pytest.raises(MissingVariableError):
            render("hi {name}", {}, strict=True)

    def test_format_mode_missing_var_lenient(self) -> None:
        assert render("hi {name}!", {}) == "hi !"

    def test_format_mode_repeated_var(self) -> None:
        assert render("{a} + {a}", {"a": "1"}) == "1 + 1"

    def test_jinja_with_json_sample_rendered(self) -> None:
        """Jinja2 + JSON sample: var substituted, sample unwrapped."""
        template = 'Hello {{name}}. Format: {{ "k": "v" }}'
        rendered = render(template, {"name": "World"})
        assert "Hello World" in rendered
        assert '{ "k": "v" }' in rendered

    def test_no_vars_template_unchanged(self) -> None:
        assert render("plain text only", {}) == "plain text only"


class TestProtectedZones:
    def test_code_fence_blocks_detection(self) -> None:
        template = "Variable: {{name}}.\nExample:\n```\n{{this_is_in_code}}\n```\n"
        d = detect(template)
        assert d.variables == ["name"]
        assert "this_is_in_code" not in d.variables

    def test_jinja_raw_blocks_detection(self) -> None:
        template = "Hello {{name}}. {% raw %}{{literal}}{% endraw %}"
        d = detect(template)
        assert d.variables == ["name"]
        assert "literal" not in d.variables

    def test_noise_words_excluded(self) -> None:
        """Common prose tokens like {TODO} should not be detected as variables."""
        d = detect("This is {TODO} and {FIXME} placeholder text with {actual_var}.")
        assert "TODO" not in d.variables
        assert "FIXME" not in d.variables
        assert "actual_var" in d.variables


class TestUserReportedBug:
    """Regression test for the exact prompt that caused 'Declared variables: []'."""

    def test_users_full_template(self) -> None:
        template = """Bạn là trợ lý trích xuất thông tin cuộc gọi cho hệ thống CRM VinFast.

## NHIỆM VỤ
Dựa trên transcript cuộc gọi, thực hiện 2 việc:
1. Điền thông tin vào template subject
2. Trích xuất các trường thông tin bắt buộc theo format JSON

## NHÓM ĐÃ PHÂN LOẠI
- TEMPLATE_GROUP: {group}
- MÔ TẢ: {group_description}

## TEMPLATE SUBJECT:
{subject_template}

## CÁC TRƯỜNG THÔNG TIN CẦN TRÍCH XUẤT:
{json_fields}

## OUTPUT FORMAT (JSON duy nhất, không markdown):
{{
  "filled_subject": "<subject đã điền>",
  "filled_fields": {{
    "<english_key_1>": "<giá trị>",
    "short_summary": "<tóm tắt>",
    "sentimental_analysis": "NEGATIVE hoặc NOT_NEGATIVE",
    "emergency": "true hoặc false"
  }}
}}
"""
        d = detect(template)
        assert d.syntax == Syntax.FORMAT
        # All four real variables found
        assert set(d.variables) == {
            "group",
            "group_description",
            "subject_template",
            "json_fields",
        }
        # No false positives from the JSON example
        assert "filled_subject" not in d.variables
        assert "filled_fields" not in d.variables
        assert "english_key_1" not in d.variables

    def test_users_template_renders(self) -> None:
        """The full template renders without losing the JSON output example."""
        template = (
            "Group: {group}\n"
            "Output format:\n"
            "{{\n"
            '  "filled_subject": "<value>",\n'
            '  "filled_fields": {{\n'
            '    "key": "value"\n'
            "  }}\n"
            "}}\n"
        )
        rendered = render(template, {"group": "TestGroup"})
        assert "Group: TestGroup" in rendered
        # JSON braces should be single in output
        assert '"filled_subject"' in rendered
        # No leftover doubles
        assert "{{" not in rendered
        assert "}}" not in rendered


class TestDetectionConfidence:
    """Confidence score helps the UI surface ambiguous cases."""

    def test_pure_syntax_has_high_confidence(self) -> None:
        d = detect("{{name}} {{age}}")
        assert d.confidence == 1.0

    def test_ambiguous_template_has_lower_confidence(self) -> None:
        # Both {{x}} (jinja) and {y} (format) — picks jinja but confidence < 1
        d = detect("{{name}} and also {something}")
        assert d.syntax == Syntax.JINJA2
        assert d.confidence < 1.0
