from app.agents.scorers import (
    contains_all,
    contains_none,
    exact_match,
    json_valid,
    length_within,
    regex_match,
)


class TestExactMatch:
    def test_match_case_insensitive(self) -> None:
        assert exact_match("Hello", "hello") == 1.0

    def test_mismatch(self) -> None:
        assert exact_match("hello", "world") == 0.0

    def test_strips_whitespace(self) -> None:
        assert exact_match("  hello\n", "hello") == 1.0

    def test_case_sensitive(self) -> None:
        assert exact_match("Hi", "hi", case_sensitive=True) == 0.0


class TestRegexMatch:
    def test_match(self) -> None:
        assert regex_match("phone: 555-1234", r"\d{3}-\d{4}") == 1.0

    def test_no_match(self) -> None:
        assert regex_match("hi", r"\d") == 0.0


class TestJsonValid:
    def test_valid(self) -> None:
        assert json_valid('{"a": 1}') == 1.0

    def test_invalid(self) -> None:
        assert json_valid("not json") == 0.0

    def test_valid_array(self) -> None:
        assert json_valid("[1, 2, 3]") == 1.0


class TestLengthWithin:
    def test_in_range(self) -> None:
        assert length_within("hello", min_chars=3, max_chars=10) == 1.0

    def test_too_short(self) -> None:
        assert length_within("hi", min_chars=5) == 0.0

    def test_too_long(self) -> None:
        assert length_within("x" * 100, max_chars=10) == 0.0

    def test_no_upper_bound(self) -> None:
        assert length_within("x" * 1_000_000, min_chars=0, max_chars=None) == 1.0


class TestContains:
    def test_all_present(self) -> None:
        assert contains_all("hello world there", ["hello", "world"]) == 1.0

    def test_all_missing_one(self) -> None:
        assert contains_all("hello", ["hello", "world"]) == 0.0

    def test_none_present(self) -> None:
        assert contains_none("hello", ["password", "secret"]) == 1.0

    def test_none_contains_one(self) -> None:
        assert contains_none("my password is x", ["password"]) == 0.0
