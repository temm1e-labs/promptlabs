from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.writer import WriterOutput, write_cold, write_warm


def _fake_response(content: str) -> MagicMock:
    msg = MagicMock(content=content)
    choice = MagicMock(message=msg)
    resp = MagicMock(choices=[choice])
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 200
    return resp


@pytest.mark.asyncio
async def test_write_cold_produces_template_with_declared_vars() -> None:
    payload = """
    {
        "prompt": "Classify the customer email below.\\n\\nEmail: {{email}}\\n\\nLabel:",
        "variables": [
            {"name": "email", "description": "The raw email text to classify.",
             "example_value": "Hi, I want to cancel my subscription."}
        ],
        "rationale": "Direct instruction + single placeholder for the email body.",
        "assumptions": ["Email is plain text, not HTML."]
    }
    """
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response(payload)),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.002),
    ):
        result = await write_cold(
            intent="classify customer emails into intent buckets",
            requirements=None,
            objectives=["accuracy", "format_adherence"],
            model="openai/gpt-5",
        )

    assert isinstance(result.output, WriterOutput)
    assert "{{email}}" in result.output.prompt
    assert len(result.output.variables) == 1
    assert result.output.variables[0].name == "email"
    assert result.cost_usd == 0.002


@pytest.mark.asyncio
async def test_write_cold_repairs_undeclared_variables() -> None:
    """If LLM uses {{x}} but forgets to declare it, the agent fills in a declaration."""
    payload = """
    {
        "prompt": "Process {{undeclared_var}} and reply.",
        "variables": [],
        "rationale": "x",
        "assumptions": []
    }
    """
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response(payload)),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await write_cold(
            intent="x",
            requirements=None,
            objectives=["accuracy"],
            model="openai/gpt-5",
        )

    names = [v.name for v in result.output.variables]
    assert names == ["undeclared_var"]


@pytest.mark.asyncio
async def test_write_cold_drops_spurious_declarations() -> None:
    """If LLM declares vars not used in prompt, they're filtered out."""
    payload = """
    {
        "prompt": "Reply to {{question}}.",
        "variables": [
            {"name": "question", "description": "the question", "example_value": "hi"},
            {"name": "ghost", "description": "unused", "example_value": "x"}
        ],
        "rationale": "",
        "assumptions": []
    }
    """
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response(payload)),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await write_cold(
            intent="x", requirements=None, objectives=[], model="openai/gpt-5"
        )

    names = {v.name for v in result.output.variables}
    assert names == {"question"}


@pytest.mark.asyncio
async def test_write_warm_preserves_existing_prompt_verbatim() -> None:
    """Warm mode must not let the LLM rewrite the user's prompt."""
    original = "Existing prompt with {{my_var}} placeholder."
    naughty_payload = """
    {
        "prompt": "DIFFERENT REWRITTEN PROMPT",
        "variables": [
            {"name": "my_var", "description": "x", "example_value": "y"}
        ],
        "rationale": "",
        "assumptions": []
    }
    """
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response(naughty_payload)),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.001),
    ):
        result = await write_warm(
            existing_prompt=original,
            known_issues="too verbose sometimes",
            objectives=["brevity"],
            model="openai/gpt-5",
        )

    assert result.output.prompt == original


@pytest.mark.asyncio
async def test_write_cold_handles_malformed_json_with_fallback() -> None:
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response("not json at all")),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await write_cold(
            intent="x", requirements=None, objectives=[], model="openai/gpt-5"
        )

    assert "{{input}}" in result.output.prompt
    assert "(degraded)" in result.output.rationale
