"""Smaller live tests against Gemini — exercise Writer + EvalGen separately."""

from __future__ import annotations

import os

import pytest

from app.agents import evalgen, writer

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
LIVE = pytest.mark.skipif(not GEMINI_KEY, reason="GEMINI_API_KEY not set")
MODEL = "gemini/gemini-3-flash-preview"


@LIVE
@pytest.mark.asyncio
async def test_writer_cold_with_gemini() -> None:
    result = await writer.write_cold(
        intent="Classify a customer support email into billing / technical / account / other.",
        requirements="Reply with one word, lowercase.",
        objectives=["accuracy", "format_adherence"],
        model=MODEL,
    )
    print("\n--- Writer cold result ---")
    print("prompt:", repr(result.output.prompt[:400]))
    print("variables:", [v.model_dump() for v in result.output.variables])
    print("cost:", result.cost_usd)
    assert "{{" in result.output.prompt, "prompt should contain a {{var}} placeholder"
    assert len(result.output.variables) >= 1


@LIVE
@pytest.mark.asyncio
async def test_evalgen_with_gemini() -> None:
    """Feed EvalGen the Writer's output directly to see the var-name mismatch."""
    writer_result = await writer.write_cold(
        intent="Classify a customer support email into billing / technical / account / other.",
        requirements="Reply with one word, lowercase.",
        objectives=["accuracy", "format_adherence"],
        model=MODEL,
    )
    print("\n--- Writer prompt vars ---", [v.name for v in writer_result.output.variables])

    eg = await evalgen.generate(
        intent="Classify a customer support email into billing / technical / account / other.",
        prompt_v0=writer_result.output.prompt,
        variables=list(writer_result.output.variables),
        objectives=["accuracy", "format_adherence"],
        known_issues=None,
        eval_size=4,
        train_ratio=0.5,
        model=MODEL,
    )
    print("--- EvalGen result ---")
    print("rubric:", [c.name for c in eg.rubric])
    print("train items:", [(it.label, list(it.input_vars.keys())) for it in eg.train_items])
    print("holdout items:", [(it.label, list(it.input_vars.keys())) for it in eg.holdout_items])
    print("cost:", eg.cost_usd)
    assert len(eg.rubric) >= 1
    assert len(eg.train_items) + len(eg.holdout_items) >= 1, (
        "EvalGen should produce at least one usable item"
    )
