"""Writer agent — produces the v0 prompt.

Two modes:
  * cold: (intent, requirements, objectives) → designed prompt with {{var}} placeholders.
  * warm: (existing_prompt, known_issues) → existing prompt is preserved verbatim;
    the agent only extracts variable declarations and notes observations.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.core import providers, template
from app.core.logging import log
from app.schemas.common import PromptVariable


class WriterOutput(BaseModel):
    """Strict schema for Writer responses."""

    prompt: str = Field(
        description=(
            "The prompt template. MUST contain at least one {{variable_name}} placeholder "
            "for inputs that vary across test cases. Variable names must be valid identifiers "
            "(letters/digits/underscores, not starting with a digit)."
        )
    )
    variables: list[PromptVariable] = Field(
        default_factory=list,
        description="Every {{var}} placeholder in the prompt must be declared here.",
    )
    rationale: str = Field(description="One paragraph: why this prompt should work.")
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions the prompt makes about the inputs or task. Short bullets.",
    )


@dataclass
class WriterResult:
    output: WriterOutput
    cost_usd: float
    cache_hit: bool


_OBJECTIVE_GUIDANCE = {
    "accuracy": "Be specific and precise. Define what 'correct' means for the task.",
    "cost": "Keep the prompt itself compact. Encourage concise model responses.",
    "latency": "Minimize prompt length. Avoid asking for long-form reasoning chains.",
    "robustness": (
        "Include explicit guardrails: stay on topic, refuse off-task inputs, "
        "don't fabricate, follow instructions strictly."
    ),
    "format_adherence": "State the exact output format. If JSON, provide the schema in-prompt.",
    "brevity": "Tell the model to keep responses short and direct.",
    "tone": "Specify the target voice/register clearly with one example.",
}


def _objective_block(objectives: list[str]) -> str:
    lines = []
    for obj in objectives:
        guidance = _OBJECTIVE_GUIDANCE.get(obj, "")
        lines.append(f"- {obj}: {guidance}")
    return "\n".join(lines)


_COLD_SYSTEM = """You are a senior prompt engineer. Your job is to design a v0 prompt for the
user's task. This prompt will be the seed for an iterative optimization loop that will rewrite
it surgically based on evaluation results.

Rules:
1. The prompt MUST contain at least one {{variable_name}} placeholder for inputs that vary
   across test cases. Variable names must be valid identifiers (letters, digits, underscores;
   not starting with a digit).
2. Declare every variable you use in the `variables` field, with a clear description and a
   realistic example value. Do not declare variables you don't reference.
3. Optimize the v0 draft for the specified objectives — they shape your style and content.
4. Be specific. Avoid generic phrases. A precise v0 saves the loop several iterations.
5. Output strict JSON conforming to the schema."""


_WARM_SYSTEM = """You are reviewing an existing prompt as the seed (v0) for an iterative
optimization loop. You DO NOT rewrite the prompt — the loop will do that.

Your tasks:
1. Return the user's prompt VERBATIM as `prompt`. Do not edit, reword, or normalize it.
2. Extract every variable placeholder it contains and declare each with a description +
   realistic example value. Supported placeholder syntaxes:
     - {{variable_name}}    (Jinja2/Mustache/Handlebars)
     - {variable_name}      (Python .format)
     - ${variable_name}     (shell / JS template literal)
     - %(variable_name)s    (Python old-style)
   IMPORTANT: a {{...}} block whose contents look like JSON (multi-line, quoted keys,
   colons, commas) is a SAMPLE OUTPUT EXAMPLE, not a variable — do NOT treat it as one.
   Likewise, treat the prompt's described output schema as literal, not as variables.
   If a placeholder name has surrounding whitespace, extract the identifier inside.
3. If the prompt contains NO placeholders, you MUST still return the prompt verbatim AND
   declare an empty `variables` list — flag this in `assumptions` so the loop can wrap it.
4. In `rationale`, briefly describe what the prompt is trying to do (your read of it).
5. In `assumptions`, list anything notable (missing guardrails, unclear sections, the
   user-reported issues that the optimizer should target first).
6. Output strict JSON conforming to the schema."""


def _objectives_text(objectives: list[str]) -> str:
    return _objective_block(objectives) or "- accuracy: be correct on the task."


async def write_cold(
    *,
    intent: str,
    requirements: str | None,
    objectives: list[str],
    model: str,
) -> WriterResult:
    user_msg = (
        f"User intent:\n{intent}\n\n"
        f"Requirements:\n{requirements or '(none provided)'}\n\n"
        f"Optimization objectives (drives style):\n{_objectives_text(objectives)}\n\n"
        "Design the v0 prompt now."
    )
    result = await providers.complete(
        model=model,
        messages=[
            {"role": "system", "content": _COLD_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format=WriterOutput,
        temperature=0.6,
    )
    output = _coerce_or_repair(result.parsed, result.content, intent=intent)
    return WriterResult(output=output, cost_usd=result.cost_usd, cache_hit=result.cache_hit)


async def write_warm(
    *,
    existing_prompt: str,
    known_issues: str | None,
    objectives: list[str],
    model: str,
) -> WriterResult:
    user_msg = (
        f'Existing prompt (return VERBATIM in `prompt`):\n"""\n{existing_prompt}\n"""\n\n'
        f"User-reported issues to seed the optimizer with:\n"
        f"{known_issues or '(none provided)'}\n\n"
        f"Optimization objectives:\n{_objectives_text(objectives)}"
    )
    result = await providers.complete(
        model=model,
        messages=[
            {"role": "system", "content": _WARM_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format=WriterOutput,
        temperature=0.2,
    )
    output = _coerce_or_repair(
        result.parsed,
        result.content,
        intent=None,
        warm_fallback_prompt=existing_prompt,
    )
    # In warm mode we ENFORCE verbatim preservation regardless of what the LLM returned.
    if output.prompt.strip() != existing_prompt.strip():
        log.warning(
            "writer.warm.prompt_modified_by_llm",
            note="LLM altered the prompt; restoring user-supplied verbatim text",
        )
        output = output.model_copy(update={"prompt": existing_prompt})
    return WriterResult(output=output, cost_usd=result.cost_usd, cache_hit=result.cache_hit)


def _coerce_or_repair(
    parsed: BaseModel | None,
    raw: str,
    intent: str | None,
    warm_fallback_prompt: str | None = None,
) -> WriterOutput:
    """If the model failed to produce valid structured output, salvage what we can."""
    if isinstance(parsed, WriterOutput):
        # Ensure declared variables match what's actually in the prompt
        declared = {v.name for v in parsed.variables}
        actual = set(template.extract_variables(parsed.prompt))
        # Drop spurious declarations not present in the prompt
        cleaned_vars = [v for v in parsed.variables if v.name in actual]
        # Synthesize declarations for any actually-referenced variables that weren't declared
        for missing in sorted(actual - declared):
            cleaned_vars.append(
                PromptVariable(
                    name=missing,
                    description=f"(auto-declared) value for {{{{{missing}}}}}",
                    example_value="",
                )
            )
        return parsed.model_copy(update={"variables": cleaned_vars})

    # Parse failed entirely — build a degraded but valid WriterOutput
    prompt = (
        warm_fallback_prompt
        if warm_fallback_prompt is not None
        else f"You are asked to: {intent or '(unspecified)'}\n\nInput: {{{{input}}}}\n\nResponse:"
    )
    fallback_vars = template.extract_variables(prompt)
    return WriterOutput(
        prompt=prompt,
        variables=[
            PromptVariable(
                name=v,
                description=f"(degraded) value for {{{{{v}}}}}",
                example_value="",
            )
            for v in fallback_vars
        ],
        rationale="(degraded) Writer failed structured output; using fallback.",
        assumptions=["Writer LLM returned malformed JSON; this prompt is a placeholder."],
    )
