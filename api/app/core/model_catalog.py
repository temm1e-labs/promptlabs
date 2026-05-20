"""Curated default model list. Users can also type any LiteLLM identifier."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    id: str  # the LiteLLM model identifier
    label: str  # human-readable
    provider: str
    family: str | None = None
    is_default: bool = False


DEFAULT_CATALOG: tuple[ModelEntry, ...] = (
    # Anthropic
    ModelEntry("anthropic/claude-opus-4-7", "Claude Opus 4.7", "Anthropic", "claude", True),
    ModelEntry("anthropic/claude-sonnet-4-6", "Claude Sonnet 4.6", "Anthropic", "claude"),
    ModelEntry("anthropic/claude-haiku-4-5", "Claude Haiku 4.5", "Anthropic", "claude"),
    # OpenAI
    ModelEntry("openai/gpt-5", "GPT-5", "OpenAI", "gpt", True),
    ModelEntry("openai/gpt-5-mini", "GPT-5 mini", "OpenAI", "gpt"),
    ModelEntry("openai/gpt-4.1", "GPT-4.1", "OpenAI", "gpt"),
    ModelEntry("openai/o5-mini", "o5 mini", "OpenAI", "o-series"),
    # Google
    ModelEntry(
        "gemini/gemini-3.5-flash",
        "Gemini 3.5 Flash",
        "Google",
        "gemini",
    ),
    ModelEntry(
        "gemini/gemini-3.1-pro-preview",
        "Gemini 3.1 Pro (preview)",
        "Google",
        "gemini",
    ),
    ModelEntry(
        "gemini/gemini-3-flash-preview",
        "Gemini 3 Flash (preview)",
        "Google",
        "gemini",
    ),
    ModelEntry("gemini/gemini-2.5-pro", "Gemini 2.5 Pro", "Google", "gemini", True),
    ModelEntry("gemini/gemini-2.5-flash", "Gemini 2.5 Flash", "Google", "gemini"),
    # Mistral
    ModelEntry("mistral/mistral-large-latest", "Mistral Large", "Mistral", "mistral"),
    ModelEntry("mistral/mistral-medium-latest", "Mistral Medium", "Mistral", "mistral"),
    # Groq (fast inference)
    ModelEntry("groq/llama-3.3-70b-versatile", "Llama 3.3 70B (Groq)", "Groq", "llama"),
    # DeepSeek
    ModelEntry("deepseek/deepseek-chat", "DeepSeek Chat", "DeepSeek", "deepseek"),
    ModelEntry("deepseek/deepseek-reasoner", "DeepSeek Reasoner", "DeepSeek", "deepseek"),
)


def grouped_by_provider() -> dict[str, list[ModelEntry]]:
    groups: dict[str, list[ModelEntry]] = {}
    for m in DEFAULT_CATALOG:
        groups.setdefault(m.provider, []).append(m)
    return groups
