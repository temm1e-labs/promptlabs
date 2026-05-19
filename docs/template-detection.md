# Template variable detection

PromptLabs accepts prompt templates in several syntaxes and auto-detects which
one a given template uses. The detector lives in `api/app/core/template.py`
and is the only path through which the system reads variables from a template
(used by Writer, Optimizer, Runner, EvalGen, and the experiment loop).

## Supported syntaxes

| Syntax       | Example                | Where it comes from        |
|--------------|------------------------|----------------------------|
| Jinja2       | `{{user_name}}`        | Mustache / Handlebars / Jinja2 |
| Python `.format` | `{user_name}`      | `str.format()` style       |
| Shell        | `${user_name}`         | POSIX shell / JS template literal |
| Python `%`   | `%(user_name)s`        | Python old-style percent formatting |

Detection happens per-template. The detector picks the syntax with the most
clean matches; ties break in a fixed priority order (shell → percent → jinja2
→ format).

## Sample-output disambiguation

The hard case the detector is built to handle:

```text
## OUTPUT FORMAT:
{{
  "key": "value",
  "score": 0.95
}}
```

This is a JSON output example, not a Jinja2 variable. The detector classifies
each `{{...}}` block by its content:

- Single identifier (with optional Jinja2 filter chain like `name | upper`)
  → **variable**, substituted at render time.
- Multi-line, contains JSON-like punctuation, exceeds length cap, or content
  that doesn't parse as an identifier → **sample output**, rendered as a
  single-braced literal at render time.

This means the user's prompt:

```text
- TEMPLATE_GROUP: {group}
- DESCRIPTION: {group_description}

## OUTPUT FORMAT:
{{
  "filled_subject": "<value>",
  "filled_fields": {{...}}
}}
```

…is detected as FORMAT syntax with 2 variables (`group`, `group_description`),
the JSON block correctly marked as a sample-output zone. Rendered output
preserves single-braced JSON.

## Protected zones

These are excluded from detection entirely:

- Fenced code blocks ` ``` ... ``` `
- Jinja2 raw blocks `{% raw %} ... {% endraw %}`

If you have a real placeholder you want to highlight in docs without it being
detected, use a fenced code block.

## Confidence

`DetectionResult.confidence` is `1.0` when one syntax clearly wins and lower
when multiple syntaxes scored similarly. The current code does not surface
confidence to the UI; doing so is a near-term followup (see below).

---

# Known limitations

The detector ships with deliberate heuristics that catch the common case but
have known failure modes:

## Heuristic boundaries

1. **80-char single-line cap on Jinja2 expressions.** A long but legitimate
   filter chain like `{{ items | join(', ') | truncate(120, killwords=True) }}`
   gets classified as sample output instead of variable.

2. **"Has quotes or colons → not a variable" rule over-fires.** Valid Jinja2
   expressions like `{{ user[:5] }}` (Python slice) or
   `{{ user.role | default('admin') }}` get rejected.

3. **Code-fence masking is triple-backtick only.** Inline code like
   `` use `{name}` to reference the field `` in prose gets mis-detected as
   a variable.

4. **Noise-word list is hardcoded.** `{TODO}` and `{FIXME}` are filtered;
   `{user_id_here}`, `{INSERT_VALUE}`, `{placeholder}` are not.

5. **No typo repair.** `{{user-name}}` (hyphen) or `{User Name}` (space) are
   silently dropped because they fail the identifier regex. The user sees
   "no variables detected" with no hint that they had near-misses.

6. **Custom syntaxes unsupported.** `<<var>>`, `[[var]]`, `<%= var %>` are
   invisible to the detector. Users with templates in these formats get
   "no variables detected" with no path forward.

7. **No semantic metadata.** The detector returns variable *names* only.
   `description` and `example_value` for EvalGen are filled with stub text
   like `(auto-declared) value for {name}`, which makes generated test
   cases shallow.

## When the rule-based approach is the wrong tool

The detector cannot tell:

- Whether `{name}` in prose like "The `{name}` field stores the display name"
  is documentation or a real placeholder.
- Whether a user typed `{{user-name}}` because they meant `{{user_name}}`
  or because they meant a literal hyphenated string.
- Whether a `{{ x }}` block where `x` is a single weird identifier (e.g.,
  `{{ q }}`) is a real variable or an OCR artifact.

These require semantic understanding the rules do not have.

---

# Improvement plan

In rough order of value-per-effort.

## 1. UI confirmation panel (highest value, no LLM needed)

After Writer extracts variables, show the user what was detected:

```
Detected 4 variables using FORMAT syntax (confidence: high)
  • group
  • group_description
  • subject_template
  • json_fields

1 block at lines 142–168 marked as sample output (won't be substituted).

[ Looks right — continue ]    [ Edit variables ]    [ Re-detect ]
```

This is the single biggest reliability win. Even a *perfect* detector
benefits from explicit user confirmation. A wrong detection becomes a
one-click fix instead of a silent failure.

**Effort**: medium (frontend work in `web/components/experiments/`,
backend already exposes everything needed via the Writer result).

## 2. LLM fallback for uncertainty

Wire `detect()` to call a small LLM (Haiku 4.5 / Gemini Flash) when:

- Score is 0 but template is substantial (`len(template) > 100` and not just prose)
- Top two syntaxes scored within 20% of each other
- User explicitly clicks "Re-detect with AI"

The LLM call:

- Identifies variables in any syntax (including custom ones).
- Suggests repairs for malformed identifiers (`{{user-name}}` → `user_name`).
- Generates description + example_value for each variable.
- Flags sample-output blocks the rules missed.

Cache aggressively: keyed by `sha256(template)`. Pay the LLM cost once per
template, never on rendering.

**Effort**: low (one prompt + a fallback branch in `detect()` + a small
cache table).

## 3. Semantic metadata generation

Currently auto-declared variables get descriptions like
`(auto-declared) value for {name}` and empty `example_value`. EvalGen
relies on these to generate realistic test cases.

When LLM fallback runs, have it also produce:

- One-sentence `description` per variable.
- Realistic `example_value` per variable.
- Optional `role` (user-input vs context) to help `_pick_input_var` in
  `evalgen.py` choose the right slot for `input_text`.

This would noticeably improve EvalGen output quality on warm-mode
experiments.

**Effort**: low (extend the LLM fallback prompt + plumb fields through
`PromptVariable`).

## 4. Typo repair suggestions

On low-confidence detection or 0-vars-found cases, surface a "Did you
mean…?" panel. Examples:

- `{{user-name}}` → did you mean `{{user_name}}`?
- `{User Name}` → did you mean `{user_name}` or `{username}`?
- `${ user }` (space inside) → did you mean `${user}`?

The LLM fallback can produce these as part of detection. The UI surfaces
them as one-click apply-and-update suggestions.

**Effort**: medium (mostly UI work; backend just needs the suggestions
in the response).

## 5. Confidence + diagnostics in the UI

`DetectionResult.confidence` and `DetectionResult.notes` already exist
but aren't surfaced. Show them in the experiment-creation flow so users
understand *why* the detector picked what it picked, and so ambiguity
becomes a visible signal rather than a silent guess.

**Effort**: low (frontend display only).

---

# Why not always-LLM

The temptation to "just use an LLM" was considered and rejected for these
specific reasons:

- **Render path can't afford it.** Rendering happens once per eval-item ×
  many iterations. LLM latency would dominate experiment runtime.
- **Non-determinism breaks CI.** Unit tests can't reliably assert "LLM
  extracted these variables" across runs and model versions.
- **Hallucination is silent corruption.** A detector that invents a
  `{user_id}` not in the template is worse than one that misses a real
  variable — at least the latter fails loudly.
- **Vendor outage = product outage.** Hard dependency on Anthropic/Google
  uptime for a core read-path operation is unacceptable.
- **Prompt injection.** The template *is* the input. A hostile template
  could try to manipulate the detector.

The right architecture is hybrid: rules as the fast deterministic primary,
LLM as the targeted fallback for cases rules can't handle. We have step 1
(rules); steps 2–4 above complete the hybrid.
