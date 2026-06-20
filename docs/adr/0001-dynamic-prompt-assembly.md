# ADR-0001: Dynamic Prompt Assembly with Context Injection

## Status

Proposed

## Context

Currently, every node in the LangGraph agent uses a hard-coded module-level constant for its system prompt (e.g., `EXECUTOR_SYSTEM_PROMPT`, `ROUTER_SYSTEM_PROMPT`). This works for static instructions, but it creates several problems as the agent grows:

1. **Tool drift**: When a new tool is added to `tools/`, every prompt that lists available tools must be manually updated. The executor prompt already enumerates `read_file`, `write_file`, `edit_file`, and `run_command` by hand.
2. **Project rules are invisible**: `AGENTS.md` contains critical project constraints (dependency direction, banned libraries, file size limits, commit conventions), but none of this context reaches the LLM. The agent cannot self-enforce rules it has never read.
3. **No runtime adaptation**: Prompts cannot vary based on the current state. For example, the router cannot be told "the last action failed with X, so reconsider the route" via its system prompt.
4. **Duplication**: Tool usage examples and path rules are copy-pasted across `executor.py`, `planner.py`, and `file_tools.py` docstrings.

## Decision

We will introduce a **PromptTemplate** abstraction that replaces raw string constants. The design follows three layers:

### 1. PromptTemplate (Template Layer)

A `PromptTemplate` is a dataclass that holds a template string with named placeholders and a list of required context keys.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PromptTemplate:
    template: str
    required_keys: set[str]

    def render(self, context: dict[str, str]) -> str:
        missing = self.required_keys - context.keys()
        if missing:
            raise ValueError(f"Missing context keys: {missing}")
        return self.template.format(**context)
```

Templates use Python's built-in `str.format()` — no external templating engine is required, keeping dependencies minimal.

### 2. ContextProvider (Provider Layer)

`ContextProvider` is a protocol (or abstract base class) that produces key-value pairs at runtime. Implementations are stateless and return new dicts on every call.

```python
from typing import Protocol

class ContextProvider(Protocol):
    def provide(self, state: CodingAgentState | None = None) -> dict[str, str]:
        ...
```

Planned implementations:

| Provider | Responsibility | Example Output |
|----------|---------------|----------------|
| `ToolRegistryProvider` | Lists registered tools and their signatures | `{"tools": "read_file(path, offset?, limit?)\nwrite_file(...)"}` |
| `RulesFileProvider` | Reads `AGENTS.md` (or a cached excerpt) | `{"project_rules": "- No extra HTTP clients\n- File size ≤ 400 lines"}` |
| `StateSnapshotProvider` | Injects current plan, errors, step count | `{"current_step": "3/5", "last_error": "File not found"}` |

Providers are **composable**: multiple providers are merged into a single context dict before rendering. If two providers define the same key, the later one wins (explicit override).

### 3. PromptRegistry (Registry Layer)

A single registry object per node holds the templates and the ordered list of providers.

```python
class PromptRegistry:
    def __init__(self):
        self._templates: dict[str, PromptTemplate] = {}
        self._providers: list[ContextProvider] = []

    def register(self, name: str, template: PromptTemplate) -> None:
        self._templates[name] = template

    def add_provider(self, provider: ContextProvider) -> None:
        self._providers.append(provider)

    def render(self, name: str, state: CodingAgentState | None = None) -> str:
        template = self._templates[name]
        context: dict[str, str] = {}
        for provider in self._providers:
            context.update(provider.provide(state))
        return template.render(context)
```

### 4. Token Budget and Context Truncation

Injecting full `AGENTS.md` on every node call is wasteful. We enforce a per-provider token budget via a `TruncationPolicy`.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TruncationPolicy:
    max_chars: int = 2000
    truncation_marker: str = "\n... (truncated)"

    def apply(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars] + self.truncation_marker
```

**Relevance-based filtering**: `RulesFileProvider` does not inject the whole file. It parses `AGENTS.md` into sections and returns only the sections tagged for the requesting node (e.g., executor gets "Tool Usage Rules", planner gets "Architecture Constraints").

### 5. Prompt Injection Safety

`StateSnapshotProvider` may include user-controlled text. If that text contains `{tools}` or other template placeholders, `str.format()` will attempt to replace them, leading to injection or `KeyError`.

**Mitigation**: split context into two channels:

```python
class PromptRegistry:
    def render(self, name: str, state: CodingAgentState | None = None,
               user_context: dict[str, str] | None = None) -> RenderedPrompt:
        # system_context comes from providers (trusted)
        system_context = self._build_system_context(state)
        # user_context is escaped before merging
        safe_user = {k: v.replace("{", "{{").replace("}", "}}")
                     for k, v in (user_context or {}).items()}
        merged = {**safe_user, **system_context}  # system wins on collision
        return self._templates[name].render(merged)
```

User-provided values are brace-escaped so they are treated as literal text by `.format()`.

### 6. Observability and Debugging

When an LLM produces unexpected output, the first debugging step is to inspect the exact prompt it received.

`PromptRegistry.render()` returns a `RenderedPrompt` dataclass instead of a bare string:

```python
@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    sources: list[str]              # which providers contributed
    keys_used: set[str]             # which placeholders were filled
    keys_missing: set[str] | None   # empty if all required keys present
```

Nodes log this at `DEBUG` level (via the stdlib `logging` module) before calling `llm.invoke()`. The log entry is structured and can be grep-ed by node name or provider key.

### 7. Integration with Existing Code

The registry is introduced without breaking existing `llm.invoke(system=...)` calls. Each node follows this pattern:

```python
# executor.py (after migration)
def executor_node(state: CodingAgentState, llm: LLMClient,
                  registry: PromptRegistry) -> dict:
    rendered = registry.render("executor", state=state)
    logger.debug("Executor prompt keys: %s", rendered.keys_used)
    response = llm.invoke(
        prompt=f"Current step: {step}\n\nRespond with the tool call only.",
        system=rendered.text,
    )
    ...
```

`PromptRegistry` is instantiated once in `main.py` and injected into all node functions. This keeps the prompt layer a pure dependency, consistent with the project's "inject dependencies explicitly" rule.

### 8. Graceful Degradation

Not every provider is critical. We distinguish `required` from `optional` providers:

```python
class PromptRegistry:
    def add_provider(self, provider: ContextProvider, required: bool = True) -> None:
        self._providers.append((provider, required))

    def render(self, name: str, state: CodingAgentState | None = None) -> RenderedPrompt:
        context: dict[str, str] = {}
        for provider, required in self._providers:
            try:
                context.update(provider.provide(state))
            except Exception:
                if required:
                    raise
                # optional provider fails silently, continues with partial context
```

For example, `RulesFileProvider` can be marked optional: if `AGENTS.md` is temporarily missing, the agent still works with a slightly degraded prompt.

### Migration Path

1. **Phase 1** (immediate): Create `src/core/prompts.py` with `PromptTemplate`, `PromptRegistry`, `RenderedPrompt`, `TruncationPolicy`, and the `ContextProvider` protocol.
2. **Phase 2**: Convert the executor node first — it benefits the most from auto-generated tool lists. Replace `EXECUTOR_SYSTEM_PROMPT` with a template that accepts `{tools}`.
3. **Phase 3**: Convert router and planner. The router template will accept `{project_rules}` so the LLM knows not to suggest banned dependencies.
4. **Phase 4**: Add a `StateSnapshotProvider` for runtime context (plan progress, recent errors).
5. **Phase 5** (optional): Add `variant` support to `PromptRegistry.render()` for A/B testing prompt versions.

## Consequences

### Positive

- **Single source of truth**: Tool definitions live in `tools/` and are reflected in prompts automatically. Adding `search_codebase` does not require editing `executor.py`.
- **Rule awareness**: The LLM can reference `AGENTS.md` constraints during planning and execution, reducing violations.
- **Testability**: Prompts become pure functions of `(template, context)`. Unit tests can assert exact output strings without calling LLMs.
- **No new dependencies**: `str.format()` is sufficient; no Jinja2 or other templating library is needed.
- **Observability**: `RenderedPrompt` exposes which providers and keys contributed to the final text, making prompt-level debugging straightforward.
- **Safety**: Brace-escaping on user context prevents accidental placeholder substitution and reduces prompt injection surface.
- **Resilience**: Optional providers allow the agent to degrade gracefully when auxiliary context (e.g., `AGENTS.md`) is unavailable.

### Negative / Trade-offs

- **Indirection**: A developer must look at the registry + providers to understand the final prompt, rather than reading a single constant.
- **Runtime cost**: Reading `AGENTS.md` and scanning tool signatures on every node invocation adds I/O. This is mitigated by:
  - `TruncationPolicy` caps injected text length.
  - Relevance-based filtering sends only node-specific rule sections.
  - Explicit caching (e.g., `functools.lru_cache` on provider methods) is allowed, but the cache must be attached to an injected object, not a module global.
- **Error feedback loop**: Required provider failures surface at render time. Optional providers fail silently. Eager validation at registry construction catches missing required keys early.

## Alternatives Considered

| Approach | Why Rejected |
|----------|--------------|
| Jinja2 templating | Adds an external dependency; `str.format()` is sufficient for our use case |
| LangChain `PromptTemplate` | Tightly couples to LangChain internals; we want a thin, framework-agnostic layer |
| Static code generation (generate prompts at build time) | Loses runtime adaptability; cannot inject `state` snapshots |
| Directly pass `state` dict to `.format(**state)` | Too coarse; exposes internal state fields the prompt should not see, and mixes concerns |
