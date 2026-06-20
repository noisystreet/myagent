# ADR-0002: Rolling Plan-Execute-Reflect Loop

## Status

Proposed

## Context

The current task execution pipeline in `myagent` follows a **one-shot planning** model:

```
planner → executor(每步) → output → END
              ↑
         失败时直接跳 output（放弃）
```

This design has three critical limitations:

### 1. No Error Recovery

In [`executor_node`](src/nodes/executor.py), when a tool call fails, the routing logic at line 52 is:

```python
"next_action": "output" if errors else "executor"
```

A single failure immediately terminates execution and jumps to summary. There is no retry, no fix attempt, and no feedback loop. For a coding agent that must handle compilation errors, missing imports, or test failures, this means **any non-trivial task will fail on the first error**.

### 2. Static Plan

The [`planner_node`](src/nodes/planner.py) generates a complete step-by-step plan once at the start of execution. This plan cannot be updated based on what the executor discovers during execution. Common scenarios where this breaks:

- **File not found**: planner assumes a file exists; executor discovers it doesn't. No mechanism to add a creation step.
- **API changed**: executor finds the actual API differs from planner's assumption. Plan becomes invalid.
- **Test reveals new requirement**: running tests exposes edge cases not in original scope.

### 3. No Completion Verification

The [`output_node`](src/nodes/output.py) only formats results into a human-readable summary. It does not verify whether the task was actually accomplished. If the last tool call succeeded but produced wrong output, the agent reports success anyway.

### Industry Precedent

All major coding agents use iterative/rolling approaches:

| Agent | Strategy | Loop Mechanism |
|-------|----------|----------------|
| OpenHands (OpenDevin) | ReAct loop with observation | action → observe → think → next action |
| Devin | Hierarchical planning with replanning | sub-agent loops with parent oversight |
| Cursor Composer | Multi-turn edit-execute cycle | edit → compile → test → edit |
| LangGraph ReAct | Built-in agent loop | act → observe → decide |

## Decision

We will replace the linear pipeline with a **Rolling Plan-Execute-Reflect** loop:

```
                    ┌──────────────────────────────┐
                    │                              │
                    ▼                              │
START → router → planner → executor → reflector ───┘
                              ↑          │
                          continue    ┌───┼────────┐
                                     │   │        │
                                  done  retry   replan
                                     ↓   ↓        ↓
                                   output executor  planner
                                     │
                                     ↓
                                    END
```

The key insight: **after every single step, we observe the result and decide what to do next**, rather than deciding the entire path upfront.

### Architecture Components

#### 1. Planner — Generates and Updates Plans

```python
# src/nodes/planner.py (modified)

PLANNER_SYSTEM_PROMPT = """\
You are a task planner for a coding agent. Break down tasks into ordered steps.

Available tools:
{tools}

Output a JSON object:
{
  "steps": [
    {"id": 1, "action": "read_file", "args": {"path": "..."}, "description": "..."},
    ...
  ]
}

Rules:
- Each step uses exactly one available tool.
- Steps are ordered and dependent.
- Prefer small, verifiable steps over large ones.
- Include verification steps (run tests, check output).
"""
```

**Behavior changes:**
- **First invocation**: Generate plan from scratch (unchanged from current behavior).
- **Subsequent invocations (replan)**: Receive reflection context + completed history. May append, remove, reorder, or completely restructure remaining steps.

#### 2. Executor — Executes One Step

```python
# src/nodes/executor.py (modified)
# Core execution logic unchanged, but routing simplified:

def executor_node(state, llm) -> dict:
    step = state["plan"][state["current_step"]]
    result = _resolve_and_run(step, workspace=state["workspace"])

    # Always route to reflector — no more local error→output shortcut
    return {
        "tool_results": [*state["tool_results"], result],
        "current_step": state["current_step"],
        "next_action": "reflector",  # unified exit point
    }
```

**Key change**: executor no longer makes routing decisions. It executes one step and passes control to reflector.

#### 3. Reflector — Observes and Decides (NEW)

```python
# src/nodes/reflector.py (new file)

REFLECTOR_SYSTEM_PROMPT = """\
You are an execution observer for a coding agent. After each step, you decide the next action.

Your inputs:
- The planned step that was just executed
- The actual tool result (success/failure + output/error)
- History of previous steps and their results
- Remaining steps in the current plan

Decide ONE of these actions:
1. **continue**: Step succeeded. Advance to the next step in the plan.
2. **retry**: Step failed but is fixable. Suggest how to adjust and try again.
3. **replan**: The remaining plan is invalid due to unexpected outcomes.
   Return to the planner to restructure the approach.
4. **done**: The overall task is complete, even if steps remain.

Anti-patterns to avoid:
- Do not retry more than 2 times for the same error type.
- If a file doesn't exist after read_file fails, do not keep trying to read it — suggest replan.
- If tests fail with identical errors 3 times in a row, stop and report done with explanation.
- Do not suggest adding steps that duplicate already-completed work.
"""
```

Schema definition in [`src/llm/schemas.py`](src/llm/schemas.py):

```python
class ReflectDecision(BaseModel):
    """Decision made by the reflector after observing a step result."""

    verdict: Literal["continue", "retry", "replan", "done"]
    reason: str
    suggestion: str | None = None  # hint for retry/replan
```

#### 4. State Changes

```python
# src/core/state.py (modified additions)

class CodingAgentState(TypedDict):
    # ... existing fields ...

    # Rolling loop fields
    reflections: Annotated[list[dict], operator.add]  # reflection history
    loop_count: int       # total iterations (anti-infinite-loop)
    max_loops: int        # global cap, default 30
    max_retries: int      # per-step retry cap, default 2
    completed: bool       # reflector marks task complete
```

#### 5. Graph Wiring

```python
# src/core/graph.py (modified)

def build_graph(llm: LLMClient, config: AgentConfig):
    graph = StateGraph(CodingAgentState)

    # Nodes
    graph.add_node("router", router_node)
    graph.add_node("chat", chat_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    # NEW
    graph.add_node("reflector", reflector_node)
    graph.add_node("output", output_node)

    # Edges
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda s: s.get("next_action"),
        {"chat": "chat", "task": "planner"},
    )
    graph.add_edge("chat", END)
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reflector")  # always go through reflector
    # Reflector routes to next action
    graph.add_conditional_edges(
        "reflector",
        _reflect_route,
        {
            "continue": "executor",
            "retry": "executor",
            "replan": "planner",
            "done": "output",
        },
    )
    graph.add_edge("output", END)

    return graph.compile()
```

### Loop Safety Mechanisms

| Mechanism | Location | Trigger |
|-----------|----------|---------|
| Global loop cap | `graph.py` entry guard | `loop_count >= max_loops` → force output |
| Per-step retry cap | `reflector.py` | Same step retried `max_retries` times → force replan or done |
| Stuck detection | `reflector.py` prompt instruction | LLM instructed to detect repetitive failures |
| Plan exhaustion | `reflector.py` | All steps executed → auto-done |

### Data Flow Example

A realistic scenario: "Add a `search_files` tool that supports glob patterns"

```
Step 0: [planner] Generate initial plan:
  1. Read tools/__init__.py to understand registration pattern
  2. Create src/tools/search_tools.py with search function
  3. Register in __init__.py
  4. Run tests to verify

Step 1: [executor] Read tools/__init__.py → SUCCESS
        [reflector] verdict=continue

Step 2: [executor] Write search_tools.py → SUCCESS
        [reflector] verdict=continue

Step 3: [executor] Edit __init__.py to register → FAIL
        (error: import conflict with existing name)
        [reflector] verdict=retry, suggestion="use different import name"

Step 4: [executor] Retry edit with corrected name → SUCCESS
        [reflector] verdict=continue

Step 5: [executor] Run tests → FAIL
        (error: test_registry expects 5 tools, now has 6)
        [reflector] verdict=replan,
                   reason="test count assertion needs updating"
        [planner] Appends step: update test_registry.py expectation
        [executor] Update test → SUCCESS
        [reflector] verdict=continue

Step 6: [executor] Run tests again → SUCCESS
        [reflector] verdict=done (all steps complete, tests pass)

[output] Summarize: Added search_files tool, fixed import naming, updated tests
```

## Design Considerations (Cross-Cutting Factors)

This section covers factors that span multiple components and must be addressed holistically rather than within any single node.

### C1. Error Classification (Severity Levels)

Current [`ToolResult`](src/core/state.py) uses binary `success: bool`. In a rolling loop, not all failures are equal:

| Severity | Example | Expected Response | ReflectAction |
|----------|---------|-------------------|---------------|
| **recoverable** | SyntaxError, ImportError, test assertion fail | Auto-retry or auto-fix | `retry` (with suggestion) |
| **confirm-required** | File deletion, `rm -rf`, modifying `.env` | Ask user before proceeding | `interrupt` (new branch) |
| **unrecoverable** | API key invalid, network unreachable, permission denied | Stop with explanation | `done` (with error summary) |
| **partial-success** | 8/10 tests pass, lint warns but passes | Continue with warning or replan | `continue` + warning annotation |

**Implementation approach:** Add a `severity` field to `ReflectDecision`. The reflector LLM classifies the failure severity as part of its reasoning. This avoids changing `ToolResult`'s interface (backward compatible) while enabling richer routing logic.

```python
class ReflectDecision(BaseModel):
    verdict: Literal["continue", "retry", "replan", "done", "interrupt"]
    reason: str
    suggestion: str | None = None
    severity: Literal["info", "warning", "error", "critical"] = "info"
    warning: str | None = None  # populated when severity=warning
```

The `interrupt` verdict connects to the existing [`human_confirm`](src/core/config.py) flag already defined in `AgentConfig` but never implemented. When `human_confirm=True`, `interrupt` pauses execution and presents the pending action to the user; when `False`, `interrupt` falls through to `done`.

### C2. Three-Tier Loop Safety (Precise Semantics)

The ADR previously listed four safety mechanisms. Here we define their precise interaction semantics:

```
Tier 1: Per-step retry cap (fastest, most local)
  - Counter: retries_for_current_step (reset on step advance or replan)
  - Limit: max_retries (default 2)
  - Action: exceeded → force replan (not done, because the step might be salvageable)

Tier 2: Global loop cap (medium, prevents resource exhaustion)
  - Counter: loop_count (incremented on every executor→reflector cycle)
  - Limit: max_loops (default 30)
  - Action: exceeded → force output with "max loops reached" summary

Tier 3: Stuck detection (smartest, catches semantic repetition)
  - Detector: compare last N reflections for similar error patterns
  - Signal: same error_type + same step_description in 3 consecutive reflections
  - Action: force done with "stuck" diagnosis, bypassing further LLM calls
```

These tiers are checked in order: Tier 1 first (cheapest), then Tier 2, then Tier 3. Any tier triggering short-circuits the rest. The checks happen in `_reflect_route` (the conditional edge function) **before** calling the reflector LLM, so stuck detection does not waste tokens on an obvious infinite loop.

### C3. Progress Display During Loop Execution

The project supports streaming output ([`streaming_output`](src/main.py)). The rolling loop creates a multi-step execution scenario where the user needs real-time feedback about progress, not just the final result.

**Problem:** Without intermediate display, the user sees:
1. "Thinking..." (long pause, possibly 30+ seconds for complex tasks)
2. Final output dump (everything at once)

This is worse UX than the current linear model because the loop introduces more latency per effective unit of work.

**Proposed display format** (rendered after each reflector decision):

```
⠋ Step 3/7: edit_file(src/utils.py)     ✅ done
⠋ Step 4/7: run_command(make test)      ❌ failed → retry (ImportError)
⠋ Step 4/7: run_command(make test)      ✅ done
⠋ Step 5/7: read_file(test_utils.py)    ⏳ running...
```

**Design rules:**
- Display happens **after reflector decides**, not after executor runs (so the verdict is known)
- Each line overwrites the previous one (using `\r` carriage return, same technique as current streaming)
- Final line includes total stats: `Done: 7 steps, 2 retries, 0 replans, 45s elapsed`
- When `verdict=interrupt`, display pauses and waits for user input (`y/n`)
- Output is written to stderr (to separate from final stdout result)

**Implementation location:** A lightweight `ProgressRenderer` in [`src/main.py`](src/main.py) (or extracted to `src/core/render.py` if it exceeds 50 lines). It subscribes to state changes via a callback or post-processing hook.

### C4. Token Budget Integration

[`AgentConfig.cost_budget`](src/core/config.py) is defined as `float = 0.0` (0 means unlimited). The rolling loop amplifies token consumption significantly:

```
Cost formula per iteration:
  executor_call_tokens  ≈ 500-2000  (depends on step complexity)
  reflector_call_tokens ≈ 300-800   (structured decision output)
  planner_call_tokens   ≈ 800-2500  (only on replan)

Typical 10-step task:
  Best case:  10 × (executor + reflector)          = ~15K tokens
  With 2 replans: 12 × (executor + reflector) + 2 × planner = ~22K tokens
  Worst case (max retries): up to ~60K tokens
```

**Budget enforcement strategy:**

```python
# Pseudocode in graph entry guard or reflector pre-check
def check_budget(state, llm) -> str | None:
    if state.get("cost_budget", 0) <= 0:
        return None  # unlimited

    estimated_next = estimate_tokens(state)  # rough heuristic
    used_so_far = state.get("total_tokens_used", 0)

    if used_so_far + estimated_next > state["cost_budget"]:
        return "done"  # force termination
    return None
```

**Token tracking approach:**
- Add `total_tokens_used: int` to `CodingAgentState`
- After each `llm.invoke()`, accumulate `response.usage.total_tokens` (available from OpenAI/DeepSeek responses)
- Check budget in the `_reflect_route` edge function (same place as loop caps)
- On forced termination, include remaining budget info in the output summary

**Per-node model selection** (cost optimization):
- `reflector`: can use a cheaper/faster model (decisions are structurally simple)
- `planner`: should use the strongest available model (plan quality is critical)
- `executor`: uses the default configured model
- This requires extending [`LLMClient`](src/llm/client.py) to support per-call model override, which is a natural extension of the existing `model` parameter.

### C5. Chat / Task Unification (Runtime Intent Switching)

Currently, [`router_node`](src/nodes/router.py) classifies intent **once at entry** and commits to either chat or task mode for the entire session. In practice, users often need to switch modes mid-execution:

> User: "Add error handling to the API client"
> Agent: *(starts executing plan)*
> User: "Wait, use Result type instead of exceptions"
> Agent: *(must abandon current plan and start over)*

**Two options evaluated:**

| Option | Mechanism | Pros | Cons |
|--------|----------|------|------|
| **A. Router re-entry** | After each reflector cycle, re-run router on latest user message (if any new input arrived) | Clean separation; router logic reused | Adds latency per cycle; may cause mode thrashing |
| **B. Interrupt branch** | Reflector emits `interrupt`; CLI layer accepts user input; input fed back as context to next planner call | Natural conversational flow; user feels in control | Requires async I/O in the graph loop; complicates state management |

**Recommendation: Option B (Interrupt branch)** for Phase 2+. It maps naturally onto the `interrupt` verdict from C1 and the `human_confirm` config flag. The flow becomes:

```
executor → reflector → interrupt → [CLI waits for user input]
                                  → user types new instruction
                                  → planner(receives original_request + new_instruction + history)
                                  → executor → ...
```

For Phase 1, the reflector's `interrupt` verdict simply falls through to `done` (same behavior as current system, just with an explicit label).

### C6. Resumability via Persistent Checkpointing

Current [`graph.py`](src/core/graph.py) uses `MemorySaver()` — an in-memory checkpointer that loses all state on process restart. The rolling loop increases session duration and thus the impact of crashes:

```
Linear model:  3 steps × ~2s each = ~6s   (crash unlikely, low impact)
Rolling loop: 15-30 iterations × ~5s each = ~75-150s  (crash impactful)
```

**Migration path:**

| Checkpointer | Persistence | Latency | Use Case |
|-------------|------------|---------|----------|
| `MemorySaver` (current) | No | Lowest | Development, short tasks |
| `SqliteSaver` | Yes (file) | Low (~1ms) | Production, long tasks |
| `AsyncPostgresSaver` | Yes (remote) | Medium | Distributed deployment |

**Recommendation for Phase 1:** Keep `MemorySaver` but structure state so migration is mechanical (no schema changes needed later). Specifically:
- Ensure all new state fields (`reflections`, `loop_count`, etc.) are JSON-serializable (they are: `int`, `bool`, `list[dict]` with primitive values)
- Avoid storing file handles, connections, or non-serializable objects in state
- Document the checkpoint restore procedure: load state → set `next_action="reflector"` → resume

**Phase 2+ upgrade:** Replace `MemorySaver` with `SqliteSaver(path=".myagent/checkpoints.db")`. Zero code changes needed in nodes — only the `graph.compile(checkpointer=...)` call changes.

### C7. Prompt Injection Defense in Multi-Turn Loop

ADR-0001 addresses prompt injection for single-turn calls. The rolling loop **amplifies** the attack surface because:

1. **Accumulated injection vectors**: `tool_results` contains file contents that may have adversarial text (e.g., a source file containing `{tools}` or system prompt leakage patterns)
2. **Reflection feedback loop**: If a poisoned `ToolResult` influences the reflector's `reason` field, that poisoned text propagates into the *next* planner or executor prompt
3. **Multi-round amplification**: An injected string that survives one round appears in the *next* round's context, compounding

**Defense layers (ordered by application point):**

| Layer | Where Applied | Technique |
|-------|--------------|-----------|
| **L1: Input sanitization** | Executor → ToolResult storage | Strip/escape template placeholders (`{`, `}`, `[TOOL_CALL]`) from tool output data before storing in state |
| **L2: Reflection truncation** | Reflector output → state | Cap `reason` field at 500 chars; cap `suggestion` at 200 chars. Long LLM outputs cannot inject large payloads |
| **L3: Context window isolation** | Prompt assembly | Pass reflector inputs as a structured section with clear delimiters, not as free-form text mixed into the system prompt template |
| **L4: Loop-based degradation monitor** | Edge function | Track similarity between consecutive reflections; if `reason` field repeats or grows monotonically, force-done with "degradation detected" |

**L1 implementation detail** (most critical):

```python
# In executor_node, before appending to tool_results
def sanitize_tool_result(result: ToolResult) -> ToolResult:
    """Remove prompt-template-unsafe characters from tool output."""
    if result.data is None:
        return result
    safe_data = _sanitize_dict(result.data)
    return ToolResult(result.tool, result.success, data=safe_data,
                       error=result.error)


def _sanitize_dict(d: dict) -> dict:
    """Recursively escape { and } in string values."""
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = v.replace("{", "{{").replace("}", "}}")
        elif isinstance(v, dict):
            out[k] = _sanitize_dict(v)
        else:
            out[k] = v
    return out
```

Note: This is the same brace-escaping technique specified in ADR-0001 Section 5, applied specifically to the rolling loop's accumulated state.

### C8. Parallel Step Execution (Future Extension)

Current plan is a linear list: `[step1, step2, step3, ...]`. Some steps are independent and could execute concurrently:

```
# Current (serial):
1. Create utils.py
2. Create config.py         # independent of step 1
3. Update __init__.py       # depends on both 1 AND 2

# Ideal (parallel):
[Create utils.py, Create config.py]  → concurrent batch
Update __init__.py                     → serial (waits for both)
```

**Why NOT included in Phase 1:**
- Parallel execution requires DAG-based plans (not linear lists), significantly increasing planner complexity
- Concurrent tool calls need careful conflict detection (two edits to the same file)
- Reflector must handle batch results (partial success within a batch)
- Error recovery becomes combinatorially more complex

**Recorded as a known limitation:** The plan schema uses `list[Step]` (linear). Future evolution to `DAG[Step]` is possible without breaking the reflector interface—the reflector operates on *one logical unit of work* regardless of whether that unit was a single step or a batch. The executor would need a `batch_execute` variant, but that is additive, not breaking.

## Consequences

### Positive

- **Error recovery**: Failed steps are retried or trigger replanning instead of terminating the entire session.
- **Adaptive plans**: The plan evolves as the executor discovers reality (missing files, API differences, test failures).
- **Separation of concerns**: executor focuses on tool calling; reflector focuses on decision-making. Each node stays under 150 lines.
- **Industry alignment**: Matches the proven patterns used by OpenHands, Cursor, and other production agents.
- **Observable**: Every decision point produces a logged `ReflectDecision`, making it easy to debug why the agent chose a particular path.
- **Graceful degradation**: Three-tier safety (per-step retry → global cap → stuck detection) ensures the agent always terminates.
- **User control**: `interrupt` verdict + `human_confirm` config gives users veto power over dangerous actions.
- **Cost awareness**: Token budget tracking and per-node model selection prevent runaway spending.
- **Resilient**: Sanitized state and reflection truncation limit prompt injection surface across multiple turns.

### Negative / Trade-offs

- **Increased token usage**: Each step adds a reflector LLM call (~500-1000 tokens). Mitigated by using fast/cheap models for reflection and by token budget enforcement (C4).
- **Longer latency per step**: Two LLM calls per iteration (executor + reflector) vs one. Acceptable for coding tasks where correctness matters more than speed. Progress display (C3) mitigates perceived latency.
- **Replan risk**: Poorly designed reflector prompts could cause excessive replanning (plan thrashing). Addressed by anti-pattern instructions in the system prompt, retry caps (C2), and stuck detection (C2 Tier 3).
- **State complexity**: New fields (`reflections`, `loop_count`, `completed`, `total_tokens_used`) increase state surface area. All are simple types (int, bool, list) with clear semantics and JSON-serializable for checkpointing (C6).
- **Prompt injection surface**: Multi-turn accumulation of untrusted content (file contents, LLM outputs) expands attack surface. Addressed by four defense layers (C7).
- **No parallel execution**: Linear plan limits throughput for independent steps. Documented as future extension (C8), not a blocker for Phase 1.

## Alternatives Considered

| Approach | Why Rejected |
|----------|--------------|
| **Executor self-routes** (current) | Executor mixes execution logic with routing decisions; violates single responsibility; no room for nuanced judgment (e.g., "this error is recoverable" vs "give up") |
| **Fixed retry count** (no LLM reflection) | Simple retry without understanding *why* it failed leads to repeating the same mistake. A syntax error might need a different fix than a missing file error. |
| **Full ReAct loop** (LangChain built-in) | LangChain's ReAct agent is opinionated about tool schema and message format. Our custom nodes give us full control over prompts, schemas, and error handling. Also avoids unnecessary dependency on langchain-agents. |
| **Hierarchical planning** (like Devin) | Over-engineered for current scale. A single-level rolling loop handles most coding tasks. Can be added later as a layer above without changing this design. |
| **Event-driven architecture** (pub/sub between nodes) | Over-abstraction for a sequential workflow. The graph's explicit edges already provide clear data flow visibility. Events would obscure the execution order. |

## Implementation Phases

### Phase 1: Reflect Node + Basic Routing

**Goal**: Replace the linear `planner → executor → output` pipeline with a minimal rolling loop that can retry failed steps. No replan support yet — on unrecoverable failure, still terminates to output.

**Scope**: ~250 lines new code, ~40 lines modified. Single PR.

---

#### 1.1 Task Breakdown

| # | Task | File(s) | Lines | Depends On |
|---|------|---------|-------|------------|
| 1.1.1 | Add `ReflectDecision` schema | [`src/llm/schemas.py`](src/llm/schemas.py) | +15 | — |
| 1.1.2 | Add loop fields to state | [`src/core/state.py`](src/core/state.py) | +10 | — |
| 1.1.3 | Create reflector node | [`src/nodes/reflector.py`](src/nodes/reflector.py) (new) | ~80 | 1.1.1, 1.1.2 |
| 1.1.4 | Create edge route function | [`src/core/graph.py`](src/core/graph.py) | +15 | 1.1.2 |
| 1.1.5 | Wire reflector into graph | [`src/core/graph.py`](src/core/graph.py) | +20 | 1.1.3, 1.1.4 |
| 1.1.6 | Simplify executor routing | [`src/nodes/executor.py`](src/nodes/executor.py) | -8/+5 | — |
| 1.1.7 | Add input sanitization in executor | [`src/nodes/executor.py`](src/nodes/executor.py) | +25 | C7 L1 |
| 1.1.8 | Unit tests for reflector | [`tests/test_reflector.py`](tests/test_reflector.py) (new) | ~60 | 1.1.3 |
| 1.1.9 | Tests for new state fields | [`tests/test_state.py`](tests/test_state.py) | +20 | 1.1.2 |
| 1.1.10 | Integration test: happy path loop | [`tests/test_graph.py`](tests/test_graph.py) | +40 | 1.1.3–1.1.6 |
| 1.1.11 | Integration test: retry on failure | [`tests/test_graph.py`](tests/test_graph.py) | +30 | 1.1.3–1.1.6 |

**Dependency graph**:
```
[1.1.1] ──┐
           ├──→ [1.1.3] ──┬──→ [1.1.5] ──┬──→ [1.1.10]
[1.1.2] ──┤              │                │
           ├──→ [1.1.4] ─┘                ├──→ [1.1.11]
                                    [1.1.6] ┘
```

Tasks 1.1.1, 1.1.2, 1.1.6 are independent and can be done in parallel.

---

#### 1.2 Detailed Specifications

##### Task 1.1.1: ReflectDecision Schema

```python
# src/llm/schemas.py — add after existing schemas

class ReflectDecision(BaseModel):
    """Decision made by the reflector after observing a step result."""

    verdict: Literal["continue", "retry", "replan", "done", "interrupt"]
    reason: str                    # Human-readable explanation (max 500 chars enforced at write)
    suggestion: str | None = None  # Hint for retry/replan (max 200 chars)
    severity: Literal["info", "warning", "error", "critical"] = "info"
```

**Acceptance criteria**:
- [ ] Schema validates all 5 verdict values
- [ ] `reason` is required, `suggestion` optional
- [ ] `severity` defaults to `"info"`
- [ ] Pydantic v2 compatible (uses `Literal` from `typing`)

##### Task 1.1.2: State Field Additions

```python
# src/core/state.py — additions only, no removals or renames

class CodingAgentState(TypedDict):
    # ... ALL existing fields unchanged ...

    # Rolling loop fields (Phase 1)
    reflections: Annotated[list[dict], operator.add]   # reflection history
    loop_count: int                                     # total iterations
    max_loops: int                                      # global cap, default 30
    max_retries: int                                    # per-step cap, default 2
    completed: bool                                     # reflector marks done
```

**Acceptance criteria**:
- [ ] All existing fields preserved (backward compat)
- [ ] New fields have sensible defaults in graph entry point
- [ ] `reflections` uses `Annotated[list, operator.add]` for reducer semantics
- [ ] All new field types are JSON-serializable (for future C6 migration)
- [ ] Existing tests still pass without modification

##### Task 1.1.3: Reflector Node

```python
# src/nodes/reflector.py (new file)

"""Reflector node: observes step results and decides next action."""

import logging

from ..llm.client import LLMClient
from ..llm.schemas import ReflectDecision

logger = logging.getLogger(__name__)

# AI-generated

REFLECTOR_SYSTEM_PROMPT = """\
You are an execution observer for a coding agent. After each step, decide what to do next.

Inputs provided:
- current_step: The plan step that was just executed
- tool_result: Success/failure status, output data, or error details
- recent_history: Last 3 steps and their results (for context)
- remaining_plan: Steps not yet executed

Decide ONE action:
- **continue**: Step succeeded, advance to next step.
- **retry**: Step failed but fixable. Provide a specific suggestion.
- **replan**: Remaining plan is invalid. Needs full re-planning.
- **done**: Task complete (even if plan has remaining steps).
- **interrupt**: Requires human confirmation before proceeding.

Rules:
- Do not retry the same error more than twice total.
- If a file read fails, do not suggest re-reading it.
- If tests fail identically 3 times, choose done with explanation.
- Keep reason under 500 characters.
- Keep suggestion under 200 characters if provided.
"""


def _format_step(step: dict) -> str:
    """Format a single plan step for the prompt."""
    return f"  [{step.get('id', '?')}] {step.get('action', '?')}({step.get('args', {})}) — {step.get('description', '')}"


def _format_result(result: dict) -> str:
    """Format a ToolResult-like dict for the prompt."""
    if result.get("success"):
        return f"  ✅ {result.get('tool', '?')}: {str(result.get('data', ''))[:200]}"
    err = result.get("error", {})
    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
    return f"  ❌ {result.get('tool', '?')}: {msg[:200]}"


def _build_reflect_prompt(
    current_step: dict,
    tool_result: dict,
    recent_history: list[dict],
    remaining_plan: list[dict],
) -> str:
    """Build the user prompt for the reflector LLM call."""
    parts = [
        f"Current step:\n{_format_step(current_step)}",
        f"\nResult:\n{_format_result(tool_result)}",
    ]
    if recent_history:
        parts.append("\nRecent history:")
        for h in recent_history[-3:]:
            parts.append(_format_result(h))
    if remaining_plan:
        parts.append("\nRemaining plan:")
        for s in remaining_plan[:5]:  # cap display to avoid token blowup
            parts.append(_format_step(s))
        if len(remaining_plan) > 5:
            parts.append(f"  ... and {len(remaining_plan) - 5} more steps")
    else:
        parts.append("\nNo remaining steps.")
    return "\n".join(parts)


def reflector_node(state: dict, llm: LLMClient) -> dict:
    """Observe the latest step result and decide the next action."""
    current_idx = state.get("current_step", 0)
    plan = state.get("plan", [])
    tool_results = state.get("tool_results", [])

    if not plan or current_idx >= len(plan):
        return {
            "completed": True,
            "next_action": "done",
            "reflections": [{"verdict": "done", "reason": "Plan exhausted"}],
        }

    current_step = plan[current_idx]
    last_result = tool_results[-1] if tool_results else {"tool": "unknown", "success": False, "error": "No results"}
    recent_history = tool_results[:-1]  # exclude current step's result
    remaining = plan[current_idx + 1:]

    prompt = _build_reflect_prompt(current_step, last_result, recent_history, remaining)

    try:
        decision: ReflectDecision = llm.invoke(
            prompt=prompt,
            system=REFLECTOR_SYSTEM_PROMPT,
            schema=ReflectDecision,
        )
    except Exception as e:
        logger.warning("Reflector LLM call failed: %s", e)
        # Fallback: if last step succeeded continue, otherwise done
        fallback_verdict = "continue" if last_result.get("success") else "done"
        decision = ReflectDecision(
            verdict=fallback_verdict,
            reason=f"LLM fallback: {e}",
        )

    # Enforce length limits (C7 L2)
    reason = decision.reason[:500] if len(decision.reason) > 500 else decision.reason
    suggestion = None
    if decision.suggestion:
        suggestion = decision.suggestion[:200] if len(decision.suggestion) > 200 else decision.suggestion

    reflection_entry = {
        "step_index": current_idx,
        "verdict": decision.verdict,
        "reason": reason,
        "suggestion": suggestion,
        "severity": decision.severity,
    }

    # Phase 1: interrupt falls through to done
    next_action = decision.verdict if decision.verdict != "interrupt" else "done"

    return {
        "next_action": next_action,
        "completed": next_action == "done",
        "reflections": [reflection_entry],
        "loop_count": state.get("loop_count", 0) + 1,
    }
```

**Acceptance criteria**:
- [ ] Function accepts state dict + LLMClient, returns state update dict
- [ ] Calls `llm.invoke()` with `schema=ReflectDecision`
- [ ] On LLM failure, returns safe fallback (not exception)
- [ ] Caps `reason` at 500 chars, `suggestion` at 200 chars (C7 L2)
- [ ] Handles empty plan / out-of-bounds index gracefully
- [ ] Limits displayed history to last 3 items, remaining plan to 5 items
- [ ] `interrupt` verdict maps to `done` in Phase 1
- [ ] Total file size ≤ 150 lines (exempting docstrings/comments)
- [ ] Uses `logging` not `print()`

##### Task 1.1.4–1.1.5: Graph Wiring & Edge Route

```python
# src/core/graph.py — additions

def _reflect_route(state: dict) -> str:
    """Route after reflector. Enforces safety caps before LLM verdict."""
    # C2 Tier 1: per-step retry cap
    reflections = state.get("reflections", [])
    if reflections:
        current_idx = state.get("current_step", 0)
        retries_on_current = sum(
            1 for r in reflections[-5:]
            if r.get("step_index") == current_idx and r["verdict"] == "retry"
        )
        max_retries = state.get("max_retries", 2)
        if retries_on_current >= max_retries:
            logger.info(
                "Per-step retry cap reached (%d/%d) for step %d, forcing replan",
                retries_on_current, max_retries, current_idx,
            )
            return "replan"

    # C2 Tier 2: global loop cap
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 30)
    if loop_count >= max_loops:
        logger.info(
            "Global loop cap reached (%d/%d), forcing output",
            loop_count, max_loops,
        )
        return "done"

    # Default: use reflector's verdict
    return state.get("next_action", "done")
```

In `build_graph()`:
```python
# After existing node definitions:
graph.add_node("reflector", reflector_node)

# Replace old executor→output conditional edge:
# OLD: graph.add_conditional_edges("executor", ..., {"output": "output", "executor": "executor"})
# NEW:
graph.add_edge("executor", "reflector")
graph.add_conditional_edges(
    "reflector",
    _reflect_route,
    {"continue": "executor", "retry": "executor", "replan": "planner", "done": "output"},
)
```

**Acceptance criteria**:
- [ ] `_reflect_route` checks per-step retry cap first (cheapest check)
- [ ] Then checks global loop cap
- [ ] Then delegates to reflector's `next_action`
- [ ] Retry cap forces `replan` (not `done`) — step may be salvageable
- [ ] Global cap forces `done` with logged message
- [ ] Executor always routes to reflector (no more direct output path)
- [ ] Replan routes back to planner (loop closed)

##### Task 1.1.6: Executor Simplification

In [`executor_node`](src/nodes/executor.py), change the routing logic from:

```python
# BEFORE:
errors = [r for r in results if not r.success]
return {
    ...
    "next_action": "output" if errors else "executor",
}
```

To:

```python
# AFTER:
return {
    ...
    "next_action": "reflector",  # always go through reflector
}
```

Remove any local error-counting routing. The reflector now owns all routing decisions.

**Acceptance criteria**:
- [ ] Executor no longer contains routing logic (`if errors` branches removed)
- [ ] Executor always sets `next_action="reflector"`
- [ ] Executor behavior otherwise unchanged (still calls tools, increments step)

##### Task 1.1.7: Input Sanitization (C7 L1)

Add a sanitization function in executor (or extract to a shared util if reused):

```python
def _sanitize_value(v: Any) -> Any:
    """Recursively escape brace characters in string values to prevent prompt injection."""
    if isinstance(v, str):
        return v.replace("{", "{{").replace("}", "}}")
    if isinstance(v, dict):
        return {k: _sanitize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_sanitize_value(item) for item in v]
    return v
```

Apply to `ToolResult.data` before appending to `tool_results`.

**Acceptance criteria**:
- [ ] All string values in `ToolResult.data` have `{` and `}` escaped
- [ ] Nested dicts/lists are recursively sanitized
- [ ] Non-string types (int, bool, None) pass through unchanged
- [ ] Sanitization happens after tool execution, before state update
- [ ] Original tool output is NOT mutated (sanitized copy stored)

---

#### 1.3 Test Plan

##### Unit Tests ([`tests/test_reflector.py`](tests/test_reflector.py))

| Test Case | What It Validates |
|-----------|-------------------|
| `test_continue_on_success` | Successful tool result → verdict=continue |
| `test_retry_on_recoverable_error` | SyntaxError in result → verdict=retry with suggestion |
| `test_replan_on_file_not_found` | Repeated missing file → verdict=replan |
| `test_done_on_plan_exhausted` | Empty remaining plan → verdict=done |
| `test_fallback_on_llm_failure` | LLM raises exception → safe fallback verdict |
| `test_interrupt_maps_to_done` | verdict=interrupt → next_action=done (Phase 1) |
| `test_reason_truncated_at_500` | Long reason (>500 chars) → capped |
| `test_suggestion_truncated_at_200` | Long suggestion (>200 chars) → capped |
| `test_empty_plan_handling` | Empty plan list → done, no crash |
| `test_history_capped_at_3` | 10 previous results → only last 3 in prompt |
| `test_remaining_plan_capped_at_5` | 20 remaining steps → only 5 shown + "...and 15 more" |

##### Integration Tests ([`tests/test_graph.py`](tests/test_graph.py))

| Test Case | What It Validates |
|-----------|-------------------|
| `test_happy_path_loop_3_steps` | 3-step plan, all succeed → executor×3 → reflector×3 → output |
| `test_retry_then_continue` | Step fails once, retry succeeds → continues to next step |
| `test_retry_cap_forces_replan` | Same step fails 3 times (max_retries=2) → forces replan |
| `test_global_loop_cap` | 31 iterations (max_loops=30) → forces output |
| `test_executor_always_goes_to_reflector` | Even on error, next_action=reflector never=output directly |
| `test_replan_returns_to_planner` | Verdict=replan → next node is planner, not executor |

##### State Tests ([`tests/test_state.py`](tests/test_state.py))

| Test Case | What It Validates |
|-----------|-------------------|
| `test_new_fields_have_defaults` | Fresh state has loop_count=0, completed=False, etc. |
| `test_reflections_reducer_appends` | Annotated list appends, doesn't overwrite |
| `test_all_new_fields_json_serializable` | json.dumps(state) succeeds without custom encoders |

---

#### 1.4 Verification Checklist

Before declaring Phase 1 complete:

```bash
# 1. Code quality
make lint          # ruff must pass
make complexity    # lizard: no function > 150 lines, no file > 400 lines

# 2. All tests (existing + new)
make test-cov      # coverage ≥34% threshold

# 3. Specific new test files run
python -m pytest tests/test_reflector.py -v
python -m pytest tests/test_graph.py -v

# 4. Manual smoke test (end-to-end)
python -m src.main 'read the file src/main.py and summarize it'
# Expected: should enter task mode, execute 2 steps (read_file → output), exit cleanly
# Should NOT show the old "direct to output on error" behavior

# 5. Verify reflector logs
# Run with DEBUG logging, confirm reflector decisions appear in log output
LOG_LEVEL=DEBUG python -m src.main 'what time is it' 2>&1 | grep -i reflector
```

---

#### 1.5 Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Reflector LLM produces invalid verdict (typo in enum) | Medium | Loop breaks (route key miss) | `_reflect_route` uses `.get()` with default `"done"`; unknown verdicts safely terminate |
| Token cost doubles (every step needs reflector call) | Certain | Medium | Documented in Consequences; Phase 3 will add per-node model selection (C4) |
| Existing tests break due to state shape change | Low | High | New fields use `Annotated` with reducers; old code ignoring them is unaffected |
| Infinite loop despite caps (replan creates new steps forever) | Low | Critical | Global `max_loops=30` is absolute ceiling; cannot be bypassed by replan |
| Reflector quality varies by model (weak model = bad decisions) | High | Medium | Fallback logic on LLM failure; Phase 3 will add stuck detection (C2 T3) |

---

### Phase 2: Enhanced Planner (Replan Support)

**Goal**: Enable the reflector's `replan` verdict to actually work. Planner can receive execution history and generate corrected plans. Also adds progress display and basic user interruption.

**Scope**: ~300 lines new code, ~60 lines modified. 1–2 PRs.

---

#### 2.1 Task Breakdown

| # | Task | File(s) | Lines | Depends On |
|---|------|---------|-------|------------|
| 2.1.1 | Add replan context builder | [`src/nodes/planner.py`](src/nodes/planner.py) | +40 | Phase 1 |
| 2.1.2 | Modify planner to detect replan mode | [`src/nodes/planner.py`](src/nodes/planner.py) | +30 | 2.1.1 |
| 2.1.3 | Plan merge logic (keep completed, regenerate rest) | [`src/nodes/planner.py`](src/nodes/planner.py) | +35 | 2.1.2 |
| 2.1.4 | Reset loop counters on replan | [`src/core/graph.py`](src/core/graph.py) | +10 | Phase 1 |
| 2.1.5 | Progress renderer (C3) | [`src/core/render.py`](src/core/render.py) (new) | ~80 | Phase 1 |
| 2.1.6 | Wire renderer into main loop | [`src/main.py`](src/main.py) | +15 | 2.1.5 |
| 2.1.7 | Interrupt handler skeleton (C5) | [`src/main.py`](src/main.py) | +25 | Phase 1 |
| 2.1.8 | Tests: replan happy path | [`tests/test_planner.py`](tests/test_planner.py) | +50 | 2.1.1–2.1.3 |
| 2.1.9 | Tests: plan merge edge cases | [`tests/test_planner.py`](tests/test_planner.py) | +40 | 2.1.3 |
| 2.1.10 | Tests: progress renderer | [`tests/test_render.py`](tests/test_render.py) (new) | ~40 | 2.1.5 |
| 2.1.11 | E2E: replan scenario | [`tests/test_graph.py`](tests/test_graph.py) | +50 | 2.1.1–2.1.4 |

---

#### 2.2 Detailed Specifications

##### Task 2.1.1–2.1.3: Replan-Capable Planner

The planner must distinguish two invocation modes:

**First call (no reflections)**: Behaves exactly as today — generates plan from scratch.

**Replan call (has reflections)**: Receives additional context:

```python
def _build_replan_context(state: dict) -> str:
    """Build context string for replan mode."""
    original_request = state.get("messages", [{}])[-1].get("content", "")
    reflections = state.get("reflections", [])
    completed_steps = state.get("plan", [])[:state.get("current_step", 0)]
    results = state.get("tool_results", [])

    parts = [
        f"Original request: {original_request}",
        f"\n=== Completed steps ({len(completed_steps)}) ===",
    ]
    for i, step in enumerate(completed_steps):
        r = results[i] if i < len(results) else {}
        status = "✅" if r.get("success") else "❌"
        parts.append(f"  {status} [{i+1}] {step.get('description', '')}")

    parts.append("\n=== Latest reflection (why we're replanning) ===")
    if reflections:
        last = reflections[-1]
        parts.append(f"  Verdict: {last['verdict']}")
        parts.append(f"  Reason: {last['reason']}")
        if last.get("suggestion"):
            parts.append(f"  Suggestion: {last['suggestion']}")

    parts.append("\n=== Old remaining plan (may be partially valid) ===")
    old_remaining = state.get("plan", [])[state.get("current_step", 0):]
    for step in old_remaining:
        parts.append(f"  - {step.get('description', '')}")

    parts.append(
        "\nGenerate updated REMAINING steps only. "
        "Keep completed steps as-is. You may remove, add, or reorder remaining steps."
    )
    return "\n".join(parts)
```

**Plan merge strategy** (Task 2.1.3):

```python
def _merge_plan(completed: list[dict], new_remaining: list[dict]) -> list[dict]:
    """Merge completed steps with newly generated remaining steps."""
    # Renumber IDs sequentially
    merged = list(completed)
    next_id = len(completed) + 1
    for step in new_remaining:
        merged.append({**step, "id": next_id})
        next_id += 1
    return merged
```

**Key design decisions**:
- Completed steps are **never modified** by replan — they represent historical truth
- Only remaining steps are regenerated
- Step IDs are renumbered after merge to stay sequential
- If the planner returns an empty list for remaining, the loop naturally ends (reflector sees empty plan → done)

**Acceptance criteria**:
- [ ] First call (empty reflections): identical behavior to pre-Phase 2
- [ ] Replan call: includes completed steps, latest reflection, old remaining plan in prompt
- [ ] Merge preserves completed steps unchanged
- [ ] Merge renumbers step IDs sequentially
- [ ] Empty new_remaining → plan ends naturally (no special handling needed)
- [ ] Planner file stays ≤ 400 lines (may need extraction if it grows)

##### Task 2.1.4: Counter Reset on Replan

When the edge function routes to `"replan"` (via reflector or via forced retry cap), reset per-step counters but preserve global counter:

```python
# In graph.py or a helper called during replan transition
def _on_replan(state: dict) -> dict:
    """Reset per-step state when entering replan."""
    return {
        # Do NOT reset loop_count (global budget persists)
        # Do NOT reset reflections (history is valuable for debugging)
        # Optionally reset a conceptual "replan_count" if added later
    }
```

This is intentionally minimal. The reflector's `loop_count` keeps incrementing (global budget). The per-step retry counter is implicit in `_reflect_route` (counts from `reflections` list), so it auto-resets when `current_step` changes after replan.

##### Tasks 2.1.5–2.1.6: Progress Renderer (C3)

```python
# src/core/render.py (new file)

"""Progress rendering for rolling plan-execute-reflect loop."""

import sys
import time
from typing import Any


# AI-generated

class ProgressRenderer:
    """Renders single-line progress updates to stderr."""

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._total_steps: int = 0
        self._completed: int = 0
        self._retries: int = 0
        self._replans: int = 0

    def start(self, total_steps: int) -> None:
        """Initialize renderer with expected total steps."""
        self._start_time = time.time()
        self._total_steps = total_steps
        self._completed = 0
        self._retries = 0
        self._replans = 0

    def on_decision(self, step_num: int, action: str, description: str, success: bool | None = None) -> None:
        """Render a progress line after reflector decision."""
        elapsed = int(time.time() - self._start_time)

        icon = {"continue": "✅", "retry": "🔁", "replan": "🔄", "done": "🏁", "interrupt": "⏸️"}
        symbol = icon.get(action, "?")
        status = "" if success is None else ("ok" if success else "FAIL")

        line = (
            f"⠋ Step {step_num}/{self._total_steps}: "
            f"{description:<40} {symbol} {action:<10} {status}"
        )

        # Overwrite previous line using carriage return
        sys.stderr.write(f"\r{line}")
        sys.stderr.flush()

        # Track stats
        if action == "continue":
            self._completed += 1
        elif action == "retry":
            self._retries += 1
        elif action == "replan":
            self._replans += 1

    def finish(self) -> None:
        """Print final summary line."""
        elapsed = int(time.time() - self._start_time)
        summary = (
            f"\nDone: {self._completed} steps, "
            f"{self._retries} retries, "
            f"{self._replans} replans, "
            f"{elapsed}s elapsed\n"
        )
        sys.stderr.write(summary)
        sys.stderr.flush()
```

**Integration point** in main loop:
```python
# In main.py's streaming/event-processing section, after each reflector cycle:
renderer.on_decision(
    step_num=current_step + 1,
    action=state["next_action"],
    description=plan[current_step].get("description", ""),
    success=last_result.get("success"),
)
```

**Acceptance criteria**:
- [ ] Output goes to stderr (not stdout, which is reserved for final answer)
- [ ] Each line overwrites previous (`\r` prefix)
- [ ] Final `finish()` prints newline + summary stats
- [ ] Works correctly when piped (non-TTY stderr still shows output)
- [ ] File size ≤ 100 lines

##### Task 2.1.7: Interrupt Handler Skeleton (C5)

Phase 2 implements the **mechanism** but keeps it simple:

```python
# In main.py or a new src/core/interrupt.py

def handle_interrupt(state: dict) -> str | None:
    """If reflector emitted interrupt, prompt user for input.

    Returns user's text response, or None if human_confirm is off.
    """
    config = state.get("_config")  # injected config
    if not getattr(config, "human_confirm", False):
        return None  # Fall through to done (same as Phase 1)

    print("\n⚠ Agent requests confirmation:", file=sys.stderr)
    print(f"  Reason: {state['reflections'][-1]['reason']}", file=sys.stderr)
    try:
        response = input("Continue? [y/N/edit] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "n"

    if response.startswith("y"):
        return "continue"  # Resume with current plan
    elif response.startswith("edit"):
        return input("New instruction: ")  # Feed to planner as override
    else:
        return "stop"  # Terminate to output
```

**Acceptance criteria**:
- [ ] When `human_confirm=False`: returns None immediately (zero overhead)
- [ ] When `human_confirm=True`: blocks for user input
- [ ] Supports y (continue), n/stop (terminate), edit (new instruction)
- [ ] Ctrl+C / EOF handled gracefully (treated as "stop")

---

#### 2.3 Test Plan

| Test Case | File | Validates |
|-----------|------|-----------|
| `test_replan_includes_completed_steps` | test_planner.py | Replan prompt lists finished steps |
| `test_replan_includes_latest_reflection` | test_planner.py | Reflection reason appears in replan context |
| `test_merge_preserves_completed` | test_planner.py | Completed steps identical before/after merge |
| `test_merge_renumbers_ids` | test_planner.py | IDs are 1..N sequential after merge |
| `test_merge_empty_remaining_ends_plan` | test_planner.py | Empty new_remaining → shorter plan |
| `test_renderer_overwrites_line` | test_render.py | Two writes produce one line (not two) |
| `test_renderer_finish_summary` | test_render.py | Summary contains correct counts |
| `test_interrupt_disabled_no_block` | test_main.py | human_confirm=False → immediate return |
| `test_interrupt_yes_continues` | test_main.py | Input "y" → "continue" |
| `test_interrupt_edit_returns_text` | test_main.py | Input "edit" then "foo" → "foo" |
| `test_e2e_replan_cycle` | test_graph.py | Full loop: exec→fail→reflect(replan)→planner→exec→success |

---

#### 2.4 Verification

```bash
make lint && make complexity && make test-cov

# Specific:
python -m pytest tests/test_planner.py -v -k replan
python -m pytest tests/test_render.py -v
python -m pytest tests/test_graph.py -v -k replan

# Smoke: trigger a real replan scenario
python -m src.main 'edit the non-existent file /tmp/no_such_file_XYZ123.txt and add a comment'
# Expected: executor fails (file not found) → reflector → replan → planner adjusts → output with explanation
```

---

#### 2.5 Risks

| Risk | Mitigation |
|------|-----------|
| Replan creates worse plan than original (quality regression) | Include completed successful steps as positive examples in replan prompt |
| Progress renderer flickers on slow terminals | Add configurable refresh rate; skip intermediate renders if < 100ms apart |
| Interrupt blocking hangs in non-interactive environments (CI, pipes) | Auto-detect TTY; fall back to `human_confirm=False` when stdin is not a terminal |
| Plan merge creates duplicate steps | Deduplicate by step description in merge logic |

---

### Phase 3: Smart Termination & Safety

**Goal**: Make termination intelligent (not just "plan exhausted"). Add hard safety mechanisms. Connect cost budget and human confirmation.

**Scope**: ~350 lines new code, ~70 lines modified. 2 PRs.

---

#### 3.1 Task Breakdown

| # | Task | File(s) | Lines | Depends On |
|---|------|---------|-------|------------|
| 3.1.1 | Completion verifier module | [`src/nodes/verifier.py`](src/nodes/verifier.py) (new) | ~80 | Phase 2 |
| 3.1.2 | Integrate verifier into reflector | [`src/nodes/reflector.py`](src/nodes/reflector.py) | +20 | 3.1.1 |
| 3.1.3 | Stuck detection heuristics (C2 T3) | [`src/core/graph.py`](src/core/graph.py) | +30 | Phase 1 |
| 3.1.4 | Token tracker in LLMClient | [`src/llm/client.py`](src/llm/client.py) | +20 | — |
| 3.1.5 | Budget checker in edge function | [`src/core/graph.py`](src/core/graph.py) | +15 | 3.1.4 |
| 3.1.6 | Per-node model selection (C4) | [`src/llm/client.py`](src/llm/client.py) + nodes | +35 | 3.1.4 |
| 3.1.7 | Full human_confirm integration (C1) | [`src/main.py`](src/main.py) + [`src/core/interrupt.py`](src/core/interrupt.py) | +30 | Phase 2 (skeleton) |
| 3.1.8 | Quality scoring in reflector | [`src/nodes/reflector.py`](src/nodes/reflector.py) | +25 | 3.1.1 |
| 3.1.9 | Tests: verifier | [`tests/test_verifier.py`](tests/test_verifier.py) (new) | ~60 | 3.1.1 |
| 3.1.10 | Tests: stuck detection | [`tests/test_graph.py`](tests/test_graph.py) | +30 | 3.1.3 |
| 3.1.11 | Tests: budget enforcement | [`tests/test_graph.py`](tests/test_graph.py) | +25 | 3.1.4–3.1.5 |
| 3.1.12 | E2E: safety boundaries | [`tests/test_graph.py`](tests/test_graph.py) | +40 | 3.1.3–3.1.5 |

---

#### 3.2 Key Design Decisions

##### Completion Verifier (3.1.1–3.1.2)

The reflector's `done` decision should ideally be grounded in evidence, not just LLM intuition. The verifier provides optional checks:

```python
class VerificationCheck:
    """A single verifiable condition."""

    name: str              # e.g., "file_exists", "tests_pass", "no_syntax_errors"
    description: str       # Human-readable
    check_fn: Callable[[dict], bool]  # Takes state, returns True if passed

# Built-in checks:
VERIFICATION_CHECKS = {
    "file_exists": VerificationCheck(
        name="file_exists",
        description="Target file exists on disk",
        check_fn=lambda s: Path(s.get("target_file", "")).exists(),
    ),
    "tests_pass": VerificationCheck(
        name="tests_pass",
        description="Project tests pass",
        check_fn=lambda s: _run_tests_and_check(s),  # invokes make test internally
    ),
    "syntax_ok": VerificationCheck(
        name="syntax_ok",
        description="Modified files have no syntax errors",
        check_fn=lambda s: _check_syntax(s.get("modified_files", [])),
    ),
}
```

**Integration approach**: The planner annotates each step with optional `verify: list[str]`. The reflector, when considering `done`, runs applicable checks and includes results in its reasoning prompt. This keeps verification **advisory** (the LLM can still override) rather than a hard gate (which would cause false negatives).

**Important**: Verification checks must be fast (< 2s each). The syntax check uses `py_compile`; the test check uses `pytest --tb=no -q`. File existence is O(1).

##### Stuck Detection (C2 Tier 3) (Task 3.1.3)

Implemented in `_reflect_route`, **before** the LLM call:

```python
def _detect_stuck(reflections: list[dict], window: int = 3) -> bool:
    """Detect if the agent is repeating the same failure pattern."""
    if len(reflections) < window:
        return False

    recent = reflections[-window:]
    # Check: same verdict + same step_index in all window entries?
    verdicts = {r["verdict"] for r in recent}
    step_indices = {r["step_index"] for r in recent}

    if verdicts == {"retry"} and len(step_indices) == 1:
        return True  # Same step retried N times in a row

    # Check: oscillating between two states?
    if len(verdicts) == 2 and recent[0]["verdict"] == recent[2]["verdict"]:
        # Pattern: A → B → A (oscillation)
        reasons = [r["reason"][:50] for r in recent]
        if len(set(reasons)) <= 2:
            return True  # Similar reasons, oscillating pattern

    return False
```

When stuck is detected, the edge function returns `"done"` **without calling the reflector LLM**, saving tokens on a clearly degenerate case.

##### Token Budget & Model Selection (C4) (Tasks 3.1.4–3.1.6)

```python
# In LLMClient, extend invoke() to track usage:
def invoke(self, *, schema=None, model_override=None, **kwargs) -> Any:
    model = model_override or self.model
    response = self._model_call(model=model, **kwargs)
    # Extract token count (OpenAI/DeepSeek compatible)
    tokens_used = getattr(response, "usage", None)
    if tokens_used:
        self._total_tokens += tokens_used.total_tokens
    return response
```

Per-node model selection via config:

```python
# In AgentConfig (extend):
class AgentConfig:
    # ... existing ...
    reflector_model: str | None = None     # None = use default model
    planner_model: str | None = None       # None = use default model
    cost_budget: float = 0.0              # 0 = unlimited

# In graph wiring, inject model overrides:
def _make_reflector_node(llm, config):
    def reflector_node(state):
        model = config.reflector_model or llm.model
        # ... use model for this call only
```

Budget checking in edge function (after Tier 1 and Tier 2, before Tier 3):

```python
# In _reflect_route, add between Tier 2 and default:
budget = state.get("cost_budget", 0.0)
used = state.get("total_tokens_used", 0)
if budget > 0 and used >= budget:
    logger.info("Token budget exhausted (%.0f/%.0f)", used, budget)
    return "done"
```

##### Human Confirmation (C1 Full) (Task 3.1.7)

Upgrade the Phase 2 interrupt handler to integrate with the graph loop:

- Dangerous tool patterns detected (delete, `rm -rf`, `.env` modification) → reflector emits `interrupt` with severity=critical
- CLI layer catches interrupt, prompts user
- User's response feeds back into the loop (either continue, modify instruction, or stop)
- This requires the main event loop to support **async interruption**, which means either:
  - Running the graph in `async` mode with checkpoint resumption, or
  - Using LangGraph's `interrupt()` primitive (if available in the installed version)

**Recommendation**: Use LangGraph's built-in `interrupt()` if available (LangGraph ≥ 0.2.22). Otherwise implement a synchronous pause-resume mechanism using checkpoints.

---

#### 3.3 Test Plan

| Test Case | Validates |
|-----------|-----------|
| `test_verifier_file_exists` | Detects missing target file |
| `test_verifier_syntax_error` | Catches Python syntax issues |
| `test_verifier_tests_fail` | Detects test failures |
| `test_stuck_same_retry_pattern` | 3× same-step retry → stuck detected |
| `test_stuck_oscillation` | A→B→A pattern → stuck detected |
| `test_stuck_not_triggered_normal_progress` | Normal mixed verdicts → not stuck |
| `test_budget_exhaustion_forces_done` | Used >= budget → done |
| `test_token_counter_accumulates` | Multiple calls → total increases |
| `test_per_node_model_override` | reflector uses different model when configured |
| `test_human_confirm_dangerous_delete` | Delete operation → interrupt → user prompted |
| `test_human_confirm_safe_operation` | Read operation → no interrupt |

---

#### 3.4 Verification

```bash
make lint && make complexity && make test-cov

# Safety-specific:
python -m pytest tests/ -v -k "stuck or budget or verify or interrupt"

# Manual: force budget exhaustion
python -m src.main 'list all python files' --cost-budget 100
# Expected: should terminate early with budget message

# Manual: trigger stuck detection
# (Construct a test case where a file edit repeatedly fails the same way)
```

---

#### 3.5 Risks

| Risk | Mitigation |
|------|-----------|
| Verification checks are slow (especially test suite) | Run checks asynchronously; timeout after 5s; treat timeout as "inconclusive" (don't block done) |
| Stuck detection has false positives (normal back-and-forth looks like oscillation) | Require window ≥ 3; require reason similarity > 80%; tune thresholds based on metrics (Phase 4) |
| Token counting differs across API providers (some don't return usage) | Handle missing `usage` gracefully (count=0); estimate from input size as fallback |
| Per-node model selection adds config complexity | Sensible defaults (None = use primary model); document in AGENTS.md |

---

### Phase 4: Observability & Tuning

**Goal**: Production readiness. Persistent state, metrics collection, and exploration of advanced features.

**Scope**: ~250 lines new code, ~40 lines modified. Multiple small PRs.

---

#### 4.1 Task Breakdown

| # | Task | File(s) | Lines | Depends On |
|---|------|---------|-------|------------|
| 4.1.1 | Structured debug logging for decisions | [`src/nodes/reflector.py`](src/nodes/reflector.py) + [`src/nodes/planner.py`](src/nodes/planner.py) | +20 | Phase 3 |
| 4.1.2 | Metrics collector class | [`src/core/metrics.py`](src/core/metrics.py) (new) | ~60 | Phase 3 |
| 4.1.3 | Metrics hooks in graph edges | [`src/core/graph.py`](src/core/graph.py) | +15 | 4.1.2 |
| 4.1.4 | Metrics summary on completion | [`src/nodes/output.py`](src/nodes/output.py) | +15 | 4.1.2 |
| 4.1.5 | SqliteSaver migration (C6) | [`src/core/graph.py`](src/core/graph.py) | +10 | Phase 1 (state already serializable) |
| 4.1.6 | Checkpoint restore CLI command | [`src/main.py`](src/main.py) | +25 | 4.1.5 |
| 4.1.7 | Parallel execution feasibility study (C8) | [`docs/adr/0003-parallel-steps.md`](docs/adr/0003-parallel-steps.md) (new ADR) | — | Phase 3 |
| 4.1.8 | Tuning guide from collected metrics | [`docs/tuning.md`](docs/tuning.md) | — | 4.1.2–4.1.4 |

---

#### 4.2 Key Design Decisions

##### Structured Logging (4.1.1)

Every reflector decision and planner output gets logged at DEBUG level as a single JSON line:

```json
{"event": "reflect_decision", "step": 3, "verdict": "retry", "reason": "ImportError: os not imported", "severity": "warning", "loop": 7, "tokens": 342, "ts_ms": 1705791234567}
```

This enables post-hoc analysis with `jq`:

```bash
# Find all retries
cat agent.log | jq 'select(.event=="reflect_decision" and .verdict=="retry")'

# Average tokens per reflector call
cat agent.log | jq -s '[.[] | select(.event=="reflect_decision") | .tokens] | add / length'

# Replan frequency
cat agent.log | jq -s '[.[] | select(.event=="reflect_decision" and .verdict=="replan")] | length'
```

##### Metrics Collector (4.1.2)

```python
@dataclasses.dataclass
class SessionMetrics:
    """Accumulated metrics for a single agent session."""

    total_loops: int = 0
    total_retries: int = 0
    total_replans: int = 0
    total_reflect_calls: int = 0
    total_planner_calls: int = 0
    total_executor_calls: int = 0
    total_tokens_used: int = 0
    total_duration_seconds: float = 0.0
    unique_errors: set[str] = dataclasses.field(default_factory=set)
    verdict_distribution: dict[str, int] = dataclasses.field(default_factory=dict)

    def record_reflection(self, decision: ReflectDecision, tokens: int) -> None:
        self.total_reflect_calls += 1
        self.total_tokens_used += tokens
        self.verdict_distribution[decision.verdict] = \
            self.verdict_distribution.get(decision.verdict, 0) + 1
        if decision.verdict == "retry":
            self.total_retries += 1
        elif decision.verdict == "replan":
            self.total_replans += 1
        self.total_loops += 1
```

Metrics are included in the final output summary so users can see session statistics.

##### SqliteSaver Migration (4.1.5–4.1.6)

Because all state fields were designed to be JSON-serializable in Phase 1, migration is a single-line change:

```python
# BEFORE:
checkpointer = MemorySaver()

# AFTER:
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
checkpointer = AsyncSqliteSaver.from_conn_string(".myagent/checkpoints.db")
```

Restore flow:
```bash
# List saved sessions
python -m src.main --list-sessions

# Resume a specific session
python -m src.main --resume <session_id>
```

The restore procedure: load checkpoint → set `next_action="reflector"` (so it re-enters the loop) → the reflector observes the last result and decides whether to continue or declare done.

##### Parallel Execution Study (4.1.7)

This is a research/deliverable task, not a code task. Output is an ADR evaluating:

- DAG-based plan schema design options
- Conflict detection algorithms for concurrent file edits
- Batch executor interface sketch
- Performance benchmarks (serial vs parallel for typical coding tasks)
- Recommendation: proceed or defer

---

#### 4.3 Verification

```bash
make lint && make complexity && make test-cov

# Observability-specific:
python -m src.main 'create a hello world python script' --log-level DEBUG 2> agent.log
jq '.' agent.log | head -20  # Verify structured log format

# Persistence:
python -m src.main 'write a function'  # Run normally
python -m src.main --list-sessions     # Should show the saved session
python -m src.main --resume <id>       # Should resume from checkpoint
```

---

#### 4.4 Risks

| Risk | Mitigation |
|------|-----------|
| SQLite checkpointer adds a dependency | `langgraph-checkpoint-sqlite` is lightweight; falls back to MemorySaver if import fails |
| Log volume grows large for long sessions | Rotate log files; structured format compresses well; DEBUG level can be turned off in production |
| Metrics overhead affects performance | Metrics collection is dict updates only (negligible); JSON serialization only at end-of-session |
| Parallel study concludes "not worth it" | That's a valid outcome; document findings and close C8 as "declined" |

---

### Phase Dependency Graph

```
Phase 1 (Reflect + Route)
    │
    ├──→ Phase 2 (Replan + Display + Interrupt)
    │       │
    │       └──→ Phase 3 (Safety + Budget + Verify)
    │               │
    │               └──→ Phase 4 (Observability + Persist + Explore)
    │
    └──→ (Each phase can branch independently for bug fixes)

Cross-phase constraints:
  - Phase 2 requires Phase 1's reflector node and graph changes
  - Phase 3 requires Phase 2's replan-capable planner
  - Phase 4 requires Phase 3's safety mechanisms (to measure their effectiveness)
  - Phases can be developed in parallel by different developers IF interfaces are stable
```

### Rollback Criteria

If any phase introduces regressions beyond acceptable thresholds:

| Metric | Threshold | Action |
|--------|-----------|--------|
| Test coverage drop | > 3% absolute | Block merge; add tests |
| Latency increase (per step) | > 2× vs baseline | Investigate; consider async LLM calls |
| New test failures | Any | Block merge |
| `make complexity` warnings | Any | Split file or extract helpers |
| LLM cost per task (token count) | > 3× vs baseline | Review reflector prompt size; add truncation
