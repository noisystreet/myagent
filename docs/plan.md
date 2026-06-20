# Programming Agent Phased Implementation Plan

> Based on the [Architecture Design](file:///home/gzz/creativity/myagent/docs/adr/design.md), divided into 5 phases for incremental delivery.

---

## Phase Overview

```
                      Phase 4: HITL
                    /     + Cost Control
                   /           │
Phase 1:    Phase 2:    Phase 3:    │      Phase 5:
Core MVP → Reliability → Memory  → HITL     → Ship
             + Security   Persist   + Cost     + Test
                   \          /    + Control    │
                    \        /                   │
                    Phase 2: Security             │
                     (early)                      │
                                                 ▼
                                         Deliverable Agent
```

| Phase | Name | Duration | Deliverable |
|-------|------|----------|-------------|
| **P1** | Core MVP | Week 1-2 | Agent prototype that can perform simple coding tasks |
| **P2** | Reliability + Security | Week 3-4 | Stable version with error recovery and sandbox isolation |
| **P3** | Memory Persistence | Week 5 | Cross-session memory, project context awareness |
| **P4** | HITL + Cost Control | Week 6 | Production-grade agent with user intervention and cost management |
| **P5** | Test & Ship | Week 7-8 | Benchmarked, documented, deliverable version |
| **P6** | Ecosystem (MCP + Skill) | Week 9-12 | Extensible agent with MCP tool discovery and pluggable skills |

---

## Phase 1: Core MVP (Weeks 1-2)

**Goal**: Complete end-to-end graph flow, capable of "read file → modify file → verify" tasks.

### Task Breakdown

#### 1.1 Project Scaffold (Day 1-2)

- [ ] Initialize Python project structure
- [ ] Configure `pyproject.toml`, declare dependencies
- [ ] Configure `.env` for API key management
- [ ] Create `src/` directory structure

```
myagent/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── state.py          # CodingAgentState
│   │   ├── graph.py          # Graph definition & compilation
│   │   └── config.py         # AgentConfig
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── planner.py
│   │   ├── executor.py
│   │   └── output.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── file_tools.py     # read, write, edit, delete
│   │   └── command_tools.py  # run_command
│   └── llm/
│       ├── __init__.py
│       └── client.py         # LLMClient
├── tests/
├── prompts/
└── pyproject.toml
```

**Dependencies:**

```
langgraph>=0.2.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
pydantic>=2.0
python-dotenv>=1.0
```

#### 1.2 State Definition (Day 2)

- [ ] Implement `ToolResult` data class
- [ ] Implement `CodingAgentState` (basic fields only: `messages`, `plan`, `current_step`, `workspace`, `tool_results`, `errors`, `next_action`)
- [ ] Exclude advanced fields for memory/cost/test

#### 1.3 LLM Client (Day 2-3)

- [ ] Implement `LLMClient` (basic invoke + exponential backoff retry)
- [ ] Implement `PlanSchema` structured output
- [ ] Support GPT-4o and Claude 3.5 (switchable via config)

#### 1.4 Tool Implementation (Day 3-4)

- [ ] `read_file` — support `offset`/`limit` segmented reading
- [ ] `write_file` — write/overwrite
- [ ] `edit_file` — Search/Replace editing
- [ ] `run_command` — blocking execution, timeout control
- [ ] Unified tool contract: `tool_wrapper` decorator

#### 1.5 Node Implementation (Day 4-6)

- [ ] `planner_node` — call LLM to generate `PlanSchema`, set `next_action=executor`
- [ ] `executor_node` — execute current step, set `next_action` based on result (executor / output)
- [ ] `output_node` — aggregate `tool_results`, generate summary
- [ ] `should_continue` routing function (executor ↔ output only)

#### 1.6 Graph Assembly (Day 6-7)

- [ ] Assemble nodes with `StateGraph`
- [ ] Configure `add_conditional_edges`
- [ ] Compile graph

```python
# P1 graph structure (minimal)
START → planner → executor ←→ output → END
                      ↓ (on error)
                   evaluator (pass-through for now)
```

#### 1.7 Demo Verification (Day 7)

- [ ] Implement CLI entry `main.py`: accept user input → invoke graph → display result
- [ ] Test cases: create hello.py, modify file content, find files
- [ ] Verify: at least 3 simple end-to-end tasks pass

### P1 Milestone

> ✅ Agent accepts natural language instructions and auto-completes "read → modify → verify" tasks
> ✅ All tools can be called normally and return structured results
> ✅ CLI interaction works

---

## Phase 2: Reliability + Security Sandbox (Weeks 3-4)

**Goal**: Introduce error recovery and security isolation. Agent evolves from "usable" to "reliable."

### 2.1 Evaluator Node (Day 8-10)

- [ ] Implement `evaluator_node`
  - Read `errors`, call LLM to analyze errors
  - Support three decisions: `retry` (retry directly), `adjust` (adjust step and retry), `replan` (replan)
- [ ] Implement `FixSchema` structured output
- [ ] Implement retry counter + circuit breaker logic

```
P2 graph structure:
START → planner → executor ←→ evaluator ←→ executor
                  ↓                           ↓
               output ←──────────────────── output
```

**Routing table update:**

| Current node | Condition | Next node |
|-------------|-----------|-----------|
| executor | Success & more steps | executor |
| executor | Success & last step | output |
| executor | Failure & retries < limit | evaluator |
| evaluator | retry | executor |
| evaluator | adjust | executor (with modified step) |
| evaluator | replan | planner |
| evaluator | Retries exhausted / unfixable | output (with error report) |

### 2.2 Error Classification (Day 10-11)

- [ ] Parse common error types (syntax errors, runtime exceptions, timeouts, tool parameter errors)
- [ ] Fix strategies for each type
- [ ] `check_circuit_breaker` function

### 2.3 Security Sandbox (Day 12-14)

- [ ] Implement `SecurityManager`
  - Command allowlist (default set)
  - Path escape detection (including symlinks)
  - Sensitive info redaction (API Key, Token regex replacement)
- [ ] Implement `SandboxExecutor`
  - subprocess execution + process group isolation
  - Timeout kills entire process group (`os.killpg`)
  - Output truncation (`max_output=10_000`)
- [ ] Implement `SecurityConfig`
- [ ] Integrate into `executor_node`: validate before tool calls

### 2.4 Prompt Template Management (Day 14)

- [ ] Create `prompts/` directory, write YAML templates for each node
- [ ] Implement `PromptRegistry` (YAML loading + version selection)
- [ ] Implement `PromptTemplate` (`safe_substitute` rendering)

```
prompts/
├── planner.yaml
├── executor.yaml
├── evaluator.yaml
└── output.yaml
```

### P2 Milestone

> ✅ Agent can auto-analyze and fix errors after execution failure (at least 3 error types)
> ✅ Dangerous commands blocked, path escapes detected
> ✅ Sensitive info (API Key) redacted in output
> ✅ Prompt templates loaded from YAML files, versioned

---

## Phase 3: Memory Persistence (Week 5)

**Goal**: Agent remembers project structure and past experiences, provides consistent cross-session service.

### 3.1 Session Memory (Day 15-16)

- [ ] Integrate LangGraph `MemorySaver` checkpointer
- [ ] Link multiple invocations in the same session via `thread_id`
- [ ] Implement `trim_messages` window truncation strategy (keep last N turns)
- [ ] Implement summary compression (history → LLM summary + recent messages)

### 3.2 Long-term Memory (Day 16-18)

- [ ] Implement SQLite `MemoryStore`
  - `project_memory` table: project structure, key files, coding conventions
  - `episodic_memory` table: error pattern → solution
  - `user_preference` table: coding style, test framework preference
- [ ] Implement `ProjectMemory`, `EpisodicMemory`, `UserPreference` data classes
- [ ] Implement CRUD operations for all three stores

### 3.3 LangGraph Store Integration (Day 18-19)

- [ ] Implement `CodingAgentStore` (wraps `MemoryStore`, interfaces with LangGraph `BaseStore`)
- [ ] Inject `store=store` when compiling the graph
- [ ] Integrate memory retrieval in `planner_node`: inject project context + past similar errors

### 3.4 State Extension (Day 19)

- [ ] Add `session_id`, `long_term_memory`, `memory_summary` fields to `CodingAgentState`

### P3 Milestone

> ✅ Multiple calls in the same session retain context (no need for user to repeat)
> ✅ Agent can auto-retrieve and leverage project structure and past experience
> ✅ Messages auto-truncated/compressed when exceeding window size

---

## Phase 4: HITL + Cost Control (Week 6)

**Goal**: Agent can consult user at critical points and auto-optimize costs within budget.

### 4.1 Human-in-the-Loop (Day 20-23)

- [ ] Three interaction modes:
  - **Silent mode** (default): Agent executes autonomously, no interruption
  - **Confirm mode**: Pause for confirmation before risky operations (rm -rf, git push --force, etc.)
  - **Guidance mode**: Ask user when encountering ambiguity
- [ ] Implement `executor_with_human_confirm` — `interrupt()` pause + user choice
- [ ] Implement `planner_with_clarification` — ambiguity detection + option generation
- [ ] Implement `output_with_feedback` — "accept/modify/retry/discard" after output
- [ ] Integrate `SqliteSaver` checkpointer for pause-and-resume support

### 4.2 Cost Tracking (Day 23-24)

- [ ] Implement `CostTracker`
  - Record token consumption and cost per LLM call
  - Model pricing table (gpt-4o, gpt-4o-mini, claude-3.5-sonnet)
  - Budget check: `is_over_budget()`
- [ ] Implement `TokenUsage`, `SessionCost` data classes
- [ ] Integrate cost tracking into `LLMClient` callbacks

### 4.3 Model Selection Strategy (Day 24-25)

- [ ] Implement `ModelSelector`
  - Task complexity heuristic (length/keywords)
  - Auto-degrade to `gpt-4o-mini` when over budget
  - Assign stronger models to complex tasks
- [ ] Select model based on complexity in `planner_node`

### 4.4 Configuration System (Day 25)

- [ ] Extend `AgentConfig`
  - Module toggles: `memory_enabled`, `sandbox_enabled`, `human_confirm`, etc.
  - Budget cap: `cost_budget`
- [ ] Support loading config from YAML file

### P4 Milestone

> ✅ User receives confirmation prompt before risky operations (rm -rf, git force push, etc.)
> ✅ User can interrupt execution and modify instructions
> ✅ Agent auto-switches to cheaper model when budget is exceeded
> ✅ Token consumption and cost report displayed after each execution

---

## Phase 5: Test & Ship (Weeks 7-8)

**Goal**: Establish complete test suite, pass benchmark evaluation, deliver a production-ready agent.

### 5.1 Unit Tests (Day 26-28)

- [ ] `tests/test_security.py` — command allowlist, path escape, sensitive info redaction
- [ ] `tests/test_tools.py` — normal/error paths for each tool
- [ ] `tests/test_state.py` — state field merging, routing decisions
- [ ] `tests/test_prompts.py` — template rendering, version selection

### 5.2 Node/Integration Tests (Day 28-30)

- [ ] `tests/test_planner.py` — mock LLM response, verify plan output
- [ ] `tests/test_executor.py` — tool calls + routing logic
- [ ] `tests/test_evaluator.py` — error classification + fix strategies
- [ ] `tests/test_graph.py` — full graph execution, verify node flow

### 5.3 E2E Evaluation (Day 30-33)

- [ ] Implement `EvalTask`, `EvalResult`, `EvalSuite`
- [ ] Build evaluation dataset (at least 10 tasks):

| Task | Type | Difficulty |
|------|------|------------|
| Create hello world | Generation | Easy |
| Fix syntax error | Fix | Easy |
| Add function docstring | Edit | Easy |
| Refactor function to class | Refactor | Medium |
| Create Flask API from scratch | Generation | Medium |
| Fix runtime exception | Debug | Medium |
| Code review + fix | Review | Medium |
| Cross-file rename | Refactor | Medium |
| Add unit tests | Test | Hard |
| Performance bottleneck analysis | Analysis | Hard |

- [ ] Run benchmark, record completion rate, avg steps, avg cost

### 5.4 Observability (Day 33-34)

- [ ] Implement `AgentLogger` callbacks
- [ ] Integrate LangSmith tracing (optional)
- [ ] Record `node_timings` in State
- [ ] Output execution report after completion

### 5.5 Documentation & Engineering (Day 34-36)

- [ ] README.md (install, usage, configuration guide)
- [ ] Polish CLI (argparse options: `--workspace`, `--model`, `--budget`, etc.)
- [ ] Optimize error messages (user-friendly error output)
- [ ] Example scripts (`examples/` directory)

### P5 Milestone

> ✅ Tests cover core modules (security, tools, state, routing)
> ✅ E2E benchmark completion rate ≥ 80%
> ✅ CLI tool installable and usable directly
> ✅ Complete documentation with working examples

---

## Phase 6: Ecosystem (MCP + Skill) (Weeks 9-12)

**Goal**: Make the agent extensible — dynamically discover and call MCP tools, and support pluggable Skills for domain-specific capabilities.

### Background

- **MCP (Model Context Protocol)** is an open protocol by Anthropic that standardizes how LLM applications discover and call external tools. Any MCP Server can be dynamically plugged in.
- **Skill** is a self-contained capability package (prompt + tools + knowledge) that the agent can load on demand for domain-specific scenarios.

### 6.1 MCP Client Core (Day 43-46)

- [ ] Implement `mcp/client.py` — MCP protocol client supporting stdio transport
  - `connect()` — spawn MCP Server subprocess, establish JSON-RPC connection
  - `list_tools()` — discover server capabilities
  - `call_tool()` — invoke a tool and return results
  - `close()` — gracefully shut down the connection
- [ ] Implement `mcp/transport.py` — transport abstraction layer
  - `StdioTransport` — subprocess-based for local MCP Servers
  - `SSETransport` — HTTP SSE-based for remote MCP Servers (future)
- [ ] Implement `mcp/schema.py` — MCP protocol types (JSON-RPC message, tool definition, etc.)
- [ ] Unit tests for MCP client with a mock server

### 6.2 MCP Registry & Integration (Day 47-50)

- [ ] Implement `mcp/registry.py` — manage multiple MCP Server connections
  - `discover_all(configs)` — connect all configured servers, collect tools
  - `get_dynamic_tools()` — return all MCP tools as `ToolDef` list
- [ ] Implement user configuration file `mcp_servers.yaml`
  - stdio mode: command + args to spawn local servers
  - sse mode: URL + headers for remote servers (future)
- [ ] Integrate into `executor_node`: dynamic tools join the available tool pool alongside built-in tools
- [ ] Add `mcp_tool_call()` to `tools/` — call MCP tools via registry
- [ ] Add config to `AgentConfig`: `mcp_servers_path`

### 6.3 Skill System Core (Day 51-54)

- [ ] Define `skills/<name>/skill.yaml` format
  - Metadata: name, version, description, triggers
  - Tools: builtin tools to enable, MCP servers to connect
  - Prompt: reference to `prompt.md`
  - Knowledge: list of knowledge documents
  - Config: skill-specific parameters
- [ ] Implement `skills/registry.py`
  - `scan(path)` — discover all skills in a directory
  - `match(input)` — match user input against skill triggers
  - `load(name)` — load a skill's full definition
- [ ] Implement `skills/loader.py`
  - Parse `skill.yaml` frontmatter
  - Load `prompt.md` content
  - Load knowledge documents into context
- [ ] Implement `nodes/skill_router.py` — route matched input to skill executor
  - Two paths: skill match → skill_executor, no match → planner
- [ ] Add config to `AgentConfig`: `skills_dir`

```
Agent flow with skills:

User Input → intent_router
                │
                ├── chat → chat_node
                │
                └── task → skill_router (new)
                              │
                              ├── skill match → skill_executor (new)
                              │                    └── MCP tools available
                              │
                              └── no match → planner → executor
                                                          ├── file_tools
                                                          ├── command_tools
                                                          └── mcp_tools (new)
```

### 6.4 Skill-MCP Integration (Day 55-57)

- [ ] Skills can declare MCP server dependencies in `skill.yaml`
  - On skill load, auto-connect required MCP Servers
  - On skill unload, disconnect associated MCP Servers
- [ ] Implement skill lifecycle hooks
  - `on_load(skill, context)` — setup
  - `on_unload(skill, context)` — cleanup
- [ ] Skill context injection: loaded knowledge docs appended to system prompt

### 6.5 Example Skills & Documentation (Day 58-60)

- [ ] Build `skills/code-reviewer/` — code review skill
  - Builtin tools: read_file, grep, glob
  - Knowledge: Python best practices, security patterns
- [ ] Build `skills/web-developer/` — web development skill
  - Builtin tools: read_file, write_file, run_command
  - Knowledge: React patterns, CSS guide, accessibility checklist
- [ ] Update README with MCP + Skill usage documentation
- [ ] Add example `mcp_servers.yaml` to `.env.example`

### P6 Milestone

> ✅ Agent can discover and call MCP tools from any MCP Server
> ✅ User configures MCP Servers via a single YAML file
> ✅ Skills auto-load on intent match, bringing domain-specific prompt + tools
> ✅ At least 2 example skills shipped

---

## Delivery Roadmap

```
Week 1     Week 2     Week 3     Week 4     Week 5     Week 6     Week 7     Week 8     Week 9-12
├─P1: Core MVP───────┤
│ Scaffold State LLM  │
│ Tools  Nodes Graph  │
│           Demo Verify│
                      ├─P2: Reliability+Security────────┤
                      │ Evaluator Error Classification   │
                      │ Security   Sandbox               │
                      │ Prompt Template Management       │
                                                       ├─P3: Memory──┤
                                                       │ Session Long │
                                                       │ Store Integ. │
                                                                  ├─P4: HITL+Cost───┤
                                                                  │ HITL Cost Track │
                                                                  │ Model Select    │
                                                                  │ Config          │
                                                                                 ├─P5: Test+Ship───┤
                                                                                 │ Unit Integ E2E │
                                                                                 │ Observability  │
                                                                                 │ Docs           │
                                                                                                ├─P6: MCP+Skill────
                                                                                                │ MCP Client Core  │
                                                                                                │ MCP Registry     │
                                                                                                │ Skill System     │
                                                                                                │ Integration      │
                                                                                                │ Example Skills   │
```

---

## Key Dependencies

```
P1 ─── required ───→ P2 ─── required ───→ P3
                                              │
                                              ├──→ P4 (depends on P2, P3 optional)
                                              │
                                              ├──→ P5 (depends on P1-P4 complete)
                                              │
                                              └──→ P6 (depends on P1-P2, P3-P5 optional)
```

- **P1** is the prerequisite for all phases
- **P2**'s Evaluator depends on P1's Executor; Security is independent and can run in parallel with P3
- **P3** requires P2's Prompt template management as a base for memory retrieval integration
- **P4**'s HITL depends on P2's security validation (confirm mode needs to recognize dangerous operations)
- **P5** testing is based on P1-P4's complete codebase
- **P6** MCP core depends on P2's Security (sandbox for running MCP servers); Skill depends on P1's executor and tool model

---

## Risk Management

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| LLM API instability (rate limit/timeout) | P1 delay | Medium | LLMClient implements exponential backoff + fallback model |
| LangGraph API changes | All phases | Low | Pin version `langgraph>=0.2.0,<0.4` |
| High token consumption on complex tasks | P4 | Medium | Window truncation + model degradation, set budget cap |
| Security sandbox blocks legitimate commands | P2 | Medium | Configurable allowlist, clear reason on block |
| User requirement ambiguity leads to wrong actions | P4 | High | Guidance mode proactively asks when uncertain |
| Benchmark metrics below target (<80% completion) | P5 | Medium | Iterate on Evaluator and Prompt optimization |
| MCP Server instability (crash, hang) | P6 | Medium | Health check + auto-restart + timeout per call |
| Skill trigger false positives | P6 | Medium | Multi-factor matching (pattern + file type + user confirm) |

---

## Delivery Checklist

### P1 Completion Check
- [ ] `python main.py "Create a hello.py"` works
- [ ] Agent can read file content and modify it correctly
- [ ] All tools return `ToolResult` format
- [ ] Graph execution completes within 30 seconds

### P2 Completion Check
- [ ] `python main.py "Fix the syntax error in buggy.py"` auto-fixes
- [ ] `rm -rf /` intercepted by security sandbox
- [ ] `cat /etc/passwd` blocked by path security
- [ ] API Key redacted to `[REDACTED]` in output

### P3 Completion Check
- [ ] Second call with same `thread_id` sees results from the first
- [ ] Error fix solutions recorded, reusable for similar errors
- [ ] Long conversation auto-truncates after 10+ turns

### P4 Completion Check
- [ ] Confirmation prompt appears before `rm -rf` operations
- [ ] Auto-uses `gpt-4o-mini` when budget set to $0.01
- [ ] Cost report displayed after each execution

### P5 Completion Check
- [ ] `pytest tests/` all pass
- [ ] E2E benchmark completion rate ≥ 80%
- [ ] `pip install .` installable, `myagent --help` works
- [ ] README includes quick-start example

### P6 Completion Check
- [ ] MCP client connects to stdio-based MCP Server and calls tools
- [ ] Tools from MCP Server appear in executor's tool pool
- [ ] Skills auto-match on user input and load prompt + tools
- [ ] Skill can declare MCP dependencies and connect on load
- [ ] At least 2 example skills (`code-reviewer`, `web-developer`) are functional
- [ ] `mcp_servers.yaml` configuration documented and working
