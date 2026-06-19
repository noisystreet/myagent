# Programming Agent Architecture Design

## 1. Overview

This document describes the overall architecture design of a general-purpose programming agent built with **LangGraph**. The agent can understand user requirements and automate coding, debugging, refactoring, and other software engineering tasks.

### 1.1 Design Goals

- **Orchestrated**: Task steps are clear and traceable, with support for interruption and recovery
- **Fault-tolerant**: Comprehensive error detection and self-healing capabilities
- **Extensible**: Easy to add new tools, nodes, and routing strategies
- **Observable**: Transparent execution with inspectable state

### 1.2 Tech Stack

| Component | Choice |
|-----------|--------|
| Graph framework | LangGraph ≥0.2.0 |
| LLM | GPT-4o / Claude 3.5+ (with Tool Calling support) |
| Runtime | Python ≥3.11 |
| Tool sandbox | Local subprocess / Docker |

---

## 2. Overall Architecture

### 2.1 Graph Structure

```
                         ┌─────────────┐
                         │  __start__  │
                         └──────┬──────┘
                                │
                                ▼
                         ┌─────────────┐
                         │   planner   │  ← analyze request, generate plan
                         └──────┬──────┘
                                │
                                ▼
                   ┌─────────────────────┐
                   │   should_continue   │  ← conditional routing
                   └──┬──────┬──────┬───┘
                      │      │      │
            ┌─────────┘      │      └──────────┐
            ▼                 ▼                  ▼
     ┌──────────┐    ┌─────────────┐    ┌──────────────┐
     │ executor │    │  evaluator  │    │   output     │
     │ (loop)   │    │  (fix)      │    │  (summary)   │
     └────┬─────┘    └──────┬──────┘    └──────┬───────┘
          │                 │                  │
          └─────────────────┘                  │
            (return to should_continue)         │
                                                ▼
                                         ┌─────────────┐
                                         │  __end__    │
                                         └─────────────┘
```

### 2.2 Node Responsibilities

| Node | Responsibility | Input | Output |
|------|---------------|-------|--------|
| `planner` | Parse request, decompose into ordered steps | User message | `plan: List[str]` |
| `executor` | Execute current step via tool calls | `plan[current_step]` | Tool execution results |
| `evaluator` | Check errors, decide fix/retry/replan | `errors`, `tool_results` | Modified plan or retry flag |
| `output` | Summarize results, generate final reply | `plan`, `tool_results` | Final response |

---

## 3. State Design

### 3.1 State Definition

```python
from typing import TypedDict, Annotated, List, Optional, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class ToolResult:
    """Structured representation of a tool execution result."""
    tool_name: str
    success: bool
    data: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0

class CodingAgentState(TypedDict):
    # --- Message layer (auto-merged by LangGraph) ---
    messages: Annotated[List[AnyMessage], add_messages]

    # --- Task plan ---
    plan: List[str]                    # Step list, e.g. ["read src/main.py", "fix bug in ..."]
    current_step: int                  # Current step index (0-based)
    max_steps: int                     # Max execution steps (circuit breaker)

    # --- Context ---
    workspace: str                     # Working directory path
    file_context: dict                 # {filepath: content_hash}, read file cache
    relevant_files: List[str]          # Files involved in the task

    # --- Execution tracking ---
    tool_results: List[ToolResult]     # History of all tool calls
    errors: List[str]                  # Accumulated error messages
    retry_count: int                   # Retry count for current step
    step_attempts: int                 # Total execution attempts

    # --- Control flow ---
    next_action: Literal[              # Conditional routing signal
        "planner", "executor", "evaluator", "output", "end"
    ]

    # --- Output ---
    final_output: Optional[str]        # Final response text
```

### 3.2 Design Principles

1. **Minimalism**: Only include data that needs to be shared between nodes, avoid state bloat
2. **Explicit routing**: The `next_action` field makes conditional edge logic transparent and debuggable
3. **Messages as logs**: `messages` serves not only LLM conversation but also as an execution log
4. **Circuit breaker**: `step_attempts` and `max_steps` prevent infinite loops

---

## 4. Memory Module Design

Programming agent memory is divided into three levels, ordered by lifetime from shortest to longest:

```
┌─────────────────────────────────────────────────┐
│                  Memory Levels                    │
├─────────────┬──────────────────┬─────────────────┤
│  Working    │   Session        │   Long-term     │
│  Memory     │   Memory         │   Memory        │
├─────────────┼──────────────────┼─────────────────┤
│ Current     │ Within same      │ Cross-session   │
│ task scope  │ session message  │ persistent      │
│             │ history          │                 │
├─────────────┼──────────────────┼─────────────────┤
│ State fields│ messages list    │ External store   │
│ (plans,     │ (LangGraph       │ (SQLite/         │
│ file_cache) │  auto-managed)   │  JSON file)      │
└─────────────┴──────────────────┴─────────────────┘
```

### 4.1 Working Memory

Working memory is carried by runtime fields in State, with a lifetime limited to **the current graph execution**.

```python
class WorkingMemory:
    """Current execution context — non-message fields in State."""
    plan: List[str]                # Current task execution plan
    current_step: int              # Execution progress
    file_context: dict             # {path: content_snippet} read file summaries
    relevant_files: List[str]      # Files involved in the task
    tool_results: List[ToolResult] # All tool results in this execution
    errors: List[str]              # Errors pending handling
```

**How it works:**
- LangGraph automatically passes and returns State on each node invocation
- Nodes read/write working memory through State
- Released after graph execution completes

### 4.2 Session Memory

Session memory is managed by the `messages` field, with a lifetime of **one complete session** (spanning multiple graph executions).

```python
from langgraph.checkpoint.memory import MemorySaver

# Use in-memory checkpoint to retain state after graph execution
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Multiple calls in the same session are linked by thread_id
config = {"configurable": {"thread_id": "session-001"}}
result1 = graph.invoke({"messages": [user_msg1]}, config)
result2 = graph.invoke({"messages": [user_msg2]}, config)
# messages accumulate automatically; LLM sees the full conversation history
```

**Message management strategies:**

| Strategy | Description | When to use |
|----------|-------------|-------------|
| Full retention | Keep all messages | Short sessions, debug mode |
| Window truncation | Keep only last N turns | Long sessions, token saving |
| Summary compression | History → summary + recent messages | Token-sensitive scenarios |

```python
def trim_messages(state: CodingAgentState, max_turns: int = 10) -> dict:
    """Window truncation: keep only the last N turns of conversation."""
    messages = state["messages"]
    if len(messages) > max_turns * 2:
        # Keep system prompt + most recent max_turns turns
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        recent = messages[-(max_turns * 2):]
        return {"messages": system_msgs + recent}
    return {}
```

### 4.3 Long-term Memory

Long-term memory persists across sessions and stores three types of information:

#### 4.3.1 Storage Structure

```python
from dataclasses import dataclass, field, asdict
from typing import Any
import json, time, sqlite3
from pathlib import Path

@dataclass
class ProjectMemory:
    """Project-level memory: project structure, key files, architectural decisions."""
    workspace: str
    project_name: str
    file_index: dict[str, FileInfo]       # File path → metadata
    key_files: list[str]                  # Core file list
    architecture_notes: list[str]         # Architecture decision records
    conventions: dict[str, str]           # Coding conventions, e.g. {"import": "absolute"}
    last_updated: float = 0.0

@dataclass
class EpisodicMemory:
    """Episodic memory: problems encountered and their solutions."""
    id: str
    pattern: str                          # Error pattern identifier
    context: str                          # Error context
    error: str                            # Error message
    solution: str                         # Solution
    success: bool                         # Whether the solution was effective
    timestamp: float = 0.0

@dataclass
class UserPreference:
    """User preference memory."""
    preferred_model: str = "gpt-4o"
    coding_style: str = "pythonic"        # pythonic / typed / concise
    test_preference: str = "pytest"       # Test framework preference
    verbose: bool = False
    custom_rules: list[str] = field(default_factory=list)
```

#### 4.3.2 Storage Backend

```python
class MemoryStore:
    """Long-term memory store, default SQLite backend."""
    
    def __init__(self, db_path: str = "~/.coding_agent/memory.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS project_memory (
                    workspace TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id TEXT PRIMARY KEY,
                    pattern TEXT NOT NULL,
                    context TEXT,
                    error TEXT NOT NULL,
                    solution TEXT NOT NULL,
                    success INTEGER NOT NULL DEFAULT 1,
                    timestamp REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_preference (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_episodic_pattern 
                    ON episodic_memory(pattern);
            """)
    
    # --- Project memory ---
    def save_project_memory(self, mem: ProjectMemory):
        data = json.dumps(asdict(mem))
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO project_memory VALUES (?, ?, ?)",
                (mem.workspace, data, time.time())
            )
    
    def load_project_memory(self, workspace: str) -> Optional[ProjectMemory]:
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT data FROM project_memory WHERE workspace = ?",
                (workspace,)
            ).fetchone()
        return ProjectMemory(**json.loads(row[0])) if row else None
    
    # --- Episodic memory ---
    def save_episode(self, ep: EpisodicMemory):
        ep.timestamp = time.time()
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO episodic_memory 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (ep.id, ep.pattern, ep.context, ep.error, 
                 ep.solution, int(ep.success), ep.timestamp)
            )
    
    def search_episodes(self, pattern: str, limit: int = 5) -> List[EpisodicMemory]:
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                """SELECT * FROM episodic_memory 
                   WHERE pattern LIKE ? OR error LIKE ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (f"%{pattern}%", f"%{pattern}%", limit)
            ).fetchall()
        return [
            EpisodicMemory(id=r[0], pattern=r[1], context=r[2],
                          error=r[3], solution=r[4], success=bool(r[5]),
                          timestamp=r[6])
            for r in rows
        ]
    
    # --- Preference memory ---
    def save_preference(self, pref: UserPreference):
        with sqlite3.connect(str(self.db_path)) as conn:
            for key, value in asdict(pref).items():
                conn.execute(
                    "INSERT OR REPLACE INTO user_preference VALUES (?, ?)",
                    (key, json.dumps(value))
                )
    
    def load_preference(self) -> UserPreference:
        pref = UserPreference()
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute("SELECT key, value FROM user_preference").fetchall()
        for key, value in rows:
            if hasattr(pref, key):
                setattr(pref, key, json.loads(value))
        return pref
```

#### 4.3.3 Memory I/O Timing

```
Memory interactions during graph execution:

Planner
  │
  ├── Reads: project memory (project structure, conventions)
  ├── Reads: episodic memory (similar problem solutions)
  ├── Reads: user preferences (coding style, test framework)
  │
  ▼
Executor
  │
  ├── Writes: working memory (tool_results, errors)
  │
  ▼
Evaluator
  │
  ├── Writes: episodic memory (solutions for new errors)
  │
  ▼
Output
  │
  ├── Writes: project memory (update file index)
  ├── Writes: session summary (message compression)
```

### 4.4 Integration with LangGraph Native Mechanisms

```python
# 1. Checkpointer — session-level persistence
checkpointer = MemorySaver()  # or PostgresSaver / SqliteSaver

# 2. Store — LangGraph long-term memory API (v0.3+)
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

class CodingAgentStore(BaseStore):
    """Wraps long-term memory operations, exposing unified get/put/delete interface."""
    
    def __init__(self, memory_store: MemoryStore):
        self._store = memory_store
    
    def get(self, namespace: tuple[str, ...], key: str) -> Optional[dict]:
        ns = "/".join(namespace)
        if ns == "project":
            mem = self._store.load_project_memory(key)
            return asdict(mem) if mem else None
        elif ns == "episode":
            episodes = self._store.search_episodes(key, limit=1)
            return asdict(episodes[0]) if episodes else None
        elif ns == "preference":
            return asdict(self._store.load_preference())
        return None
    
    def put(self, namespace: tuple[str, ...], key: str, value: dict):
        ns = "/".join(namespace)
        if ns == "project":
            self._store.save_project_memory(ProjectMemory(**value))
        elif ns == "episode":
            self._store.save_episode(EpisodicMemory(**value))
        elif ns == "preference":
            self._store.save_preference(UserPreference(**value))

# 3. Inject Store into graph nodes
store = CodingAgentStore(MemoryStore())
graph = builder.compile(checkpointer=checkpointer, store=store)
```

### 4.5 Memory-Augmented Generation

Inject long-term memory into the Prompt in planner and evaluator nodes:

```python
def planner_with_memory(state: CodingAgentState, store: CodingAgentStore) -> dict:
    """Planner node augmented with long-term memory."""
    
    workspace = state["workspace"]
    
    # Retrieve relevant memories
    project_mem = store.get(("project",), workspace)
    recent_episodes = store.search_episodes(
        state["messages"][-1].content[:50], limit=3
    )
    preferences = store.get(("preference",), "user")
    
    memory_context = ""
    if project_mem:
        memory_context += f"""
# Project Context
Key files: {project_mem['key_files'][:5]}
Coding conventions: {json.dumps(project_mem['conventions'], indent=2)}
"""
    
    if recent_episodes:
        memory_context += "\n# Similar Issues in History\n"
        for ep in recent_episodes:
            memory_context += f"- Issue: {ep['error'][:100]}\n  Solution: {ep['solution'][:200]}\n"
    
    prompt = f"{memory_context}\nUser request: {state['messages'][-1].content}"
    # ... subsequent planning logic
```

### 4.6 Memory Maintenance Strategies

| Dimension | Strategy | Description |
|-----------|----------|-------------|
| **Expiry** | TTL + LRU eviction | Project memory unmodified for 7 days marked as "possibly stale" |
| **Deduplication** | Error pattern-based dedup | Only keep the last 3 records for the same error type |
| **Merge** | Periodic merge of similar entries | Operations on the same file merged into a file change history |
| **Forgetting** | Manual cleanup / auto pruning | Episodic memory keeps only the last 100 entries |
| **Serialization** | JSON, single entry ≤ 1MB | Large files store reference paths, not content |

### 4.7 Memory-related State Fields

Extend `CodingAgentState` with the following fields:

```python
class CodingAgentState(TypedDict):
    # ... existing fields ...
    
    # --- Memory (new) ---
    session_id: str                    # Current session ID for checkpointer
    long_term_memory: dict             # Retrieved results cache from Store
    memory_summary: Optional[str]      # Session summary from previous turn (after truncation)
```

---

## 5. Node Design

### 5.1 Planner Node

```python
def planner_node(state: CodingAgentState) -> dict:
    """Analyze user request and generate an executable plan."""
    system_prompt = """You are a programming agent planner. Given the user's request:
1. Analyze what needs to be done
2. Break it down into sequential steps
3. Identify files that need to be read/modified
4. Return a structured plan

Each step should be one concrete action: read a file, edit a file, run a command, etc."""
    
    response = llm.with_structured_output(PlanSchema).invoke([
        SystemMessage(system_prompt),
        *state["messages"]
    ])
    
    return {
        "plan": response.steps,
        "current_step": 0,
        "relevant_files": response.files,
        "next_action": "executor"
    }
```

**Key design points:**
- Use `with_structured_output` to ensure plan format
- Return `relevant_files` for subsequent nodes to pre-cache
- Reset `current_step = 0`, clear previous turn's `errors`

### 5.2 Executor Node

```python
def executor_node(state: CodingAgentState) -> dict:
    """Execute the current step via tool calls."""
    step = state["plan"][state["current_step"]]
    
    # Let LLM decide which tool to call and what arguments to pass
    response = llm.bind_tools(tools).invoke([
        SystemMessage(EXECUTOR_PROMPT),
        *state["messages"],
        HumanMessage(f"Execute step {state['current_step']}: {step}")
    ])
    
    results = []
    for tool_call in response.tool_calls:
        result = execute_tool(tool_call, state["workspace"])
        results.append(result)
    
    # Check for errors
    errors = [r.error for r in results if not r.success]
    
    next_action = "evaluator" if errors else "executor"
    if state["current_step"] >= len(state["plan"]) - 1 and not errors:
        next_action = "output"
    
    return {
        "tool_results": results,
        "errors": errors,
        "step_attempts": state["step_attempts"] + 1,
        "current_step": state["current_step"] + (0 if errors else 1),
        "next_action": next_action
    }
```

**Key design points:**
- Each step lets the LLM make one or a set of parallel Tool Calls
- On success, `current_step + 1`; on failure, keep current step and enter evaluator
- All tool results are appended to `tool_results`, forming an audit trail

### 5.3 Evaluator Node

```python
RETRY_LIMIT = 3

def evaluator_node(state: CodingAgentState) -> dict:
    """Analyze errors and decide on a fix strategy."""
    error = state["errors"][-1]
    step = state["plan"][state["current_step"]]
    retries = state["retry_count"]
    
    if retries >= RETRY_LIMIT:
        # Exceeded retry limit → replan
        return {
            "next_action": "planner",
            "retry_count": 0,
        }
    
    # Let LLM analyze the error and generate a fix
    fix = llm.invoke([
        SystemMessage(ANALYZE_ERROR_PROMPT),
        HumanMessage(
            f"Step: {step}\nError: {error}\nRetry #{retries + 1}\n"
            "Analyze and suggest a fix."
        )
    ])
    
    if fix.needs_replan:
        # Need to replan (e.g., step order is wrong)
        return {
            "plan": fix.new_plan,
            "current_step": 0,
            "errors": [],
            "retry_count": 0,
            "next_action": "planner"
        }
    
    if fix.adjust_step:
        # Adjust current step description and retry
        new_plan = state["plan"].copy()
        new_plan[state["current_step"]] = fix.adjusted_step
        return {
            "plan": new_plan,
            "retry_count": retries + 1,
            "next_action": "executor"
        }
    
    # Simple retry
    return {
        "retry_count": retries + 1,
        "next_action": "executor"
    }
```

**Error classification handling:**

| Error type | Handling strategy | Example |
|------------|-------------------|---------|
| Syntax/compile error | Read error line → generate fix patch | `SyntaxError`, `TS2304` |
| Runtime exception | Add debug logging → retry | `KeyError`, `TypeError` |
| Timeout | Simplify operation / increase timeout | Command execution >30s |
| Tool argument error | Fix argument format and retry | File path not found |
| Logic not as expected | Replan steps | User feedback "not what I meant" |

### 5.4 Output Node

```python
def output_node(state: CodingAgentState) -> dict:
    """Summarize execution results and generate the final reply."""
    summary = llm.invoke([
        SystemMessage(SUMMARIZE_PROMPT),
        HumanMessage(
            f"Plan: {state['plan']}\n"
            f"Results: {json.dumps(state['tool_results'], indent=2)}\n"
            f"Generate a summary of what was done."
        )
    ])
    
    return {
        "final_output": summary.content,
        "next_action": "end"
    }
```

---

## 6. Conditional Edges and Routing

### 6.1 Route Function

```python
def should_continue(state: CodingAgentState) -> str:
    """Global conditional route, determines next node based on next_action."""
    
    # Circuit breaker
    if state["step_attempts"] > state["max_steps"]:
        return "output"
    
    return state.get("next_action", "planner")
```

### 6.2 Graph Compilation

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(CodingAgentState)

# Add nodes
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("output", output_node)

# Add edges
builder.add_edge(START, "planner")
builder.add_conditional_edges(
    "planner",
    should_continue,
    {"executor": "executor", "output": "output"}
)
builder.add_conditional_edges(
    "executor",
    should_continue,
    {
        "executor": "executor",    # Continue to next step
        "evaluator": "evaluator",  # Error occurred
        "output": "output",        # All steps done
        "planner": "planner",      # Needs replanning
    }
)
builder.add_conditional_edges(
    "evaluator",
    should_continue,
    {
        "executor": "executor",
        "planner": "planner",
        "output": "output"
    }
)
builder.add_edge("output", END)

graph = builder.compile()
```

---

## 7. Tool Design

### 7.1 Tool List

| Tool | Signature | Description |
|------|-----------|-------------|
| `read_file` | `(path, offset?, limit?) → ToolResult` | Read file content, supports range |
| `write_file` | `(path, content) → ToolResult` | Write/overwrite file |
| `edit_file` | `(path, old_str, new_str) → ToolResult` | Search/Replace edit |
| `delete_file` | `(path) → ToolResult` | Delete file |
| `glob` | `(pattern, path?) → ToolResult` | File name search |
| `grep` | `(pattern, path?, include?) → ToolResult` | Content search |
| `run_command` | `(cmd, cwd?, timeout?) → ToolResult` | Execute shell command |
| `semantic_search` | `(query, path?) → ToolResult` | Semantic code search |

### 7.2 Tool Contract

Every tool must follow a unified contract:

```python
@dataclass
class ToolResult:
    tool_name: str
    success: bool
    data: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0

def tool_wrapper(func):
    """Tool wrapper: unified exception handling, timing, logging."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=func.__name__,
                success=True,
                data=str(result),
                duration_ms=duration
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=func.__name__,
                success=False,
                error=str(e),
                duration_ms=duration
            )
    return wrapper
```

### 7.3 Security Constraints

```python
ALLOWED_COMMANDS = {"python", "node", "npm", "git", "ls", "cat", "pwd", "mkdir", "cp", "mv"}
BLOCKED_PATHS = {"/etc", "/root", "/var", "/sys"}
MAX_COMMAND_TIMEOUT = 60  # seconds
MAX_OUTPUT_SIZE = 10_000  # characters

def validate_tool_call(tool_name: str, args: dict, workspace: str):
    """Tool call security validation."""
    if tool_name == "run_command":
        cmd = args.get("command", "").split()[0]
        if cmd not in ALLOWED_COMMANDS:
            raise PermissionError(f"Command '{cmd}' not allowed")
    
    if tool_name in ("read_file", "write_file", "edit_file", "delete_file"):
        path = Path(args["path"]).resolve()
        if not str(path).startswith(workspace):
            raise PermissionError(f"Path outside workspace: {path}")
        if any(str(path).startswith(p) for p in BLOCKED_PATHS):
            raise PermissionError(f"Path blocked: {path}")
```

---

## 8. Error Recovery Strategy

### 8.1 Recovery Levels

```
Error occurred
    │
    ├── Level 1: Direct retry (same operation again)
    │    Applies to: network hiccup, process contention
    │
    ├── Level 2: Adjust parameters and retry
    │    Applies to: path error, timeout (increase timeout)
    │
    ├── Level 3: LLM analyzes error and fixes
    │    Applies to: syntax error, logic error
    │
    └── Level 4: Replan (back to planner)
         Applies to: misunderstood requirements, wrong step order
```

### 8.2 Circuit Breaker

```python
MAX_RETRIES_PER_STEP = 3
MAX_STEP_ATTEMPTS = 20
MAX_CONSECUTIVE_ERRORS = 5

def check_circuit_breaker(state: CodingAgentState) -> bool:
    """Check whether circuit breaker should trip."""
    if state["step_attempts"] >= MAX_STEP_ATTEMPTS:
        return True  # Trip: total attempts exceeded
    
    if state["retry_count"] >= MAX_RETRIES_PER_STEP:
        return True  # Trip: per-step retries exceeded
    
    # Consecutive error detection
    recent = state["tool_results"][-MAX_CONSECUTIVE_ERRORS:]
    if len(recent) >= MAX_CONSECUTIVE_ERRORS and \
       all(not r.success for r in recent):
        return True  # Trip: N consecutive failures
    
    return False
```

---

## 9. Observability

### 9.1 LangGraph Built-in Support

```python
graph = builder.compile()
```

- `graph.get_state(config)` — Get state snapshot at any point in time
- `graph.get_state_history(config)` — Get execution history (requires `checkpointer`)
- LangSmith callbacks — Full trace tracking

### 9.2 Custom Callbacks

```python
class AgentLogger:
    def on_node_start(self, node: str, state: dict):
        logger.info(f"[{node}] step={state['current_step']}, "
                    f"errors={len(state['errors'])}")
    
    def on_node_end(self, node: str, state: dict):
        logger.info(f"[{node}] completed, next={state['next_action']}")
```

---

## 10. Prompt Management Module

### 10.1 Layered Prompt Architecture

```
┌──────────────────────────────────────────┐
│           System-level System Prompt     │ ← Agent role definition, capability boundary
├──────────────────────────────────────────┤
│           Node-level Task Prompt         │ ← Node-specific instructions
│  ┌────────┐ ┌────────┐ ┌──────────┐     │
│  │Planner │ │Executor│ │Evaluator │ ... │
│  └────────┘ └────────┘ └──────────┘     │
├──────────────────────────────────────────┤
│           Memory context injection        │ ← Dynamically concatenated long-term memory
├──────────────────────────────────────────┤
│           User message                   │ ← Raw input
└──────────────────────────────────────────┘
```

### 10.2 Prompt Template Management

```python
from string import Template
from typing import Protocol
import json, yaml
from pathlib import Path

class PromptTemplate:
    """Versioned prompt template."""
    
    def __init__(self, name: str, version: str, template: str):
        self.name = name
        self.version = version
        self._template = Template(template)
    
    def render(self, **kwargs) -> str:
        return self._template.safe_substitute(**kwargs)


class PromptRegistry:
    """Prompt registry with multi-version and A/B testing support."""
    
    def __init__(self, prompts_dir: str = "prompts/"):
        self.prompts_dir = Path(prompts_dir)
        self._registry: dict[str, dict[str, PromptTemplate]] = {}
        self._load_all()
    
    def _load_all(self):
        """Load all YAML template files from prompts/ directory."""
        for file in self.prompts_dir.glob("*.yaml"):
            with open(file) as f:
                data = yaml.safe_load(f)
                for name, versions in data.items():
                    self._registry[name] = {}
                    for ver, template in versions.items():
                        self._registry[name][ver] = PromptTemplate(
                            name=name, version=ver, template=template
                        )
    
    def get(self, name: str, version: str = "latest") -> PromptTemplate:
        """Get template by name and version; 'latest' picks the highest version."""
        versions = self._registry.get(name, {})
        if not versions:
            raise KeyError(f"Prompt '{name}' not found")
        if version == "latest":
            version = sorted(versions.keys())[-1]
        return versions[version]
    
    def list_templates(self) -> list[dict]:
        return [
            {"name": n, "versions": list(v.keys())}
            for n, v in self._registry.items()
        ]
```

### 10.3 Node Prompt Design

| Node | Core instruction | Output format |
|------|-----------------|---------------|
| **Planner** | Analyze request → ordered steps → identify files | `PlanSchema` (structured) |
| **Executor** | "Execute step N: {step_desc}", available tools... | Tool Call |
| **Evaluator** | Analyze error → determine level (retry/adjust/replan) | `FixSchema` (structured) |
| **Output** | Summarize results → human-readable summary | Free text |

### 10.4 Structured Output Schemas

```python
from pydantic import BaseModel, Field
from typing import Literal

class PlanSchema(BaseModel):
    """Planner node output format."""
    analysis: str = Field(description="Summary of task analysis")
    steps: list[str] = Field(description="Ordered execution steps, each a concrete action")
    files: list[str] = Field(description="All file paths involved in the task")

class FixSchema(BaseModel):
    """Evaluator node output format."""
    error_type: Literal["retry", "adjust", "replan", "escalate"]
    analysis: str = Field(description="Root cause analysis")
    adjusted_step: str | None = Field(None, description="Adjusted step description")
    new_plan: list[str] | None = Field(None, description="Replanned steps")
```

### 10.5 LLM Call Wrapper

```python
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
import time, asyncio

class LLMClient:
    """LLM invocation wrapper with retry, backoff, structured output."""
    
    def __init__(self, model: BaseChatModel, max_retries: int = 3):
        self._model = model
        self.max_retries = max_retries
    
    def invoke(self, prompt: str, system: str = "", 
               schema: type[BaseModel] | None = None) -> str | BaseModel:
        """LLM call with retry."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                msgs = []
                if system:
                    msgs.append(SystemMessage(system))
                msgs.append(HumanMessage(prompt))
                
                if schema:
                    return self._model.with_structured_output(schema).invoke(msgs)
                return self._model.invoke(msgs).content
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
        raise last_error
    
    async def ainvoke(self, prompt: str, system: str = "",
                      schema: type[BaseModel] | None = None):
        """Async version."""
        # ... similar implementation
```

### 10.6 Prompt Version Management Directory Structure

```
prompts/
├── planner.yaml
│   ├── v1: "Initial version"
│   └── v2: "Added project memory context injection"
├── executor.yaml
│   ├── v1: "Basic tool calling instructions"
│   └── v2: "Added parallel tool call support"
├── evaluator.yaml
│   └── v1: "Four-level error classification handling"
└── output.yaml
    └── v1: "Structured summary"
```

---

## 11. Security Sandbox Module

### 11.1 Security Model

```
User input
    │
    ▼
┌──────────────────────┐
│  Input sanitization   │ ← Injection detection, path traversal detection
├──────────────────────┤
│  Permission check     │ ← Command allowlist, path allowlist
├──────────────────────┤
│  Execution isolation  │ ← Sandbox environment (Docker/subprocess)
├──────────────────────┤
│  Output filtering     │ ← Sensitive info redaction (API Key, Token)
└──────────────────────┘
    │
    ▼
User output
```

### 11.2 Three-layer Security Check

```python
import re, shlex
from pathlib import Path

class SecurityManager:
    """Security manager: input sanitization → permission check → output filtering."""
    
    def __init__(self, workspace: str, allowed_commands: set[str] | None = None):
        self.workspace = Path(workspace).resolve()
        self.allowed_commands = allowed_commands or {
            "python", "node", "npm", "npx", "git", "ls", "cat",
            "pwd", "mkdir", "cp", "mv", "rm", "echo", "touch",
            "head", "tail", "wc", "sort", "uniq", "diff", "patch",
        }
        self.blocked_patterns = [
            r"(rm\s+(-rf?\s+)?[/~]|sudo|chmod\s+777|:\(\)\s*\{|eval\s*\(|exec\s*\(|import\s+os)",
            r"(/etc/passwd|/etc/shadow|/root/|/var/log)",
        ]
        self.sensitive_patterns = [
            r'(?i)(api[_-]?key|secret|token|password|credential)\s*[:=]\s*["\']?\w{16,}',
            r'(?i)(ghp_|gho_|ghu_|ghs_|github_pat_)\w+',
            r'(?i)(sk-[a-zA-Z0-9]{20,})',  # OpenAI Key
        ]
    
    # --- Layer 1: Input sanitization ---
    def sanitize_input(self, text: str) -> str:
        """Detect and remove potential injection content."""
        for pattern in self.blocked_patterns:
            if re.search(pattern, text):
                raise SecurityError(f"Blocked dangerous pattern: {pattern}")
        return shlex.quote(text) if text else text
    
    # --- Layer 2: Permission check ---
    def validate_command(self, command: str) -> bool:
        """Check if command is in the allowlist."""
        cmd_name = command.strip().split()[0]
        base_cmd = Path(cmd_name).name
        if base_cmd not in self.allowed_commands:
            raise SecurityError(f"Command '{base_cmd}' not in allowlist")
        for segment in command.split("|"):
            seg_cmd = segment.strip().split()[0]
            if seg_cmd in ("sudo", "su", "chmod", "chown", "passwd"):
                raise SecurityError(f"Blocked privileged command: {seg_cmd}")
        return True
    
    def validate_path(self, path: str) -> Path:
        """Ensure path is within the workspace directory."""
        resolved = Path(path).resolve()
        if not str(resolved).startswith(str(self.workspace)):
            raise SecurityError(f"Path escape detected: {path} → {resolved}")
        if resolved.is_symlink():
            real = resolved.resolve()
            if not str(real).startswith(str(self.workspace)):
                raise SecurityError(f"Symlink escape: {resolved} → {real}")
        return resolved
    
    # --- Layer 3: Output filtering ---
    def sanitize_output(self, output: str) -> str:
        """Redact sensitive information."""
        result = output
        for pattern in self.sensitive_patterns:
            result = re.sub(pattern, "[REDACTED]", result)
        return result
```

### 11.3 Isolated Execution Environment

```python
import subprocess, tempfile, os, signal

class SandboxExecutor:
    """Sandbox execution environment."""
    
    def __init__(self, security: SecurityManager, 
                 timeout: int = 30, max_output: int = 10_000):
        self.security = security
        self.timeout = timeout
        self.max_output = max_output
    
    def run_command(self, command: str, cwd: str | None = None) -> dict:
        """Execute a command in the sandbox."""
        # 1. Validate
        self.security.validate_command(command)
        cwd = cwd or str(self.security.workspace)
        self.security.validate_path(cwd)
        
        # 2. Execute (set process group for cleanup)
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,  # Create new process group
                env={**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin"},
            )
        except Exception as e:
            return {"success": False, "error": f"Execution failed: {e}"}
        
        # 3. Wait + timeout control
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # Kill entire process group
            stdout, stderr = proc.communicate()
            return {
                "success": False,
                "error": f"Timeout after {self.timeout}s",
                "partial_output": stdout[-self.max_output:]
            }
        
        # 4. Truncate + redact
        output = (stdout or "") + (stderr or "")
        output = output[:self.max_output]
        output = self.security.sanitize_output(output)
        
        return {
            "success": proc.returncode == 0,
            "data": output if proc.returncode == 0 else None,
            "error": output if proc.returncode != 0 else None,
            "exit_code": proc.returncode,
        }
```

### 11.4 Security Configuration

```python
@dataclass
class SecurityConfig:
    """Security configuration."""
    sandbox_mode: Literal["none", "subprocess", "docker"] = "subprocess"
    allowed_commands: set[str] | None = None  # None = use default allowlist
    blocked_commands: set[str] = field(default_factory=lambda: {"sudo", "su", "kill"})
    block_network: bool = False                # Docker: isolate network
    max_command_timeout: int = 60
    max_output_size: int = 10_000
    audit_log: bool = True                     # Log all commands to audit trail
    sensitive_env_keys: list[str] = field(
        default_factory=lambda: ["API_KEY", "OPENAI_API_KEY", "SECRET", "TOKEN"]
    )
```

---

## 12. Human-in-the-loop Module

### 12.1 Interaction Modes

```
┌───────────────────────────────────────────────┐
│               Interaction Modes                │
├──────────────┬──────────────┬─────────────────┤
│  Silent mode  │  Confirm mode │  Guidance mode  │
│  (default)    │              │                 │
├──────────────┼──────────────┼─────────────────┤
│  Agent        │ Pause before │ Ask user when   │
│  executes     │ risky        │ ambiguous       │
│  autonomously │ operations   │                 │
├──────────────┼──────────────┼─────────────────┤
│  No break     │ write/delete │ Plan uncertain  │
│               │ /run command │ Multiple options │
└──────────────┴──────────────┴─────────────────┘
```

### 12.2 LangGraph Interrupt Mechanism

```python
from langgraph.types import interrupt, Command

# --- Confirm mode: pause before critical operations ---

def executor_with_human_confirm(state: CodingAgentState) -> dict:
    """Request user confirmation before execution."""
    
    step = state["plan"][state["current_step"]]
    
    if _needs_confirmation(step):
        user_response = interrupt({
            "type": "confirm",
            "step": step,
            "step_index": state["current_step"],
            "options": ["continue", "skip", "modify", "stop"]
        })
        
        if user_response == "skip":
            return {"current_step": state["current_step"] + 1}
        elif user_response == "stop":
            return {"next_action": "output"}
        elif user_response == "modify":
            modified = interrupt({
                "type": "input",
                "prompt": "Enter the modified step description:"
            })
            new_plan = state["plan"].copy()
            new_plan[state["current_step"]] = modified
            return {"plan": new_plan}
    
    return _execute_step(state)

def _needs_confirmation(step: str) -> bool:
    """Determine if a step needs user confirmation."""
    risky_patterns = [
        r"rm\s+-rf", r"delete\s+file", r"git\s+push.*--force",
        r"chmod", r">\s*/etc/", r"format", r"drop\s+table",
    ]
    return any(re.search(p, step, re.I) for p in risky_patterns)


# --- Guidance mode: ask user when ambiguous ---

def planner_with_clarification(state: CodingAgentState) -> dict:
    """When planning encounters ambiguity, guide user to clarify."""
    
    ambiguity = _detect_ambiguity(state["messages"][-1].content)
    if ambiguity:
        clarification = interrupt({
            "type": "clarify",
            "question": ambiguity,
            "options": _generate_options(ambiguity),
        })
        return {"messages": [HumanMessage(f"[User clarification] {clarification}")]}
    
    return _do_plan(state)
```

### 12.3 User Feedback Loop

```python
def output_with_feedback(state: CodingAgentState) -> dict:
    """Ask for user feedback after outputting results."""
    
    summary = _generate_summary(state)
    
    feedback = interrupt({
        "type": "feedback",
        "summary": summary,
        "options": ["accept", "modify", "retry", "discard"]
    })
    
    if feedback == "accept":
        return {"final_output": summary, "next_action": "end"}
    
    elif feedback == "modify":
        return {
            "messages": [HumanMessage(f"[Modification request] {feedback['detail']}")],
            "next_action": "planner",
        }
    
    elif feedback == "retry":
        return {
            "current_step": 0,
            "errors": [],
            "retry_count": 0,
            "step_attempts": 0,
            "next_action": "executor",
        }
    
    elif feedback == "discard":
        return {"final_output": "Operation discarded.", "next_action": "end"}
```

### 12.4 Checkpointer for Resume Support

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# SQLite persistent checkpoint, supports interrupt and resume
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# First call (may pause at interrupt point)
thread_config = {"configurable": {"thread_id": "session-001"}}
result = graph.invoke({"messages": [user_msg]}, thread_config)

# ... later, resume with the same thread_id
# LangGraph automatically continues from the last interrupt() point
result = graph.invoke(None, thread_config)
```

---

## 13. Cost Control Module

### 13.1 Token Tracking

```python
from dataclasses import dataclass, field
import time

@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    model: str = ""
    timestamp: float = 0.0

@dataclass
class SessionCost:
    """Session-level cost statistics."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    call_count: int = 0
    call_history: list[TokenUsage] = field(default_factory=list)
    
    @property
    def summary(self) -> str:
        return (
            f"Token: {self.total_tokens:,} "
            f"(prompt {self.total_prompt_tokens:,} + "
            f"completion {self.total_completion_tokens:,}) | "
            f"Cost: ${self.total_cost:.4f} | "
            f"Calls: {self.call_count}"
        )


class CostTracker:
    """Cost tracker, embedded in graph execution callbacks."""
    
    PRICING = {
        "gpt-4o":          {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini":     {"input": 0.00015, "output": 0.0006},
        "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    }
    
    def __init__(self, budget: float = 1.0):
        self.budget = budget
        self.session = SessionCost()
    
    def track(self, model: str, prompt_tokens: int, completion_tokens: int):
        """Record one LLM call."""
        pricing = self.PRICING.get(model, {"input": 0.002, "output": 0.002})
        cost = (prompt_tokens / 1000) * pricing["input"] \
             + (completion_tokens / 1000) * pricing["output"]
        
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            model=model,
            timestamp=time.time(),
        )
        
        self.session.call_count += 1
        self.session.total_prompt_tokens += prompt_tokens
        self.session.total_completion_tokens += completion_tokens
        self.session.total_tokens += prompt_tokens + completion_tokens
        self.session.total_cost += cost
        self.session.call_history.append(usage)
    
    def is_over_budget(self) -> bool:
        return self.session.total_cost > self.budget
    
    def should_switch_model(self) -> bool:
        """When over budget or token consumption is too fast, suggest switching to cheaper model."""
        if self.is_over_budget():
            return True
        if self.session.call_count > 10:
            avg_cost = self.session.total_cost / self.session.call_count
            return avg_cost * 20 > self.budget
        return False
```

### 13.2 Model Selection Strategy

```python
class ModelSelector:
    """Dynamic model selection: cheap model for simple tasks, strong model for complex ones."""
    
    def __init__(self, tracker: CostTracker):
        self.tracker = tracker
    
    def select(self, task: str, complexity: str = "auto") -> str:
        """Select model based on task complexity and budget."""
        
        if self.tracker.should_switch_model():
            return "gpt-4o-mini"  # Degrade to cheaper model
        
        if complexity == "auto":
            complexity = self._estimate_complexity(task)
        
        mapping = {
            "simple": "gpt-4o-mini",
            "medium": "gpt-4o",
            "complex": "claude-3.5-sonnet",
        }
        return mapping.get(complexity, "gpt-4o")
    
    def _estimate_complexity(self, task: str) -> str:
        """Simple heuristic for task complexity."""
        length = len(task)
        if length < 50:
            return "simple"
        elif length < 200:
            return "medium"
        else:
            return "complex"
```

### 13.3 Cost Budget in State

```python
class CodingAgentState(TypedDict):
    # ... existing fields ...
    
    # --- Cost control (new) ---
    cost_budget: float                     # Session budget cap (USD)
    current_cost: float                    # Current consumption
    token_usage: dict                      # {model: {"prompt": N, "completion": N}}
    cost_warning: bool                     # Whether approaching budget cap
```

---

## 14. Testing & Evaluation Module

### 14.1 Test Levels

```
┌──────────────────────────────────────────┐
│  Level 1: Unit Tests                      │
│  Test individual tool functions,          │
│  security validation, state update logic  │
├──────────────────────────────────────────┤
│  Level 2: Node Tests                      │
│  Mock state input, verify node outputs    │
│  and routing decisions                    │
├──────────────────────────────────────────┤
│  Level 3: Graph Integration Tests         │
│  End-to-end task execution,              │
│  verify graph flow                        │
├──────────────────────────────────────────┤
│  Level 4: E2E Evaluation                  │
│  Evaluate task completion rate on         │
│  benchmark dataset                        │
└──────────────────────────────────────────┘
```

### 14.2 Unit Tests

```python
# tests/test_security.py
import pytest

def test_command_allowlist():
    sm = SecurityManager(workspace="/tmp/test")
    sm.validate_command("python script.py")    # OK
    sm.validate_command("ls -la")              # OK
    with pytest.raises(SecurityError):
        sm.validate_command("sudo rm -rf /")   # Blocked

def test_path_escape_detection():
    sm = SecurityManager(workspace="/tmp/test")
    with pytest.raises(SecurityError):
        sm.validate_path("/etc/passwd")        # Escape detected
    
    sm.validate_path("/tmp/test/foo.py")       # OK

# tests/test_state.py
def test_state_transition():
    state = CodingAgentState(
        messages=[HumanMessage("fix bug")],
        plan=["read file", "fix bug", "test"],
        current_step=0,
        next_action="executor",
        step_attempts=0,
    )
    result = executor_node(state)
    assert result["next_action"] in ("executor", "evaluator", "output")
    assert result["step_attempts"] == 1
```

### 14.3 Node Integration Tests

```python
# tests/test_planner.py
from langgraph.graph import StateGraph, START

def test_planner_produces_valid_plan():
    """Verify planner outputs a valid plan structure."""
    
    builder = StateGraph(CodingAgentState)
    builder.add_node("planner", planner_node)
    builder.add_edge(START, "planner")
    graph = builder.compile()
    
    result = graph.invoke({
        "messages": [HumanMessage("Create a Python hello world script")],
        "workspace": "/tmp/test_project",
        "max_steps": 20,
        "step_attempts": 0,
    })
    
    assert len(result["plan"]) > 0
    assert isinstance(result["plan"], list)
    assert all(isinstance(s, str) for s in result["plan"])
    assert result["next_action"] == "executor"
```

### 14.4 E2E Evaluation Benchmark

```python
@dataclass
class EvalTask:
    """Evaluation task definition."""
    id: str
    description: str                      # Task description
    workspace_setup: callable             # Function to set up test environment
    expected_files: list[str]             # Expected created/modified files
    expected_content_checks: list[callable]  # Content validation functions
    max_steps: int = 20
    timeout: int = 120

@dataclass
class EvalResult:
    """Evaluation result."""
    task_id: str
    success: bool
    completed_steps: int
    total_cost: float
    errors: list[str]
    duration_s: float
    artifacts: dict                       # File snapshots during execution

class EvalSuite:
    """Evaluation suite."""
    
    def __init__(self, agent_builder: callable):
        self.agent_builder = agent_builder
        self.results: list[EvalResult] = []
    
    def add_task(self, task: EvalTask):
        self.tasks.append(task)
    
    def run_all(self):
        for task in self.tasks:
            result = self._run_single(task)
            self.results.append(result)
        return self.summary()
    
    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        return {
            "total": total,
            "passed": passed,
            "success_rate": passed / total * 100,
            "avg_cost": sum(r.total_cost for r in self.results) / total,
            "avg_steps": sum(r.completed_steps for r in self.results) / total,
        }
    
    def _run_single(self, task: EvalTask) -> EvalResult:
        """Run a single evaluation task."""
        workspace = task.workspace_setup()
        agent = self.agent_builder(workspace=workspace)
        
        start = time.time()
        try:
            result = agent.run(task.description, max_steps=task.max_steps)
            duration = time.time() - start
        except Exception as e:
            return EvalResult(
                task_id=task.id, success=False,
                completed_steps=0, total_cost=0,
                errors=[str(e)], duration_s=time.time() - start,
                artifacts={}
            )
        
        errors = []
        for expected_file in task.expected_files:
            if not Path(workspace, expected_file).exists():
                errors.append(f"Missing file: {expected_file}")
        
        for check_fn in task.expected_content_checks:
            if not check_fn(workspace):
                errors.append(f"Content check failed: {check_fn.__name__}")
        
        return EvalResult(
            task_id=task.id,
            success=len(errors) == 0,
            completed_steps=result.step_attempts,
            total_cost=result.current_cost,
            errors=errors,
            duration_s=duration,
            artifacts=result.tool_results,
        )
```

### 14.5 Evaluation Dataset Examples

```python
EVAL_TASKS = [
    EvalTask(
        id="create-hello-world",
        description="Create a Python script that prints 'Hello, World!'",
        workspace_setup=lambda: tempfile.mkdtemp(),
        expected_files=["hello.py"],
        expected_content_checks=[
            lambda w: "print" in Path(w, "hello.py").read_text(),
        ],
    ),
    EvalTask(
        id="fix-syntax-error",
        description="Fix the syntax error in buggy.py",
        workspace_setup=lambda: _setup_with_file(
            "buggy.py",
            "def greet(name)\n    print('Hello', name)\n"
        ),
        expected_files=["buggy.py"],
        expected_content_checks=[
            lambda w: "def greet(name):" in Path(w, "buggy.py").read_text(),
        ],
    ),
    EvalTask(
        id="refactor-to-class",
        description="Refactor functions.py into a Python class",
        workspace_setup=lambda: _setup_with_file(
            "functions.py",
            "def add(a, b): return a + b\ndef sub(a, b): return a - b\n"
        ),
        expected_files=["functions.py"],
        expected_content_checks=[
            lambda w: "class" in Path(w, "functions.py").read_text(),
        ],
    ),
]
```

### 14.6 State Fields for Testing

```python
class CodingAgentState(TypedDict):
    # ... existing fields ...
    
    # --- Testing & debugging (new) ---
    eval_mode: bool                        # Evaluation mode (skip interrupt)
    debug_mode: bool                       # Verbose debug output
    node_timings: dict[str, float]         # {node_name: cumulative_ms}
```

---

## 15. Configuration & Extension Points

### 15.1 Agent Configuration

```python
@dataclass
class AgentConfig:
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_steps: int = 20
    retry_limit: int = 3
    tool_timeout: int = 30
    workspace: str = "."
    verbose: bool = False
    checkpointer: Optional[Any] = None
    
    # Module toggles
    memory_enabled: bool = True
    sandbox_enabled: bool = True
    human_confirm: bool = False            # Enable confirmation mode
    cost_budget: float = 1.0              # Budget cap (USD), 0 = unlimited
    eval_mode: bool = False
```

### 15.2 Extension Points

| Extension | Description | Default Implementation |
|-----------|-------------|----------------------|
| `tools` | Tool list registration | Basic file/command tools (Chapter 7) |
| `prompts` | Custom prompt templates | `prompts/` directory loading (Chapter 9) |
| `security` | Security validation strategy | Command allowlist + path validation (Chapter 10) |
| `sandbox` | Sandbox execution engine | subprocess (Chapter 10) |
| `memory_store` | Long-term memory backend | SQLite (Chapter 4) |
| `cost_tracker` | Cost tracking strategy | Default pricing + budget control (Chapter 12) |
| `model_selector` | Model selection strategy | Complexity heuristic (Chapter 12) |

---

## 16. Edge Case Handling

| Scenario | Handling |
|----------|----------|
| Empty workspace | planner auto-creates directory structure |
| Large file | Read in segments, only read critical parts |
| Long-running command | Set `timeout`, evaluator intervenes on timeout |
| LLM malformed response | Fall back to string matching on parse failure, retry up to 2 times |
| User changes requirements mid-task | Pause via `interrupt` + checkpointer |
| Concurrent tool calls | Multiple Tool Calls in same Node execute in parallel |
| Tool returns sensitive info | Post-process to filter (API key, passwords, etc.) |
| Token exceeds context window | Summary-compress history (Section 4.2) |
| Cost over budget | Auto-degrade to cheaper model (Section 12.2) |
| N consecutive failures | Circuit breaker → output "Cannot auto-fix, manual intervention required" |

---

## 17. Future Evolution

- **Multi-agent collaboration**: Planner Agent + Coding Agent + Review Agent team
- **Persistent sessions**: Use `PostgresCheckpointer` for long-running tasks
- **Streaming output**: Use LangGraph's `.astream_events()` for real-time status push
- **Dedicated sandbox**: Docker-isolated execution environment for secure code execution
- **Online learning**: Auto-optimize prompt templates and tool selection from user feedback
- **Multi-modal**: Support screenshots, UI mockups as input (vision model analysis → code generation)
- **Plugin system**: Allow third-party registration of custom tools and nodes
