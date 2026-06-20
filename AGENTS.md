# AGENTS.md — Project Guide for AI Agents

## Project Identity

- **Project**: myagent — A LangGraph-based programming agent
- **Stack**: Python 3.11+, LangGraph ≥0.2.0, LangChain
- **Directory structure**:

```
src/
  main.py          # CLI entry point
  core/
    state.py       # CodingAgentState, ToolResult
    graph.py       # LangGraph graph definition & compilation
    config.py      # AgentConfig
  llm/
    client.py      # LLMClient (retry + structured output + JSON fallback)
    schemas.py     # Pydantic output schemas
  nodes/
    router.py      # Intent router (chat / task)
    chat.py        # Pure conversation node
    planner.py     # Task planner node
    executor.py    # Tool calling node
    output.py      # Result summarizer node
  tools/
    file_tools.py  # read_file, write_file, edit_file
    command_tools.py # run_command
```

## Hard Constraints

1. **Dependency direction**: `tools/` must not depend on `nodes/`; `nodes/` may depend on `tools/` and `llm/`; `core/` must not depend on any module
2. **Banned dependencies**: Do not add extra HTTP clients (use stdlib or LangChain built-ins); do not add extra serialization libraries (use pydantic)
3. **Document modification permission**: `docs/adr/design.md`, `docs/plan.md`, `docs/agent_workflow.md` require human approval before changes
4. **Code annotation**: Agent-generated significant changes must include a `# AI-generated` comment
5. **Security red lines**: Never hardcode API keys/tokens; command execution is restricted to an allowlist; file paths are restricted to the workspace directory
6. **Test requirement**: New tool functions must have corresponding test cases

## Coding Style

### Avoid Implicit State

Prefer pure functions — input determines output, avoid implicit dependencies on external state:

- **No global/module-level mutable state**: This includes module-level `dict`/`list` objects, `lru_cache` side effects, and implicit `os.environ` reads. Use explicit parameters or `dataclass` context objects instead.
- **Inject dependencies explicitly**: Pass config, LLM client, workspace path etc. as function parameters. Do not read them from module globals or `os.environ` implicitly. The `LLMClient` injection pattern is the correct example.
- **Do not mutate input objects**: Unless the function name clearly indicates write intent (e.g. `append`, `populate`), avoid modifying input parameters. Return new objects instead.
- **Consolidate context objects**: When multiple functions share the same state (config, workspace, etc.), use an explicit data structure to pass it around, rather than each function fetching from different global sources.

### Type Annotations

- All function parameters and return values must be annotated. Use Python 3.10+ union syntax (`str | None` not `Optional[str]`)
- Custom types like `ToolResult` should be defined in `state.py`
- Prefer `dict[str, Any]` over bare `dict` for return types

### Naming Conventions

- Functions/variables: `snake_case`
- Classes: `PascalCase` (e.g. `ToolResult`, `CodingAgentState`)
- Constants: `UPPER_CASE` (e.g. `MAX_RETRIES`, `EXECUTOR_SYSTEM_PROMPT`)
- Private functions/methods: `_` prefix (e.g. `_parse_tool_call`, `_run_tool`)
- File names: `snake_case.py`, globally unique within the project

### Input Validation Boundaries

- All external input (CLI args, LLM responses, user file paths) must be validated once at the **boundary layer**; internal functions assume valid input
- LLM response parsing must go through `executor._parse_tool_call`'s three-layer strategy (JSON → function call → free text inference), not reimplemented elsewhere
- File paths are resolved to absolute paths centrally in `executor._resolve_and_run`; downstream tools handle only valid paths

### File Size Limits

- Single source file ≤ 400 lines (excluding tests)
- Single function ≤ 150 lines (enforced by `make complexity` with lizard)
- Exceed limits → split into multiple files or extract helper functions

### Dead Code

- Do not commit commented-out code blocks, `pass` placeholders, unused imports or variables
- Ruff rules `F` (errors) and `I` (unused imports) are enabled in `pyproject.toml`; they must pass before commit
- Clean up temporary debug code (`print`, `breakpoint`) before committing

### Error Handling

- Expected errors return `ToolResult` (`success` + `data`/`error`), do not raise exceptions
- Unrecoverable errors (missing config, invalid API key) raise exceptions, caught by `main()` at the top level
- No bare `except:` — always specify the exception type

### Logging

- Use the stdlib `logging` module, not `print()`
- Level convention: `ERROR` (needs human intervention), `WARNING` (auto-recoverable), `INFO` (key business events), `DEBUG` (debugging details)

### Import Order

- stdlib → third-party → internal modules, separated by blank lines
- Internal modules use relative imports (`from ..core.state import ...`)

### Commit Scope Whitelist

Conventional Commits `scope` must be one of the following, matching the project directory structure:

| scope | Scope |
|-------|-------|
| `core` | `src/core/` (state, graph, config) |
| `llm` | `src/llm/` (client, schemas) |
| `nodes` | `src/nodes/` (router, chat, planner, executor, output) |
| `tools` | `src/tools/` (file_tools, command_tools) |
| `cli` | `src/main.py` |
| `ci` | `.github/`, `.pre-commit-config.yaml` |
| `docs` | `docs/`, README, AGENTS |
| `chore` | Non-`src/` changes (deps, config, git) |

Example: `fix(nodes): handle empty plan gracefully`, not `fix(misc)` or `fix(other)`.

## Verification Commands

```bash
make install    # Install dependencies
make test-cov   # Run tests with coverage gate (≥35%)
make lint       # ruff code linting
make complexity # Cyclomatic complexity + line length check (lizard)
make audit      # Dependency vulnerability audit
make lock       # Generate requirements.txt lock file
make check      # Full pipeline: install → lint → complexity → audit → test-cov
python -m src.main 'test task'  # End-to-end verification
```

## Language Convention

### Project Docs
- Key docs have both Chinese and English versions: `README.md` (English) + `README-zh.md` (Chinese)
- `AGENTS.md` is primarily English
- `docs/adr/`, `docs/plan.md` and other design docs are in English

### Code
- Code comments, variable names, function names, and commit messages must be in **English**
- Docstrings must be in English
- Log messages must be in English

### Commit Messages
- Bilingual format: **English summary line + Chinese summary line**, each on its own line, content corresponds to each other
- Both lines have the `<type>(<scope>)` prefix — the Chinese line is a translation of the English line
- Format:
  ```
  <type>(<scope>): <English summary>
  <type>(<scope>): <中文概要>
  
  <English body (optional, blank line after summary)>
  ```
- Example:
  ```
  feat(executor): add JSON fallback for models without structured output
  feat(executor): 为不支持结构化输出的模型添加 JSON fallback
  
  Models that don't support response_format (e.g. DeepSeek) now
  automatically fall back to text output + JSON parsing via a
  three-layer strategy: structured output → instructed JSON → free text inference.
  ```
- The English line must be a complete, meaningful sentence — not just a keyword
- The Chinese line is the corresponding translation, maintaining the same `<type>(<scope>)`
- The body (if present) is in English, explaining what was done and why

## Development Process (GitHub Flow)

This project uses the **GitHub Flow** branching model.

### Branch Naming

| Branch | Purpose | Description |
|--------|---------|-------------|
| `main` | Production-ready code | Always deployable; no direct pushes |
| `feat/<short-desc>` | New feature | Create from main, e.g. `feat/add-evaluator` |
| `fix/<short-desc>` | Bug fix | Create from main, e.g. `fix/json-fallback-crash` |
| `chore/<short-desc>` | Maintenance (CI, deps, refactor) | Create from main, e.g. `chore/update-deps` |

Naming: lowercase, `-` separated, English.

### Workflow

```
main ─── feat/xxx ─── PR ─── merge → main
  │                      ↑
  └──────────────────────┘
```

1. **Create branch**: from `main`
2. **Develop & commit**: multiple commits, each independently buildable
3. **Open PR**: push to GitHub, PR title follows Conventional Commits
4. **CI check**: GitHub Actions must pass (lint + test + build)
5. **Code review**: at least 1 approval required before merge
6. **Merge to main**: use **Squash and Merge** to keep a linear history
7. **Delete source branch**: clean up after merge

### Commit Convention (Conventional Commits)

```
<type>(<scope>): <English title>
<type>(<scope>): <中文标题>

<English body (optional, blank line after title)>
```

Types: `feat` / `fix` / `refactor` / `test` / `docs` / `chore` / `style`

Example:
```
feat(executor): add JSON fallback for models without structured output
feat(executor): 为不支持结构化输出的模型添加 JSON fallback

Models that don't support response_format (e.g. DeepSeek) now
automatically fall back to text output + JSON parsing via a
three-layer strategy: structured output → instructed JSON → free text inference.
```

### PR Rules

- **Size limit**: effective changes ≤ 400 lines (excluding tests, config, generated code)
- **Description required**: change summary + test strategy + impact scope
- **AI annotation**: AI-generated code must be marked with `🤖 AI-generated` in the PR description
- **AI self-check checklist** (must be verified before declaring task complete):

  | # | Check | Why |
  |---|-------|-----|
  | 1 | ✅ Run `make lint` — ruff must pass | Ruff 版本不同会导致 CI 失败 |
  | 2 | ✅ Run `make test` — all tests pass | 不改坏已有功能 |
  | 3 | ✅ Run `make complexity` — lizard no warnings | 圈复杂度超标会被 CI 拒绝 |
  | 4 | ✅ Create/fix PR on GitHub (`gh pr create`) | 分支推了 PR 没创建等于没做 |
  | 5 | ✅ Verify CI passes on the PR | 合并前必须绿色 |
  | 6 | ✅ Update `AGENTS.md` if adding/changing constraints | 约定要落地到文档 |
  | 7 | ✅ Update `CHANGELOG.md` if user-facing change | 用户需要知道变更 |
  | 8 | ✅ Bilingual commit format (`<type>(<scope>): EN` + `<type>(<scope>): 中文`) | 提交规范必须遵守 |

### Versioning & Release

- Follows semantic versioning (`MAJOR.MINOR.PATCH`)
- Tag from `main`: `git tag v0.1.0 && git push --tags`
- CHANGELOG.md is updated to consolidate changes before each release
