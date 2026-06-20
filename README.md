# myagent

[![CI](https://github.com/noisystreet/myagent/actions/workflows/ci.yml/badge.svg)](https://github.com/noisystreet/myagent/actions/workflows/ci.yml)

A **LangGraph-based programming agent** that understands natural language instructions and automates coding, debugging, and refactoring tasks.

## Quick Start

```bash
# Install
pip install -e .

# Configure API Key
cp .env.example .env
# Edit .env: set LLM_API_KEY, LLM_MODEL, LLM_BASE_URL

# Single-shot mode
myagent "Write a Python script hello.py"

# Interactive mode
myagent
```

## Configuration

| Env Var | Description | Example |
|---------|-------------|---------|
| `LLM_API_KEY` | API key | `sk-xxx` |
| `LLM_MODEL` | Model name | `gpt-4o`, `deepseek-v4-flash` |
| `LLM_BASE_URL` | API endpoint (optional) | `https://api.deepseek.com` |

## Features

- **Chat mode**: Answer questions, explain concepts, provide guidance
- **Task mode**: Read files, write files, edit files, run commands
- **Intent routing**: Auto-detect chat vs programming task
- **Multi-turn memory**: Cross-turn conversation history within a session
- **Model agnostic**: Supports GPT-4o, Claude, DeepSeek, Ollama, etc.

## Docs

- [Architecture Design](docs/adr/design.md)
- [Implementation Plan](docs/plan.md)
- [Agent Workflow](docs/agent_workflow.md)
- [中文文档](README-zh.md)

## Development

```bash
make install    # Install dependencies
make test       # Run tests
make lint       # Code linting
make format     # Auto-format
```

## License

MIT
