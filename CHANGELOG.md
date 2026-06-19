# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Project scaffold: LangGraph-based programming agent skeleton
- Core graph structure: intent_router → (chat | planner → executor → output)
- Chat mode: multi-turn conversation with cross-turn memory
- Task mode: read_file / write_file / edit_file / run_command tools
- Intent routing: auto-detect chat vs programming task
- LLM client: structured output + JSON fallback (compatible with DeepSeek, local models)
- Environment config: LLM_MODEL / LLM_BASE_URL / LLM_API_KEY support
- Interactive CLI: single-shot and REPL modes
- Project skeleton: LICENSE, CONTRIBUTING, SECURITY, CHANGELOG, CI, pre-commit, docs
- Coding style constraints: type annotations, naming conventions, complexity gates
