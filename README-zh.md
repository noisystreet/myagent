# myagent

[![CI](https://github.com/noisystreet/myagent/actions/workflows/ci.yml/badge.svg)](https://github.com/noisystreet/myagent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](https://pre-commit.com/)
[![PyPI](https://img.shields.io/pypi/v/myagent?label=PyPI)](https://pypi.org/project/myagent/)

基于 **LangGraph** 的编程 Agent — 能理解自然语言指令，自动完成代码编写、调试、重构等软件工程任务。

## 快速开始

```bash
# 安装
pip install -e .

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY、LLM_MODEL、LLM_BASE_URL

# 单次执行
myagent "Write a Python script hello.py"

# 交互模式
myagent
```

## 配置

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `LLM_API_KEY` | API Key | `sk-xxx` |
| `LLM_MODEL` | 模型名称 | `gpt-4o`, `deepseek-v4-flash` |
| `LLM_BASE_URL` | API 端点（可省略） | `https://api.deepseek.com` |

## 功能

- **聊天模式**：回答问题、解释概念、提供建议
- **任务模式**：读文件、写文件、编辑文件、执行命令
- **意图路由**：自动判断是聊天还是编程任务
- **多轮记忆**：同一会话内的对话历史保持
- **模型兼容**：支持 GPT-4o、Claude、DeepSeek、Ollama 等

## 文档

- [English README](README.md)
- [架构设计](docs/adr/design.md)
- [分阶段实施计划](docs/plan.md)

## 开发

```bash
make install    # 安装依赖
make test       # 运行测试
make lint       # 代码检查
make format     # 自动格式化
```

## 许可证

MIT
