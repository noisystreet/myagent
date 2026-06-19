"""Configuration for the coding agent."""

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AgentConfig:
    """Agent configuration with defaults from environment variables."""

    # LLM
    model: str = os.getenv("LLM_MODEL", "gpt-4o")
    api_key: str | None = os.getenv("LLM_API_KEY")
    base_url: str | None = os.getenv("LLM_BASE_URL")
    temperature: float = 0.0

    # Execution
    max_steps: int = 20
    retry_limit: int = 3
    tool_timeout: int = 30

    # Paths
    workspace: str = "."

    # Flags
    verbose: bool = False
    checkpointer: Any | None = None

    # Module toggles (for later phases)
    memory_enabled: bool = False
    sandbox_enabled: bool = False
    human_confirm: bool = False
    cost_budget: float = 0.0
    eval_mode: bool = False
