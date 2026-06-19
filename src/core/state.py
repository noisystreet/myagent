"""State definitions for the coding agent."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ToolResult(dict):
    """Structured result from a tool execution.

    Dict subclass so it's msgpack-serializable (for checkpointer).
    Supports both dict style (result['success']) and attribute style (result.success).
    """

    def __init__(
        self,
        tool_name: str,
        success: bool,
        data: str | None = None,
        error: str | None = None,
        duration_ms: float = 0.0,
    ):
        super().__init__(
            tool_name=tool_name,
            success=success,
            data=data,
            error=error,
            duration_ms=duration_ms,
        )

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"ToolResult has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        self[name] = value

    def __repr__(self) -> str:
        status = "OK" if self["success"] else "FAIL"
        return (
            f"ToolResult({self['tool_name']}, {status}, "
            f"data_len={len(self.get('data') or '')}, error={self['error']})"
        )


class CodingAgentState(TypedDict):
    """State of the coding agent graph."""

    # --- Message layer ---
    messages: Annotated[list[AnyMessage], add_messages]

    # --- Task plan ---
    plan: list[str]  # Ordered steps, e.g. ["read src/main.py", "fix bug"]
    current_step: int  # Index into plan (0-based)
    max_steps: int  # Circuit breaker threshold

    # --- Context ---
    workspace: str  # Working directory

    # --- Execution tracking ---
    tool_results: list[ToolResult]  # All tool call history
    errors: list[str]  # Accumulated errors
    retry_count: int  # Retries for current step
    step_attempts: int  # Total attempts across all steps

    # --- Control flow ---
    mode: Literal["chat", "task"]  # Intent classification
    next_action: Literal[
        "intent_router", "chat", "planner", "executor", "evaluator", "output", "end"
    ]

    # --- Output ---
    final_output: str | None
