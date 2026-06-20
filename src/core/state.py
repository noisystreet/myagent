"""State definitions for the coding agent."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ToolResult(dict):
    """Structured result from a tool execution.

    Dict subclass so it's msgpack-serializable (for checkpointer).
    Supports both dict style (result['success']) and attribute style (result.success).
    data and error may be str or dict for structured output.
    """

    def __init__(
        self,
        tool_name: str,
        success: bool,
        data: str | dict | None = None,
        error: str | dict | None = None,
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

    @staticmethod
    def _format_value(value: str | dict | None, indent: int = 0) -> str:
        """Render a value as structured text."""
        if value is None:
            return ""
        prefix = "  " * indent
        if isinstance(value, dict):
            lines: list[str] = []
            for k, v in value.items():
                if isinstance(v, dict):
                    lines.append(f"{prefix}{k}:")
                    lines.append(ToolResult._format_value(v, indent + 1))
                else:
                    val_lines = str(v).splitlines()
                    if len(val_lines) <= 1:
                        lines.append(f"{prefix}{k}: {v}")
                    else:
                        lines.append(f"{prefix}{k}: |")
                        for vl in val_lines:
                            lines.append(f"{prefix}  {vl}")
            return "\n".join(lines)
        return str(value)

    @property
    def data_text(self) -> str:
        """Return data rendered as text."""
        return self._format_value(self.get("data"))

    @property
    def error_text(self) -> str:
        """Return error rendered as text."""
        return self._format_value(self.get("error"))

    def to_text(self) -> str:
        """Return a single text representation of the result."""
        parts: list[str] = []
        if self.get("error"):
            parts.append(self.error_text)
        if self.get("data"):
            parts.append(self.data_text)
        return "\n".join(parts)

    def __repr__(self) -> str:
        status = "OK" if self["success"] else "FAIL"
        text = self.to_text()
        return (
            f"ToolResult({self['tool_name']}, {status}, "
            f"text_len={len(text)}, error={self.get('error') is not None})"
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
