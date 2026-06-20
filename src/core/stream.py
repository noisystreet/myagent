"""Streaming execution helpers for the LangGraph agent.

Provides node-level streaming via graph.stream(), yielding display-ready
events for real-time output in the interactive CLI.
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

from ..core.state import CodingAgentState

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """A display-ready event yielded during streaming execution."""

    node: str
    kind: str  # "start", "interim", "tool_result", "error", "end"
    content: str = ""
    data: dict = field(default_factory=dict)


def run_streaming(
    graph: Any,
    state: CodingAgentState,
    config: dict,
) -> Generator[StreamEvent, None, None]:
    """Execute the graph with node-level streaming.

    Yields StreamEvent objects for real-time display.

    This does NOT implement token-by-token LLM output streaming.
    It streams at the node granularity — you see each node's result
    as it completes, rather than waiting for the full graph.
    """
    for event in graph.stream(dict(state), config):
        for node_name, output in event.items():
            if node_name == "__end__":
                continue
            yield _event_from_node(node_name, output)


def _event_from_node(node_name: str, output: dict) -> StreamEvent:
    """Convert a node's raw output dict into a display event."""
    # intent_router: shows routing decision
    if node_name == "intent_router":
        mode = output.get("mode", "?")
        return StreamEvent(
            node=node_name,
            kind="start",
            content=f"Routing: {mode} mode",
            data=output,
        )

    # planner: shows generated plan
    if node_name == "planner":
        plan = output.get("plan", [])
        plan_str = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan))
        return StreamEvent(
            node=node_name,
            kind="interim",
            content=f"Plan ({len(plan)} steps):\n{plan_str}" if plan else "No plan generated",
            data=output,
        )

    # executor: shows tool calls and results
    if node_name == "executor":
        tool_results = output.get("tool_results", [])
        errors = output.get("errors", [])
        parts = []
        for r in tool_results:
            status = "✓" if r.success else "✗"
            parts.append(f"  {status} {r.tool_name}" + (f": {r.data[:120]}" if r.data else ""))
        if errors:
            for e in errors:
                parts.append(f"  ⚠ {e[:120]}")

        return StreamEvent(
            node=node_name,
            kind="tool_result" if tool_results else "error",
            content="\n".join(parts) if parts else "Executing...",
            data=output,
        )

    # chat: conversation response
    if node_name == "chat":
        raw = output.get("final_output", "")
        if not raw:
            msgs = output.get("messages", [])
            raw = str(msgs[-1].get("content", "")) if msgs else ""
        return StreamEvent(
            node=node_name,
            kind="interim",
            content=str(raw),
            data=output,
        )

    # output: final result
    if node_name == "output":
        return StreamEvent(
            node=node_name,
            kind="end",
            content=output.get("final_output", "") or "Done.",
            data=output,
        )

    return StreamEvent(node=node_name, kind="interim", content=str(output), data=output)
