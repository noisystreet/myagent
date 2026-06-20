"""Streaming execution helpers for the LangGraph agent.

Provides node-level streaming via graph.stream(), yielding display-ready
events for real-time output in the interactive CLI.
"""

import logging
import sys
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any

from ..core.state import CodingAgentState, tool_result_to_text

logger = logging.getLogger(__name__)

_SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Spinner:
    """A simple CLI spinner that runs in a background thread.

    Usage:
        with Spinner("Thinking..."):
            time.sleep(3)
    """

    def __init__(self, message: str = ""):
        self._message = message
        self._running = False
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        """Start the spinner in a daemon thread."""
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the spinner and clear the line."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    def _spin(self):
        """Print spinner characters in a loop."""
        idx = 0
        while self._running:
            char = _SPINNER_CHARS[idx % len(_SPINNER_CHARS)]
            msg = f"\r{char} {self._message}" if self._message else f"\r{char}"
            sys.stdout.write(msg)
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)


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
    handler = _NODE_HANDLERS.get(node_name, _default_handler)
    return handler(node_name, output)


def _handle_intent_router(node_name: str, output: dict) -> StreamEvent:
    """Build event for the intent_router node."""
    mode = output.get("mode", "?")
    return StreamEvent(
        node=node_name,
        kind="start",
        content=f"Routing: {mode} mode",
        data=output,
    )


def _handle_planner(node_name: str, output: dict) -> StreamEvent:
    """Build event for the planner node."""
    plan = output.get("plan", [])
    plan_str = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan))
    return StreamEvent(
        node=node_name,
        kind="interim",
        content=f"Plan ({len(plan)} steps):\n{plan_str}" if plan else "No plan generated",
        data=output,
    )


def _handle_executor(node_name: str, output: dict) -> StreamEvent:
    """Build event for the executor node — only shows current step result."""
    tool_results = output.get("tool_results", [])

    if not tool_results:
        return StreamEvent(node=node_name, kind="interim", content="Executing...", data=output)

    # Only display the latest result (current step), not accumulated history
    latest = tool_results[-1]
    success = latest.get("success") if isinstance(latest, dict) else latest.success
    name = latest.get("tool_name", "?") if isinstance(latest, dict) else latest.tool_name
    text = tool_result_to_text(latest)
    status = "✓" if success else "✗"
    content = f"{status} {name}" + (f": {text[:120]}" if text else "")

    return StreamEvent(
        node=node_name, kind="tool_result" if success else "error", content=content, data=output
    )


def _handle_chat(node_name: str, output: dict) -> StreamEvent:
    """Build event for the chat node."""
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


def _handle_output(node_name: str, output: dict) -> StreamEvent:
    """Build event for the output node."""
    return StreamEvent(
        node=node_name,
        kind="end",
        content=output.get("final_output", "") or "Done.",
        data=output,
    )


def _handle_reflector(node_name: str, output: dict) -> StreamEvent:
    """Build event for the reflector node — concise single-line status."""
    reflections = output.get("reflections", [])
    if not reflections:
        return StreamEvent(node=node_name, kind="interim", content="Reflecting...", data=output)

    latest = reflections[-1]
    verdict = latest.get("verdict", "?")

    _verdict_icons = {
        "continue": "→",
        "retry": "↻",
        "replan": "⟳",
        "done": "✓",
        "interrupt": "⚠",
    }
    icon = _verdict_icons.get(verdict, "?")
    kind_map = {"done": "end", "retry": "error", "replan": "start"}
    event_kind = kind_map.get(verdict, "interim")

    # Single line: icon + verdict + optional next step for continue
    content = f"{icon} {verdict.upper()}"
    if verdict == "continue":
        next_idx = output.get("current_step", 0)
        plan = output.get("plan", [])
        if plan and next_idx < len(plan):
            content += f" → {_format_plan_step(plan[next_idx])}"

    return StreamEvent(node=node_name, kind=event_kind, content=content, data=output)


def _format_plan_step(step: str | dict) -> str:
    """Format a single plan step for display."""
    if isinstance(step, dict):
        action = step.get("action", "?")
        desc = step.get("description", "")
        return f"{action}{f' — {desc}' if desc else ''}"
    return str(step)


def _default_handler(node_name: str, output: dict) -> StreamEvent:
    """Fallback handler for unknown nodes."""
    return StreamEvent(node=node_name, kind="interim", content=str(output), data=output)


_NODE_HANDLERS: dict[str, Callable[[str, dict], StreamEvent]] = {
    "intent_router": _handle_intent_router,
    "planner": _handle_planner,
    "executor": _handle_executor,
    "reflector": _handle_reflector,
    "chat": _handle_chat,
    "output": _handle_output,
}
