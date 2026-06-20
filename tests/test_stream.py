"""Tests for the streaming execution module."""

import time

from src.core.state import ToolResult
from src.core.stream import Spinner, StreamEvent, _event_from_node  # noqa: PLC2701


class TestStreamEvent:
    """StreamEvent is a simple dataclass — verify construction."""

    def test_minimal_event(self):
        event = StreamEvent(node="test", kind="start")
        assert event.node == "test"
        assert event.kind == "start"
        assert not event.content
        assert event.data == {}

    def test_full_event(self):
        content = "✓ run_command"
        event = StreamEvent(node="executor", kind="tool_result", content=content, data={"test": 1})
        assert event.node == "executor"
        assert event.kind == "tool_result"
        assert event.content == "✓ run_command"
        assert event.data == {"test": 1}


class TestSpinner:
    """Spinner is a background-thread CLI spinner."""

    def test_start_stop(self):
        """Starting and stopping should not raise."""
        spinner = Spinner("testing")
        spinner.start()
        time.sleep(0.3)
        spinner.stop()
        assert not spinner._running

    def test_context_manager(self):
        """Using as context manager should work cleanly."""
        with Spinner("testing"):
            time.sleep(0.2)
        # After exit, spinner should be stopped

    def test_multiple_start_stop(self):
        """Restarting spinner should work."""
        spinner = Spinner("test")
        for _ in range(3):
            spinner.start()
            time.sleep(0.1)
            spinner.stop()
        assert not spinner._running

    def test_empty_message(self):
        """Spinner with empty message should not crash."""
        spinner = Spinner()
        spinner.start()
        time.sleep(0.2)
        spinner.stop()
        assert not spinner._running


class TestEventFromNode:
    """_event_from_node converts raw node output into StreamEvent."""

    def test_intent_router(self):
        output = {"mode": "task", "next_action": "planner"}
        event = _event_from_node("intent_router", output)
        assert event.node == "intent_router"
        assert event.kind == "start"
        assert "task" in event.content

    def test_planner_with_plan(self):
        output = {"plan": ["step1", "step2"]}
        event = _event_from_node("planner", output)
        assert event.node == "planner"
        assert event.kind == "interim"
        assert "2 steps" in event.content

    def test_planner_empty_plan(self):
        output = {"plan": []}
        event = _event_from_node("planner", output)
        assert event.kind == "interim"
        assert "No plan" in event.content

    def test_executor_with_successful_tool(self):
        result = ToolResult("read_file", True, data="file contents")
        output = {"tool_results": [result], "errors": [], "current_step": 1}
        event = _event_from_node("executor", output)
        assert event.node == "executor"
        assert event.kind == "tool_result"
        assert "✓" in event.content

    def test_executor_with_failed_tool(self):
        result = ToolResult("read_file", False, error="not found")
        output = {"tool_results": [result], "errors": ["not found"], "current_step": 1}
        event = _event_from_node("executor", output)
        assert event.node == "executor"
        assert event.kind == "tool_result"
        assert "✗" in event.content

    def test_executor_no_results(self):
        output = {"tool_results": [], "errors": [], "current_step": 0}
        event = _event_from_node("executor", output)
        assert event.kind in {"tool_result", "error"}
        assert event.content

    def test_chat_with_final_output(self):
        output = {"final_output": "Hello!", "messages": []}
        event = _event_from_node("chat", output)
        assert event.node == "chat"
        assert event.kind == "interim"
        assert event.content == "Hello!"

    def test_chat_with_messages_fallback(self):
        output = {"final_output": "", "messages": [{"content": "Hi there"}]}
        event = _event_from_node("chat", output)
        assert event.kind == "interim"
        assert "Hi" in event.content

    def test_output_node(self):
        output = {"final_output": "Task complete"}
        event = _event_from_node("output", output)
        assert event.node == "output"
        assert event.kind == "end"
        assert event.content == "Task complete"

    def test_output_node_no_output(self):
        output = {}
        event = _event_from_node("output", output)
        assert event.kind == "end"
        assert event.content == "Done."

    def test_unknown_node(self):
        output = {"some": "data"}
        event = _event_from_node("unknown_node", output)
        assert event.node == "unknown_node"
        assert event.kind == "interim"
        assert "data" in event.content

    def test_intent_router_invalid_data(self):
        """Should handle missing keys gracefully."""
        event = _event_from_node("intent_router", {})
        assert event.node == "intent_router"
        assert event.kind == "start"

    def test_planner_invalid_data(self):
        """Should handle missing plan key."""
        event = _event_from_node("planner", {})
        assert event.node == "planner"
        assert event.kind == "interim"
