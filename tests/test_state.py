"""Tests for state definitions."""

from src.core.state import CodingAgentState, ToolResult
from langchain_core.messages import HumanMessage


class TestToolResult:
    def test_create(self):
        r = ToolResult("test", True, data="ok")
        assert r.tool_name == "test"
        assert r.success is True
        assert r.data == "ok"

    def test_create_failure(self):
        r = ToolResult("test", False, error="something went wrong")
        assert not r.success
        assert "went wrong" in r.error

    def test_dict_serializable(self):
        r = ToolResult("x", True, data="hello", duration_ms=1.5)
        import json
        d = json.dumps(dict(r))
        assert '"tool_name"' in d
        assert '"data"' in d


class TestCodingAgentState:
    def test_minimal_state(self):
        state: CodingAgentState = {
            "messages": [HumanMessage(content="hi")],
            "plan": [],
            "current_step": 0,
            "max_steps": 20,
            "workspace": "/tmp",
            "tool_results": [],
            "errors": [],
            "retry_count": 0,
            "step_attempts": 0,
            "mode": "task",
            "next_action": "intent_router",
            "final_output": None,
        }
        assert state["mode"] == "task"
        assert state["next_action"] == "intent_router"
        assert len(state["messages"]) == 1
