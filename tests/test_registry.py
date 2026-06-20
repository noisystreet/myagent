"""Tests for ToolRegistry."""

import pytest

from src.core.registry import ToolRegistry
from src.core.state import ToolResult
from src.tools import ALL_TOOLS


def dummy_tool(path: str, offset: int = 1) -> ToolResult:
    """A dummy tool for testing."""
    return ToolResult("dummy_tool", True, data=f"{path}:{offset}")


def test_registry_register_and_execute():
    """Register a tool and execute it."""
    reg = ToolRegistry()
    reg.register(dummy_tool)

    assert "dummy_tool" in reg.known_tools
    result = reg.execute("dummy_tool", {"path": "test.py", "offset": 5})
    assert result.success
    assert "test.py:5" in result.data


def test_registry_execute_unknown_tool():
    """Executing an unknown tool returns an error."""
    reg = ToolRegistry()
    result = reg.execute("missing", {})
    assert not result.success
    assert "Unknown tool" in result.error


def test_registry_from_list():
    """Initialize registry with a list of tools."""
    reg = ToolRegistry([dummy_tool])
    assert reg.known_tools == {"dummy_tool"}


def test_registry_tool_definitions():
    """Tool definitions are OpenAI-compatible."""
    reg = ToolRegistry([dummy_tool])
    defs = reg.tool_definitions
    assert len(defs) == 1
    assert defs[0]["type"] == "function"
    assert defs[0]["function"]["name"] == "dummy_tool"
    schema = defs[0]["function"]["parameters"]
    assert schema["type"] == "object"
    assert "path" in schema["properties"]
    assert schema["properties"]["path"]["type"] == "string"
    assert schema["properties"]["offset"]["type"] == "integer"
    assert "path" in schema["required"]
    assert "offset" not in schema["required"]


def test_registry_custom_name():
    """Register with a custom name."""
    reg = ToolRegistry()
    reg.register(dummy_tool, name="my_dummy")
    assert "my_dummy" in reg.known_tools


def test_registry_with_real_tools():
    """Verify real tools produce valid schemas."""
    reg = ToolRegistry(ALL_TOOLS)
    assert reg.known_tools == {"read_file", "write_file", "edit_file", "run_command"}

    defs = reg.tool_definitions
    names = {d["function"]["name"] for d in defs}
    assert names == reg.known_tools

    # Verify read_file schema
    rf = next(d for d in defs if d["function"]["name"] == "read_file")
    props = rf["function"]["parameters"]["properties"]
    assert "path" in props
    assert "offset" in props
    assert "limit" in props


@pytest.mark.parametrize("tool_name", ["read_file", "write_file", "edit_file", "run_command"])
def test_registry_execute_real_tools(tool_name):
    """Each real tool can be looked up and has a callable function."""
    reg = ToolRegistry(ALL_TOOLS)
    info = reg.get(tool_name)
    assert info is not None
    assert info.name == tool_name
    assert callable(info.func)
