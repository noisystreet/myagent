"""Tool registry: centralized tool definition and dispatch."""

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from .state import ToolResult

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


_OPTIONAL_ARG_COUNT = 2


def _get_json_type(py_type: Any) -> str:
    """Map a Python type hint to a JSON Schema type string."""
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        args = getattr(py_type, "__args__", ())
        if len(args) == _OPTIONAL_ARG_COUNT and type(None) in args:
            for arg in args:
                if arg is not type(None):
                    return _TYPE_MAP.get(arg, "string")
        return "string"
    return _TYPE_MAP.get(py_type, "string")


def _build_schema_from_signature(func: Callable[..., Any]) -> dict:
    """Build OpenAI-compatible JSON Schema from function signature."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties: dict[str, dict] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        py_type = hints.get(param_name, str)
        properties[param_name] = {"type": _get_json_type(py_type)}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


@dataclass
class ToolInfo:
    """Metadata for a registered tool."""

    name: str
    description: str
    schema: dict
    func: Callable[..., Any]


class ToolRegistry:
    """Centralized registry for tool definitions and execution."""

    def __init__(self, tools: list[Callable[..., Any]] | None = None):
        self._tools: dict[str, ToolInfo] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register a tool and auto-generate schema from its signature."""
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip()
        schema = _build_schema_from_signature(func)
        self._tools[tool_name] = ToolInfo(
            name=tool_name,
            description=tool_desc,
            schema=schema,
            func=func,
        )

    def get(self, name: str) -> ToolInfo | None:
        """Get ToolInfo by name."""
        return self._tools.get(name)

    @property
    def known_tools(self) -> set[str]:
        """Return set of registered tool names."""
        return set(self._tools.keys())

    @property
    def tool_definitions(self) -> list[dict]:
        """Return OpenAI/DeepSeek compatible tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": info.name,
                    "description": info.description,
                    "parameters": info.schema,
                },
            }
            for info in self._tools.values()
        ]

    def execute(self, name: str, args: dict) -> ToolResult:
        """Execute a registered tool by name with given arguments."""
        info = self._tools.get(name)
        if info is None:
            return ToolResult(name, False, error=f"Unknown tool: {name}")
        start = time.perf_counter()
        try:
            result = info.func(**args)
        except TypeError as e:
            result = ToolResult(name, False, error=f"Invalid arguments: {e}")
        except Exception as e:
            result = ToolResult(name, False, error=str(e))
        result.duration_ms = (time.perf_counter() - start) * 1000
        return result
