"""Executor node: executes the current plan step via tool calls."""

import json
import re
import time
from pathlib import Path
from typing import Any

from ..core.state import CodingAgentState, ToolResult
from ..llm.client import LLMClient
from ..tools.command_tools import run_command
from ..tools.file_tools import edit_file, read_file, write_file

EXECUTOR_SYSTEM_PROMPT = """\
You are a coding agent executor. Execute the current step by calling exactly ONE of these tools:

read_file(path, offset?, limit?)
write_file(path, content)
edit_file(path, old_str, new_str)
run_command(command, cwd?, timeout?)

Rules:
- Use relative paths for files (they are resolved automatically).
- For run_command, the working directory is the project root.
- Respond with the tool call ONLY, nothing else.

Examples:
write_file(path="hello.py", content="print('hello')")
read_file(path="src/main.py")
run_command(command="python hello.py")
"""

# AI-generated: OpenAI/DeepSeek compatible tool definitions for bind_tools
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "offset": {"type": "integer", "description": "Start line (1-based, optional)"},
                    "limit": {"type": "integer", "description": "Max lines to read (optional)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing old_str with new_str.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "old_str": {"type": "string", "description": "Text to replace"},
                    "new_str": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the project workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (optional)",
                    },
                    "timeout": {"type": "integer", "description": "Timeout in seconds (optional)"},
                },
                "required": ["command"],
            },
        },
    },
]


def executor_node(state: CodingAgentState, llm: LLMClient) -> dict:
    """Execute the current step using LLM-selected tools."""
    step = state["plan"][state["current_step"]]
    workspace = state["workspace"]

    response = llm.invoke(
        prompt=f"Current step: {step}\n\nRespond with the tool call only.",
        system=EXECUTOR_SYSTEM_PROMPT,
        tools=TOOL_DEFINITIONS,
    )

    results: list[ToolResult] = []

    # Strategy 1: Native tool_calls (DeepSeek / OpenAI bind_tools format)
    tool_call = _extract_tool_call_from_response(response)
    if tool_call:
        results.append(_resolve_and_run(tool_call, workspace))
    else:
        # Strategy 2: Text-based parsing (fallback)
        tool_call = _parse_tool_call(str(response).strip())
        if tool_call:
            results.append(_resolve_and_run(tool_call, workspace))
        else:
            results.append(
                ToolResult(
                    "parse", False, error=f"Could not parse tool call from: {str(response)[:200]}"
                )
            )

    errors = [r.error for r in results if not r.success]

    return {
        "tool_results": state.get("tool_results", []) + results,
        "errors": errors,
        "step_attempts": state.get("step_attempts", 0) + 1,
        "current_step": state["current_step"] + (1 if not errors else 0),
        "next_action": "output"
        if errors or state["current_step"] >= len(state["plan"]) - 1
        else "executor",
    }


def _resolve_and_run(tool_call: tuple[str, dict], workspace: str) -> ToolResult:
    """Resolve paths and execute a parsed tool call."""
    tool_name, args = tool_call
    for key in ("path", "cwd"):
        if key in args and args[key] and not args[key].startswith("/"):
            args[key] = str(Path(workspace) / args[key])
    if tool_name == "run_command" and ("cwd" not in args or not args["cwd"]):
        args["cwd"] = workspace
    return _run_tool(tool_name, args)


def _parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Parse a tool call from LLM response text.

    Handles multiple formats:
      - JSON: {"tool": "read_file", "args": {"path": "..."}}
      - Function call: write_file(path="hello.py", content="...")
      - Free text with tool name and arguments mentioned
    """
    # Strategy 1: Try JSON format
    result = _try_parse_json(text)
    if result:
        return result

    # Strategy 2: Try function-call format
    result = _try_parse_function_call(text)
    if result:
        return result

    # Strategy 3: Try to infer from free text
    result = _try_infer_from_text(text)
    if result:
        return result

    return None


KNOWN_TOOLS = {"read_file", "write_file", "edit_file", "run_command"}


def _parse_function_args(func: dict) -> tuple[str, dict]:
    """Parse function name and arguments from a DeepSeek/OpenAI function dict."""
    name = func.get("name", "")
    args_str = func.get("arguments", "{}")
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except json.JSONDecodeError:
        args = {}
    return name, args


def _extract_from_aimessage(response: Any) -> tuple[str, dict] | None:
    """Extract tool call from LangChain AIMessage.tool_calls."""
    if not (hasattr(response, "tool_calls") and response.tool_calls):
        return None
    for tc in response.tool_calls:
        if isinstance(tc, dict):
            name = tc.get("name", "")
            args = tc.get("args", {})
            if name in KNOWN_TOOLS:
                return name, args
    return None


def _extract_from_dict(response: dict) -> tuple[str, dict] | None:
    """Extract tool call from raw dict (DeepSeek API response)."""
    # {"tool_calls": [{"function": {...}}]}
    tcs = response.get("tool_calls")
    if isinstance(tcs, list):
        for tc in tcs:
            if isinstance(tc, dict) and "function" in tc:
                name, args = _parse_function_args(tc["function"])
                if name in KNOWN_TOOLS:
                    return name, args

    # {"function_call": {...}}
    fc = response.get("function_call")
    if isinstance(fc, dict):
        name, args = _parse_function_args(fc)
        if name in KNOWN_TOOLS:
            return name, args

    return None


def _extract_tool_call_from_response(response: Any) -> tuple[str, dict] | None:
    """Extract tool call from LangChain AIMessage or raw dict.

    Supports DeepSeek / OpenAI native tool_calls format:
      - AIMessage with .tool_calls [{"name": ..., "args": ...}]
      - Dict with {"tool_calls": [{"function": {"name": ..., "arguments": ...}}]}
      - Dict with {"function_call": {"name": ..., "arguments": ...}}
    """
    # AI-generated
    result = _extract_from_aimessage(response)
    if result:
        return result

    if isinstance(response, dict):
        return _extract_from_dict(response)

    return None


def _try_parse_json(text: str) -> tuple[str, dict] | None:
    """Try to parse as JSON format."""
    for candidate in _collect_json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        # First try DeepSeek / OpenAI native formats
        result = _extract_tool_call_from_response(parsed)
        if result:
            return result
        # Then try legacy formats
        result = _parse_json_as_tool_call(parsed)
        if result:
            return result
    return None


def _collect_json_candidates(text: str) -> list[str]:
    """Extract JSON substrings from text (code block or bare JSON)."""
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        return [json_match.group(1)]
    candidates = []
    for delim in ("{", "["):
        start = text.find(delim)
        if start >= 0:
            end = text.rfind("}" if delim == "{" else "]")
            if end > start:
                candidates.append(text[start : end + 1])
    return candidates


def _parse_json_as_tool_call(parsed: dict) -> tuple[str, dict] | None:
    """Extract a tool call from a parsed JSON dict."""
    # {tool: "...", args: {...}}
    if "tool" in parsed and "args" in parsed:
        return parsed["tool"].lower(), parsed["args"]
    # {action: "call_tool", tool: "...", args: {...}}
    if (
        isinstance(parsed.get("action"), str)
        and parsed["action"] == "call_tool"
        and "tool" in parsed
    ):
        return parsed["tool"].lower(), parsed.get("args", {})
    # Single key-value where key is a tool name
    for key in KNOWN_TOOLS:
        if key in parsed:
            return key, parsed[key] if isinstance(parsed[key], dict) else {}
    return None


def _try_parse_function_call(text: str) -> tuple[str, dict] | None:
    """Try to parse as function-call format:
    write_file(path="hello.py", content="print('hello')")
    """
    for tool in KNOWN_TOOLS:
        # Match: tool_name(key1=value1, key2=value2, ...)
        pattern = re.compile(rf"{re.escape(tool)}\s*\((.+?)\)\s*$", re.DOTALL)
        match = pattern.search(text)
        if not match:
            pattern = re.compile(rf"{re.escape(tool)}\s*\((.+?)\)", re.DOTALL)
            match = pattern.search(text)

        if match:
            args_str = match.group(1).strip()
            args = _parse_keyword_args(args_str)
            if args is not None:
                return tool, args

    return None


def _parse_keyword_args(args_str: str) -> dict | None:
    """Parse keyword arguments from a function call string.

    write_file(path="hello.py", content="print('hello')")
    -> {"path": "hello.py", "content": "print('hello')"}
    """
    args = {}
    # Match key="value" or key='value' or key=value (no quotes)
    pattern = re.compile(
        r"""(\w+)\s*=\s*(?:"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'|(\S+))"""
    )
    for m in pattern.finditer(args_str):
        key = m.group(1)
        # group 2: double-quoted, group 3: single-quoted, group 4: unquoted
        value = m.group(2) or m.group(3) or m.group(4)
        args[key] = value

    return args if args else None


def _try_infer_from_text(text: str) -> tuple[str, dict] | None:
    """Try to infer a tool call from free text."""
    text_lower = text.lower()

    for tool in KNOWN_TOOLS:
        if tool in text_lower:
            args = {}
            # Try to extract file path after "path:" or "file:" mentions
            path_match = re.search(r'(?:path|file)[:\s]+["\']?([^"\'\s,\)]+)', text)
            if path_match:
                args["path"] = path_match.group(1)
            # Try to extract command after "command:" or after "run "
            cmd_match = re.search(r'(?:command|run)[:\s]+["\']?([^"\'\n]+)', text)
            if cmd_match:
                args["command"] = cmd_match.group(1)
            # Try to extract content
            content_match = re.search(r'content[:\s]+["\'](.+?)["\']', text, re.DOTALL)
            if content_match:
                args["content"] = content_match.group(1)
            return tool, args

    return None


def _run_tool(tool_name: str, args: dict) -> ToolResult:
    """Dispatch to the correct tool function."""
    start = time.perf_counter()

    try:
        if tool_name == "read_file":
            result = read_file(**args)
        elif tool_name == "write_file":
            result = write_file(**args)
        elif tool_name == "edit_file":
            result = edit_file(**args)
        elif tool_name == "run_command":
            result = run_command(**args)
        else:
            result = ToolResult(tool_name, False, error=f"Unknown tool: {tool_name}")
    except TypeError as e:
        result = ToolResult(tool_name, False, error=f"Invalid arguments: {e}")
    except Exception as e:
        result = ToolResult(tool_name, False, error=str(e))

    result.duration_ms = (time.perf_counter() - start) * 1000
    return result
