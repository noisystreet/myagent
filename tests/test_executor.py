"""Tests for executor tool call parser."""

from src.nodes.executor import (
    _extract_tool_call_from_response,
    _parse_tool_call,
)


class TestExtractToolCall:
    """Test DeepSeek / OpenAI native tool call extraction."""

    def test_deepseek_tool_calls_format(self):
        """DeepSeek API response with tool_calls array."""
        response = {
            "tool_calls": [
                {
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "src/main.py"}',
                    }
                }
            ]
        }
        result = _extract_tool_call_from_response(response)
        assert result is not None
        name, args = result
        assert name == "read_file"
        assert args["path"] == "src/main.py"

    def test_openai_function_call_format(self):
        """OpenAI legacy function_call dict."""
        response = {
            "function_call": {
                "name": "write_file",
                "arguments": '{"path": "hello.py", "content": "hi"}',
            }
        }
        result = _extract_tool_call_from_response(response)
        assert result is not None
        name, args = result
        assert name == "write_file"
        assert args["path"] == "hello.py"

    def test_langchain_aimessage_tool_calls(self):
        """LangChain AIMessage with tool_calls attribute."""
        class FakeAIMessage:
            tool_calls = [
                {"name": "run_command", "args": {"command": "ls"}},
            ]

        result = _extract_tool_call_from_response(FakeAIMessage())
        assert result is not None
        name, args = result
        assert name == "run_command"
        assert args["command"] == "ls"

    def test_unknown_tool_skipped(self):
        """Unknown tool names should be ignored."""
        response = {
            "tool_calls": [
                {"function": {"name": "unknown_tool", "arguments": "{}"}}
            ]
        }
        result = _extract_tool_call_from_response(response)
        assert result is None

    def test_no_tool_calls_returns_none(self):
        """Plain text or empty dict returns None."""
        assert _extract_tool_call_from_response("plain text") is None
        assert _extract_tool_call_from_response({}) is None
        assert _extract_tool_call_from_response(None) is None


class TestParseToolCall:
    def test_function_call_format(self):
        r = _parse_tool_call(
            'write_file(path="hello.py", content="print(\'hi\')")'
        )
        assert r is not None
        name, args = r
        assert name == "write_file"
        assert args["path"] == "hello.py"

    def test_run_command(self):
        r = _parse_tool_call('run_command(command="python hello.py")')
        assert r is not None
        name, args = r
        assert name == "run_command"
        assert args["command"] == "python hello.py"

    def test_read_file(self):
        r = _parse_tool_call('read_file(path="src/main.py")')
        assert r is not None
        name, args = r
        assert name == "read_file"
        assert args["path"] == "src/main.py"

    def test_json_format(self):
        r = _parse_tool_call(
            '{"tool": "read_file", "args": {"path": "test.py"}}'
        )
        assert r is not None
        assert r[0] == "read_file"

    def test_free_text_inference(self):
        r = _parse_tool_call("We need to read the file main.py using read_file")
        assert r is not None
        name, args = r
        assert name == "read_file"
        assert "main.py" in args.get("path", "")

    def test_done(self):
        """DONE should result in no parseable tool call."""
        r = _parse_tool_call("DONE")
        assert r is None

    def test_gibberish(self):
        r = _parse_tool_call("asdfghjkl")
        assert r is None
