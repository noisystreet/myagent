"""Tests for executor tool call parser."""

from src.nodes.executor import _parse_tool_call


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
