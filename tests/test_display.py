"""Tests for the rich display module."""

from src.core.display import (
    _is_python_code,
    _looks_like_code,
    _render_markdown,
    _render_smart,
    display_banner,
    display_stream_event,
    display_turn_summary,
)
from src.core.stream import StreamEvent


class TestLooksLikeCode:
    def test_single_line(self):
        assert not _looks_like_code("hello")

    def test_multi_line_indented(self):
        assert _looks_like_code("  foo\n  bar\n  baz")

    def test_multi_line_python_keywords(self):
        assert _looks_like_code("import os\nimport sys\nprint('hi')")

    def test_multi_line_plain_text(self):
        assert not _looks_like_code("hello world\nhow are you\nfine thanks")

    def test_shell_output(self):
        assert _looks_like_code("$ ls\n$ cd\n$ git status")


class TestIsPythonCode:
    def test_import_stmt(self):
        assert _is_python_code("import os")

    def test_def_stmt(self):
        assert _is_python_code("def foo():")

    def test_class_stmt(self):
        assert _is_python_code("class Foo:")

    def test_for_loop(self):
        assert _is_python_code("for x in range(10):")

    def test_not_python(self):
        assert not _is_python_code("hello world")

    def test_shell_output_is_not_python(self):
        assert not _is_python_code("$ echo hi")


class TestRenderMarkdown:
    def test_plain_text(self):
        result = _render_markdown("hello")
        assert result is not None

    def test_empty_string(self):
        result = _render_markdown("")
        assert result is not None


class TestRenderSmart:
    def test_plain_text(self):
        result = _render_smart("hello world")
        assert result is not None

    def test_python_code(self):
        result = _render_smart("import os\nprint('hi')")
        assert result is not None


class TestDisplayStreamEventSmoke:
    """Smoke tests — verify display functions don't crash."""

    def test_start_event(self):
        event = StreamEvent(node="intent_router", kind="start", content="Routing: task mode")
        display_stream_event(event)

    def test_chat_event(self):
        event = StreamEvent(node="chat", kind="interim", content="Hello!")
        display_stream_event(event)

    def test_planner_event(self):
        event = StreamEvent(
            node="planner", kind="interim", content="Plan (1 steps):\n  1. Do thing"
        )
        display_stream_event(event)

    def test_tool_result_event(self):
        event = StreamEvent(node="executor", kind="tool_result", content="  ✓ run_command")
        display_stream_event(event)

    def test_error_event(self):
        event = StreamEvent(node="executor", kind="error", content="Something broke")
        display_stream_event(event)

    def test_end_event(self):
        event = StreamEvent(node="output", kind="end", content="Done.")
        display_stream_event(event)

    def test_empty_content(self):
        event = StreamEvent(node="test", kind="interim", content="")
        display_stream_event(event)


class TestDisplayHelpersSmoke:
    """Smoke tests for other display functions."""

    def test_display_banner(self):
        display_banner()

    def test_display_turn_summary(self):
        display_turn_summary(1, 2, "chat", 0)

    def test_display_turn_summary_with_errors(self):
        display_turn_summary(3, 5, "task", 2)
