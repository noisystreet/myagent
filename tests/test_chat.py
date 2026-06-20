"""Tests for chat node."""

from src.llm.schemas import ChatResponse, CodeBlock
from src.nodes.chat import _format_chat_response


class TestFormatChatResponse:
    def test_explanation_only(self):
        response = ChatResponse(explanation="Hello world")
        result = _format_chat_response(response)
        assert result == "Hello world"

    def test_with_code_block(self):
        response = ChatResponse(
            explanation="Here is a hello function:",
            code_blocks=[
                CodeBlock(language="python", code="def hello():\n    print('hi')")
            ],
        )
        result = _format_chat_response(response)
        assert "Here is a hello function:" in result
        assert "```python" in result
        assert "def hello():" in result
        assert "print('hi')" in result

    def test_with_references(self):
        response = ChatResponse(
            explanation="See the documentation.",
            references=["https://python.org", "src/main.py"],
        )
        result = _format_chat_response(response)
        assert "See the documentation." in result
        assert "**References:**" in result
        assert "- https://python.org" in result
        assert "- src/main.py" in result

    def test_with_caption(self):
        response = ChatResponse(
            explanation="Example:",
            code_blocks=[
                CodeBlock(language="bash", code="echo hi", caption="Run this")
            ],
        )
        result = _format_chat_response(response)
        assert "**Run this**" in result
        assert "```bash" in result
        assert "echo hi" in result

    def test_multiple_code_blocks(self):
        response = ChatResponse(
            explanation="Two examples:",
            code_blocks=[
                CodeBlock(language="python", code="x = 1"),
                CodeBlock(language="javascript", code="const x = 1;"),
            ],
        )
        result = _format_chat_response(response)
        assert "```python" in result
        assert "```javascript" in result
        assert "x = 1" in result
        assert "const x = 1;" in result

    def test_empty_response(self):
        response = ChatResponse(explanation="")
        result = _format_chat_response(response)
        assert result == ""

    def test_no_language(self):
        response = ChatResponse(
            explanation="Some code:",
            code_blocks=[CodeBlock(code="plain text")],
        )
        result = _format_chat_response(response)
        assert "```\nplain text\n```" in result
