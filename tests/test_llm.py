"""Tests for LLM client JSON extraction."""

from src.llm.client import _extract_json, _describe_type
from typing import Optional


class TestExtractJson:
    def test_json_block(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        assert _extract_json(text) == '{"key": "value"}'

    def test_json_block_no_lang(self):
        text = "```\n{\"key\": \"value\"}\n```"
        assert _extract_json(text) == '{"key": "value"}'

    def test_bare_json(self):
        text = 'Some text {"key": "value"} more text'
        assert _extract_json(text) == '{"key": "value"}'

    def test_no_json(self):
        text = "just plain text"
        assert _extract_json(text) == "just plain text"


class TestDescribeType:
    def test_str(self):
        assert _describe_type(str) == "string"

    def test_int(self):
        assert _describe_type(int) == "integer"

    def test_list_of_str(self):
        assert "list of string" in _describe_type(list[str])

    def test_optional(self):
        t = _describe_type(Optional[str])
        # Optional[str] is Union[str, None] at runtime
        assert isinstance(t, str)
