"""Pydantic schemas for structured LLM output."""

from typing import Literal

from pydantic import BaseModel, Field


class PlanSchema(BaseModel):
    """Structured output from the planner node."""

    analysis: str = Field(description="Summary of the task analysis")
    steps: list[str] = Field(description="Ordered execution steps, each a concrete action")
    files: list[str] = Field(
        default_factory=list,
        description="File paths relevant to the task",
    )


class ToolCallSchema(BaseModel):
    """Structured output from the executor node — describes which tool to call."""

    action: Literal["call_tool", "done"] = Field(
        description="Whether to call a tool or mark the step as complete"
    )
    tool: str | None = Field(
        default=None,
        description="Tool name: read_file, write_file, edit_file, or run_command",
    )
    args: dict | None = Field(
        default=None,
        description="Arguments for the tool as a JSON object",
    )
    explanation: str = Field(
        default="",
        description="Brief explanation of what is being done",
    )


class CodeBlock(BaseModel):
    """A single code block with syntax highlighting metadata."""

    language: str | None = Field(
        default=None,
        description=(
            "Programming language for syntax highlighting, e.g. 'python', 'javascript', 'bash'"
        ),
    )
    code: str = Field(description="The code content")
    caption: str | None = Field(
        default=None,
        description="Optional caption or title for the code block",
    )


class ChatResponse(BaseModel):
    """Structured chat response with code blocks and references."""

    explanation: str = Field(
        description="Main explanatory text in markdown format. Keep it concise."
    )
    code_blocks: list[CodeBlock] = Field(
        default_factory=list,
        description="Code blocks to render with syntax highlighting. Separate from explanation.",
    )
    references: list[str] = Field(
        default_factory=list,
        description="Referenced file paths, URLs, or documentation links",
    )
