"""File manipulation tools: read, write, edit, delete."""

from pathlib import Path

from ..core.state import ToolResult

MAX_FILE_SIZE = 100_000  # 100KB limit for reads


def read_file(path: str, offset: int = 1, limit: int = 500) -> ToolResult:
    """Read a file with optional line range.

    Args:
        path: Absolute or relative file path.
        offset: Starting line (1-based).
        limit: Max lines to read.
    """
    try:
        filepath = Path(path).resolve()
        if not filepath.exists():
            return ToolResult("read_file", False, error=f"File not found: {path}")
        if filepath.stat().st_size > MAX_FILE_SIZE:
            return ToolResult(
                "read_file",
                False,
                error=f"File too large ({filepath.stat().st_size} bytes), max {MAX_FILE_SIZE}",
            )

        lines = filepath.read_text(encoding="utf-8").splitlines(keepends=True)
        total_lines = len(lines)
        offset = max(offset, 1)
        selected = lines[offset - 1 : offset - 1 + limit]

        result = "".join(selected)
        summary = (
            f"File: {path} ({total_lines} lines total, showing {len(selected)} lines)\n" + result
        )
        if offset + limit < total_lines:
            summary += f"\n... ({total_lines - offset - limit + 1} more lines)"
        return ToolResult("read_file", True, data=summary)
    except Exception as e:
        return ToolResult("read_file", False, error=str(e))


def write_file(path: str, content: str) -> ToolResult:
    """Write content to a file (overwrites if exists)."""
    try:
        filepath = Path(path).resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        return ToolResult(
            "write_file",
            True,
            data=f"Written {len(content)} bytes to {path}",
        )
    except Exception as e:
        return ToolResult("write_file", False, error=str(e))


def edit_file(path: str, old_str: str, new_str: str) -> ToolResult:
    """Search-and-replace edit in a file.

    Replaces the FIRST occurrence of old_str with new_str.
    """
    try:
        filepath = Path(path).resolve()
        if not filepath.exists():
            return ToolResult("edit_file", False, error=f"File not found: {path}")

        content = filepath.read_text(encoding="utf-8")
        if old_str not in content:
            return ToolResult(
                "edit_file",
                False,
                error=f"String to replace not found in {path}",
            )

        new_content = content.replace(old_str, new_str, 1)
        filepath.write_text(new_content, encoding="utf-8")
        return ToolResult(
            "edit_file",
            True,
            data=f"Replaced 1 occurrence in {path}",
        )
    except Exception as e:
        return ToolResult("edit_file", False, error=str(e))
