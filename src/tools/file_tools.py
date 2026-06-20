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
            return ToolResult(
                "read_file",
                False,
                error={"type": "file_not_found", "message": f"File not found: {path}"},
            )
        if filepath.stat().st_size > MAX_FILE_SIZE:
            size_msg = f"File too large ({filepath.stat().st_size} bytes), max {MAX_FILE_SIZE}"
            return ToolResult(
                "read_file",
                False,
                error={"type": "file_too_large", "message": size_msg},
            )

        lines = filepath.read_text(encoding="utf-8").splitlines(keepends=True)
        total_lines = len(lines)
        offset = max(offset, 1)
        selected = lines[offset - 1 : offset - 1 + limit]

        result = "".join(selected)
        return ToolResult(
            "read_file",
            True,
            data={
                "file": str(path),
                "total_lines": total_lines,
                "shown_lines": len(selected),
                "content": result,
            },
        )
    except Exception as e:
        return ToolResult("read_file", False, error={"type": "exception", "message": str(e)})


def write_file(path: str, content: str) -> ToolResult:
    """Write content to a file (overwrites if exists)."""
    try:
        filepath = Path(path).resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        return ToolResult(
            "write_file",
            True,
            data={"file": str(path), "bytes_written": len(content)},
        )
    except Exception as e:
        return ToolResult("write_file", False, error={"type": "exception", "message": str(e)})


def edit_file(path: str, old_str: str, new_str: str) -> ToolResult:
    """Search-and-replace edit in a file.

    Replaces the FIRST occurrence of old_str with new_str.
    """
    try:
        filepath = Path(path).resolve()
        if not filepath.exists():
            return ToolResult(
                "edit_file",
                False,
                error={"type": "file_not_found", "message": f"File not found: {path}"},
            )

        content = filepath.read_text(encoding="utf-8")
        if old_str not in content:
            return ToolResult(
                "edit_file",
                False,
                error={
                    "type": "string_not_found",
                    "message": f"String to replace not found in {path}",
                },
            )

        new_content = content.replace(old_str, new_str, 1)
        filepath.write_text(new_content, encoding="utf-8")
        return ToolResult(
            "edit_file",
            True,
            data={"file": str(path), "replacements": 1},
        )
    except Exception as e:
        return ToolResult("edit_file", False, error={"type": "exception", "message": str(e)})
