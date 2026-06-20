"""Rich-based terminal output rendering.

Provides styled display for streaming events, final results,
and code/markdown content using the `rich` library.
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from .stream import StreamEvent

# Global console — single instance for proper output handling
_console = Console(highlight=False)

# Event kind → style mapping
_KIND_STYLES = {
    "start": "bold cyan",
    "interim": "default",
    "tool_result": "green",
    "error": "bold red",
    "end": "bold white",
}

_KIND_ICONS = {
    "start": "●",
    "interim": "▸",
    "tool_result": "✓",
    "error": "✗",
    "end": "■",
}


def display_stream_event(event: StreamEvent) -> None:
    """Render a streaming event to the terminal with rich styling."""
    node = event.node
    kind = event.kind
    content = event.content
    style = _KIND_STYLES.get(kind, "default")
    icon = _KIND_ICONS.get(kind, "•")

    if not content:
        return

    # Skip noisy routing events — internal detail, not useful to users
    if node == "intent_router":
        return

    if kind == "start":
        _console.print()
        _console.print(Panel(content, style=style, title=f"[{node}]", title_align="left"))

    elif kind == "interim":
        if node == "chat":
            # Chat messages — render as markdown
            _console.print()
            _console.print(_render_markdown(content))
        elif node == "planner":
            # Plan steps
            _console.print()
            _console.print(content, style=style)
        else:
            _console.print()
            _console.print(content, style=style)

    elif kind == "tool_result":
        _console.print(content, style=style)

    elif kind == "error":
        _console.print(f"\n{icon} [{node}] Error:", style=style)
        _console.print(content, style="red")

    elif kind == "end":
        # Final output — smart render: markdown if text, syntax if code
        _console.print()
        _console.print(Rule(style="dim"))
        _console.print(_render_smart(content))
        _console.print(Rule(style="dim"))


def display_result(state) -> None:
    """Print execution result summary."""
    output = state.get("final_output") or "No output generated."
    _console.print()
    _console.print(Panel(_render_smart(output), title="Result", border_style="green"))

    mode = state.get("mode", "task")
    steps = state.get("step_attempts", 0)
    errors = len(state.get("errors", []))

    table = Table.grid(padding=(0, 2))
    table.add_row("[dim]Mode[/dim]", f"[cyan]{mode}[/cyan]")
    table.add_row("[dim]Steps[/dim]", str(steps))
    table.add_row("[dim]Errors[/dim]", f"[red]{errors}[/red]")
    _console.print()
    _console.print(table)


def display_turn_summary(msg_count: int, msg_pairs: int, mode: str, errors: int) -> None:
    """Print the compact turn summary line."""
    parts = [f"Turn {msg_count}  |  Msgs: {msg_pairs} pairs  |  Mode: {mode}"]
    if errors:
        parts.append(f"  |  [red]⚠ {errors} error(s)[/red]")
    _console.print()
    _console.print(Rule("".join(parts), style="dim"))


def display_banner() -> None:
    """Print the interactive mode welcome banner."""
    _console.print()
    _console.print(
        Panel(
            "[bold]myagent[/bold] — interactive mode\n\n"
            "Type [cyan]exit[/cyan] to quit  |  [cyan]new[/cyan] for a fresh session",
            border_style="cyan",
        )
    )
    _console.print()


def _render_smart(text: str) -> Panel | Markdown | str:
    """Render content as markdown, code block, or plain text."""
    # If it contains markdown syntax, render as markdown first so code blocks
    # inside markdown are properly highlighted by Rich.
    if _looks_like_markdown(text):
        return _render_markdown(text)
    if _looks_like_code(text):
        return Syntax(text, "python" if _is_python_code(text) else "text", theme="monokai")
    return _render_markdown(text)


def _render_markdown(text: str) -> Markdown | str:
    """Render text as markdown, fallback to plain if parsing fails."""
    try:
        md = Markdown(text)
        if md.parsed:
            return md
    except Exception:
        pass
    return text


def _looks_like_markdown(text: str) -> bool:
    """Heuristic: contains markdown indicators like headers, code blocks, or emphasis."""
    markers = ("```", "#", "**", "__", "[", "](", "- ", "> ", "|")
    return any(marker in text for marker in markers)


_MIN_CODE_LINES = 2
_CODE_INDICATORS = (" ", "\t", "import", "def ", "class ", "#", "$")


def _looks_like_code(text: str) -> bool:
    """Heuristic: multi-line with indentation or common CLI patterns."""
    lines = text.splitlines()
    if len(lines) < _MIN_CODE_LINES:
        return False
    indented = sum(1 for line in lines if line.startswith(_CODE_INDICATORS))
    return indented >= len(lines) * 0.3


def _is_python_code(text: str) -> bool:
    """Heuristic: check for Python keywords."""
    keywords = {
        "import ",
        "def ",
        "class ",
        "if ",
        "elif ",
        "else:",
        "for ",
        "while ",
        "return ",
        "try:",
    }
    return any(kw in text for kw in keywords)
