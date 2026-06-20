"""Shell command execution tool."""

import os
import signal
import subprocess

from ..core.state import ToolResult

MAX_OUTPUT_SIZE = 10_000
MAX_TIMEOUT = 30


def run_command(command: str, cwd: str | None = None, timeout: int = MAX_TIMEOUT) -> ToolResult:
    """Run a shell command in a subprocess.

    Args:
        command: Shell command string.
        cwd: Working directory (defaults to current).
        timeout: Max seconds before killing the process.
    """
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,  # isolate in a new process group
        )
    except Exception as e:
        return ToolResult(
            "run_command",
            False,
            error={"type": "execution_failed", "message": f"Execution failed: {e}"},
        )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        stdout, stderr = proc.communicate()
        return ToolResult(
            "run_command",
            False,
            error={"type": "timeout", "message": f"Command timed out after {timeout}s"},
            data={"command": command, "output": (stdout or "")[:MAX_OUTPUT_SIZE]},
        )

    output = (stdout or "") + (stderr or "")
    truncated = output[:MAX_OUTPUT_SIZE]

    if proc.returncode == 0:
        return ToolResult(
            "run_command",
            True,
            data={"command": command, "return_code": 0, "output": truncated},
        )
    else:
        return ToolResult(
            "run_command",
            False,
            error={"type": "non_zero_exit", "message": f"Exit code {proc.returncode}"},
            data={"command": command, "return_code": proc.returncode, "output": truncated},
        )
