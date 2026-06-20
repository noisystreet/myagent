#!/usr/bin/env python3
"""Consolidated pre-commit check runner.

Replaces scattered .pre-commit-config.yaml entries with a single
Python script that calls tools directly via subprocess.

Checks performed:
  - ruff lint (with optional --fix)
  - ruff format --check
  - lizard complexity
  - codespell
  - trailing whitespace / missing EOF newline
  - YAML / TOML syntax validation
  - large file detection (>1MB)
  - merge conflict markers

Usage:
    python scripts/pre-commit-check.py          # run all checks
    python scripts/pre-commit-check.py --fix    # auto-fix where possible
    python scripts/pre-commit-check.py --staged # only check staged files

Exit codes:
    0 - all checks passed
    1 - one or more checks failed
"""

# AI-generated

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MAX_FILE_SIZE = 1_024 * 1_024  # 1 MB


def _tool_missing(name: str, install_hint: str) -> int:
    """Print a warning when a required tool is not installed and skip."""
    print(f"  WARNING: {name} not found. {install_hint}")
    return 0


def _run(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command via subprocess and return (rc, stdout, stderr)."""
    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    return result.returncode, result.stdout, result.stderr


def _has_tool(name: str) -> bool:
    """Return True if the named executable is available on PATH."""
    result = subprocess.run(
        ["which", name], capture_output=True, text=True
    )
    return result.returncode == 0


def _get_files(staged_only: bool) -> list[Path]:
    """Return list of files to check."""
    if staged_only:
        rc, out, _ = _run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"])
        if rc != 0:
            return []
        paths = [p for p in out.strip().split("\n") if p]
    else:
        rc, out, _ = _run(["git", "ls-files"])
        if rc != 0:
            return []
        paths = [p for p in out.strip().split("\n") if p]
    return [PROJECT_ROOT / p for p in paths]


def _print_output(stdout: str, stderr: str) -> None:
    """Print captured stdout/stderr if non-empty."""
    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, end="", file=sys.stderr)


def check_ruff_lint(*, fix: bool = False) -> int:
    """Run ruff lint on src/."""
    if not _has_tool("ruff"):
        return _tool_missing("ruff", "pip install ruff")
    cmd = ["ruff", "check", "src/"]
    if fix:
        cmd.append("--fix")
    rc, out, err = _run(cmd)
    _print_output(out, err)
    return rc


def check_ruff_format() -> int:
    """Run ruff format --check on src/."""
    if not _has_tool("ruff"):
        return _tool_missing("ruff", "pip install ruff")
    rc, out, err = _run(["ruff", "format", "--check", "src/"])
    _print_output(out, err)
    return rc


def check_lizard() -> int:
    """Run lizard complexity check on src/."""
    if not _has_tool("lizard"):
        return _tool_missing("lizard", "pip install lizard")
    rc, out, err = _run([
        "lizard", "-C", "10", "-L", "150",
        "--warnings_only", "-l", "python", "src/",
    ])
    _print_output(out, err)
    return rc


def check_codespell(files: list[Path] | None = None) -> int:
    """Run codespell. Optionally limit to specific files."""
    if not _has_tool("codespell"):
        return _tool_missing("codespell", "pip install codespell")
    cmd = ["codespell", "--skip=*.pyc,./.git,./.env"]
    if files:
        cmd.extend(str(f.relative_to(PROJECT_ROOT)) for f in files)
    rc, out, err = _run(cmd)
    _print_output(out, err)
    return rc


def check_whitespace(files: list[Path]) -> int:
    """Check for trailing whitespace, CRLF, and missing EOF newline."""
    errors = 0
    for f in files:
        if not f.is_file():
            continue
        try:
            with open(f, "rb") as fh:
                content = fh.read()
        except OSError:
            continue
        if not content:
            continue
        if b"\r\n" in content:
            print(f"  ERROR: {f.relative_to(PROJECT_ROOT)}: contains CRLF line endings")
            errors += 1
        if not content.endswith(b"\n"):
            print(f"  ERROR: {f.relative_to(PROJECT_ROOT)}: missing newline at end of file")
            errors += 1
        for i, line in enumerate(content.split(b"\n"), start=1):
            stripped = line.rstrip(b" \t")
            if stripped != line:
                print(
                    f"  ERROR: {f.relative_to(PROJECT_ROOT)}:{i}: trailing whitespace"
                )
                errors += 1
    return 1 if errors else 0


def check_yaml(files: list[Path]) -> int:
    """Validate YAML syntax for .yaml / .yml files."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        print("  WARNING: PyYAML not installed, skipping YAML check")
        return 0

    errors = 0
    for f in files:
        if f.suffix not in (".yaml", ".yml"):
            continue
        try:
            with open(f) as fh:
                yaml.safe_load(fh)
        except Exception as exc:
            print(f"  ERROR: {f.relative_to(PROJECT_ROOT)}: invalid YAML - {exc}")
            errors += 1
    return 1 if errors else 0


def check_toml(files: list[Path]) -> int:
    """Validate TOML syntax for .toml files."""
    try:
        import tomllib  # type: ignore[import-untyped]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[import-untyped]
        except ImportError:
            print("  WARNING: tomli not installed, skipping TOML check")
            return 0

    errors = 0
    for f in files:
        if f.suffix != ".toml":
            continue
        try:
            with open(f, "rb") as fh:
                tomllib.load(fh)
        except Exception as exc:
            print(f"  ERROR: {f.relative_to(PROJECT_ROOT)}: invalid TOML - {exc}")
            errors += 1
    return 1 if errors else 0


def check_large_files(files: list[Path]) -> int:
    """Detect files larger than MAX_FILE_SIZE."""
    errors = 0
    for f in files:
        if not f.is_file():
            continue
        size = f.stat().st_size
        if size > MAX_FILE_SIZE:
            rel = f.relative_to(PROJECT_ROOT)
            print(f"  ERROR: {rel}: file too large ({size} > {MAX_FILE_SIZE})")
            errors += 1
    return 1 if errors else 0


def check_merge_conflict(files: list[Path]) -> int:
    """Detect merge conflict markers in files."""
    markers = (b"<<<<<<< ", b"=======\n", b">>>>>>> ")
    errors = 0
    for f in files:
        if not f.is_file():
            continue
        try:
            with open(f, "rb") as fh:
                for i, line in enumerate(fh, start=1):
                    if any(line.startswith(m) for m in markers):
                        rel = f.relative_to(PROJECT_ROOT)
                        print(f"  ERROR: {rel}:{i}: merge conflict marker")
                        errors += 1
                        break
        except OSError:
            continue
    return 1 if errors else 0


def _run_check(name: str, check_fn: callable) -> bool:  # type: ignore[arg-type]
    """Run a single check and print status. Returns True on pass."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    rc = check_fn()
    if rc != 0:
        print(f"  -> FAILED")
        return False
    print(f"  -> PASSED")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run consolidated pre-commit checks"
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Auto-fix issues where possible (ruff only)"
    )
    parser.add_argument(
        "--staged", action="store_true",
        help="Only check staged files for file-level checks"
    )
    args = parser.parse_args()

    files = _get_files(args.staged)

    checks: list[tuple[str, callable]] = [  # type: ignore[type-arg]
        ("ruff lint", lambda: check_ruff_lint(fix=args.fix)),
        ("ruff format", check_ruff_format),
        ("lizard complexity", check_lizard),
        ("codespell", lambda: check_codespell(files if args.staged else None)),
    ]

    if files:
        checks.extend([
            ("trailing whitespace / EOF", lambda: check_whitespace(files)),
            ("YAML syntax", lambda: check_yaml(files)),
            ("TOML syntax", lambda: check_toml(files)),
            ("large files", lambda: check_large_files(files)),
            ("merge conflict markers", lambda: check_merge_conflict(files)),
        ])

    failed: list[str] = []
    for name, check_fn in checks:
        if not _run_check(name, check_fn):
            failed.append(name)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        return 1
    print("All checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
