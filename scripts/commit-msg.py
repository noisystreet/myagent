#!/usr/bin/env python3
"""commit-msg hook: validate Conventional Commits format.

Called by pre-commit with the commit message file path as argument.
Exits with non-zero if the message doesn't match the required format.

Format:
    <type>(<scope>): <English summary>

    <Chinese detail description (optional)>

Types: feat / fix / refactor / test / docs / chore / style
Scopes: core / llm / nodes / tools / cli / ci / docs / chore
"""

import re
import sys

COMMIT_MSG_FILE = sys.argv[1]

VALID_TYPES = {"feat", "fix", "refactor", "test", "docs", "chore", "style"}
VALID_SCOPES = {"core", "llm", "nodes", "tools", "cli", "ci", "docs", "chore"}

# Pattern: <type>(<scope>): <subject>
PATTERN = re.compile(r"^(?P<type>\w+)(?:\((?P<scope>\w+)\))?:\s+(?P<subject>.+)$")


def main():
    with open(COMMIT_MSG_FILE) as f:
        lines = f.readlines()

    if not lines:
        print("ERROR: Commit message is empty.")
        sys.exit(1)

    first_line = lines[0].strip()

    # Skip merge commits and auto-generated messages
    if first_line.startswith("Merge ") or first_line.startswith("Release "):
        sys.exit(0)

    match = PATTERN.match(first_line)
    if not match:
        print(
            "ERROR: Commit message does not follow Conventional Commits format.\n"
            f"  Got:       {first_line}\n"
            f"  Expected:  <type>(<scope>): <English summary>\n"
            f"  Types:     {', '.join(sorted(VALID_TYPES))}\n"
            f"  Scopes:    {', '.join(sorted(VALID_SCOPES))}\n"
            f"  Example:   feat(executor): add JSON fallback for structured output"
        )
        sys.exit(1)

    msg_type = match.group("type")
    msg_scope = match.group("scope")
    subject_len = len(match.group("subject"))

    if msg_type not in VALID_TYPES:
        print(
            f"ERROR: Invalid type '{msg_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_TYPES))}"
        )
        sys.exit(1)

    if msg_scope and msg_scope not in VALID_SCOPES:
        print(
            f"ERROR: Invalid scope '{msg_scope}'. "
            f"Must be one of: {', '.join(sorted(VALID_SCOPES))}"
        )
        sys.exit(1)

    if subject_len < 5:
        print(
            f"ERROR: Subject line too short ({subject_len} chars). "
            f"Minimum 5 characters required."
        )
        sys.exit(1)

    if subject_len > 72:
        print(
            f"WARNING: Subject line too long ({subject_len} chars). "
            f"Consider keeping it under 72 characters."
        )

    # Validate body format if present (Chinese chars should follow after a blank line)
    if len(lines) > 2:
        if lines[1].strip() != "":
            print(
                "WARNING: Body should be separated from the subject "
                "by a blank line."
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
