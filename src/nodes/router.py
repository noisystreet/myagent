"""Intent router node: decides whether the user wants a chat or a coding task."""

from ..core.state import CodingAgentState
from ..llm.client import LLMClient

ROUTER_SYSTEM_PROMPT = """\
You are a classifier. Determine if the user's message requires coding operations
(read/write/edit files, run commands, search code) or is just a conversation.

Respond with exactly one word:
- "task" if the user wants to: create/modify files, run commands, search code,
  debug, fix bugs, refactor, write code, or any operation that needs tools.
- "chat" if the user is: greeting, asking general questions, having a discussion,
  asking for explanations, or anything that doesn't need file/command operations.

Examples:
  "Write a Python script" → task
  "Fix the bug in main.py" → task
  "Hello, how are you?" → chat
  "What is Python?" → chat
  "Create a Flask API" → task
  "Can you explain closures?" → chat
"""


def intent_router_node(state: CodingAgentState, llm: LLMClient) -> dict:
    """Classify the user's intent: chat or task."""
    user_message = state["messages"][-1].content if state["messages"] else ""

    response = llm.invoke(
        prompt=user_message,
        system=ROUTER_SYSTEM_PROMPT,
    )

    mode = str(response).strip().lower()
    # Default to task if unclear
    if "task" in mode:
        next_action = "planner"
        resolved_mode = "task"
    else:
        next_action = "chat"
        resolved_mode = "chat"

    return {
        "mode": resolved_mode,
        "next_action": next_action,
    }
