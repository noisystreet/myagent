"""Planner node: analyzes user request and produces a plan."""

from ..core.state import CodingAgentState
from ..llm.client import LLMClient
from ..llm.schemas import PlanSchema

PLANNER_SYSTEM_PROMPT = """\
You are a programming agent planner. Your job is to break down a user's request into concrete, ordered steps.

Rules:
1. Each step must be a SINGLE concrete action: read a file, edit a file, run a command, etc.
2. Keep steps focused — one operation per step.
3. Identify all files that will need to be read or modified.
4. Be specific with file paths.

Example:
User: "Add error handling to worker.py"
Output:
  analysis: "The file worker.py needs try/except blocks around the main processing logic."
  steps: ["read worker.py", "add try/except block around process_task()", "run python worker.py to verify"]
  files: ["worker.py"]
"""


def planner_node(state: CodingAgentState, llm: LLMClient) -> dict:
    """Analyze user messages and produce an execution plan."""
    user_message = state["messages"][-1].content if state["messages"] else ""

    response = llm.invoke(
        prompt=user_message,
        system=PLANNER_SYSTEM_PROMPT,
        schema=PlanSchema,
    )

    if not isinstance(response, PlanSchema):
        # Fallback: parse as string
        return {
            "plan": [str(response)],
            "current_step": 0,
            "next_action": "executor",
        }

    return {
        "plan": response.steps,
        "current_step": 0,
        "next_action": "executor",
    }
