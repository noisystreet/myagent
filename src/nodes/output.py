"""Output node: summarizes execution results."""

from langchain_core.messages import AIMessage

from ..core.state import CodingAgentState
from ..llm.client import LLMClient

OUTPUT_SYSTEM_PROMPT = """\
Summarize what the coding agent did in response to the user's request.
Be concise but include:
- What steps were executed
- What files were modified
- Any errors encountered
- The final result
"""


def output_node(state: CodingAgentState, llm: LLMClient) -> dict:
    """Generate a summary of what was accomplished."""
    plan = state.get("plan", [])
    results = state.get("tool_results", [])
    errors = state.get("errors", [])

    result_summary = "\n".join(
        f"  [{'✓' if r.success else '✗'}] {r.tool_name}: {r.data or r.error or ''}" for r in results
    )

    summary = llm.invoke(
        prompt=f"""
Plan steps:
{chr(10).join(f"  {i}. {s}" for i, s in enumerate(plan))}

Execution results:
{result_summary}

Errors: {errors if errors else "None"}

Summarize what was accomplished.
""",
        system=OUTPUT_SYSTEM_PROMPT,
    )

    summary_text = str(summary)

    return {
        "messages": [AIMessage(content=summary_text)],
        "final_output": summary_text,
        "next_action": "end",
    }
