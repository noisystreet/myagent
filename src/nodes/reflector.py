"""Reflector node: observes step results and decides next action."""

import logging

from ..llm.client import LLMClient
from ..llm.schemas import ReflectDecision

logger = logging.getLogger(__name__)

# AI-generated

_MAX_REASON_LENGTH = 500
_MAX_SUGGESTION_LENGTH = 200
_MAX_DISPLAY_STEPS = 5
_RECENT_HISTORY_SIZE = 3
_RESULT_TEXT_TRUNCATE = 200

REFLECTOR_SYSTEM_PROMPT = """\
You are an execution observer for a coding agent. After each step, decide what to do next.

Inputs provided:
- current_step: The plan step that was just executed
- tool_result: Success/failure status, output data, or error details
- recent_history: Last 3 steps and their results (for context)
- remaining_plan: Steps not yet executed

Decide ONE action:
- **continue**: Step succeeded, advance to next step.
- **retry**: Step failed but fixable. Provide a specific suggestion.
- **replan**: Remaining plan is invalid. Needs full re-planning.
- **done**: Task complete (even if plan has remaining steps).
- **interrupt**: Requires human confirmation before proceeding.

Rules:
- Do not retry the same error more than twice total.
- If a file read fails, do not suggest re-reading it.
- If tests fail identically 3 times, choose done with explanation.
- Keep reason under 500 characters.
- Keep suggestion under 200 characters if provided.
"""


def _format_step(step: str | dict) -> str:
    """Format a single plan step for the prompt."""
    if isinstance(step, dict):
        return (
            f"  [{step.get('id', '?')}] {step.get('action', '?')}"
            f"({step.get('args', {})}) — {step.get('description', '')}"
        )
    return f"  - {step}"


def _format_result(result: dict) -> str:
    """Format a ToolResult-like dict for the prompt."""
    if result.get("success"):
        data_text = str(result.get("data", ""))[:_RESULT_TEXT_TRUNCATE]
        return f"  OK {result.get('tool_name', '?')}: {data_text}"
    err = result.get("error", {})
    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
    return f"  FAIL {result.get('tool_name', '?')}: {msg[:_RESULT_TEXT_TRUNCATE]}"


def _build_reflect_prompt(
    current_step: str | dict,
    tool_result: dict,
    recent_history: list[dict],
    remaining_plan: list[str] | list[dict],
) -> str:
    """Build the user prompt for the reflector LLM call."""
    parts = [
        f"Current step:\n{_format_step(current_step)}",
        f"\nResult:\n{_format_result(tool_result)}",
    ]
    if recent_history:
        parts.append("\nRecent history:")
        for h in recent_history[-_RECENT_HISTORY_SIZE:]:
            parts.append(_format_result(h))
    if remaining_plan:
        parts.append("\nRemaining plan:")
        for s in remaining_plan[:_MAX_DISPLAY_STEPS]:
            parts.append(_format_step(s))
        if len(remaining_plan) > _MAX_DISPLAY_STEPS:
            parts.append(f"  ... and {len(remaining_plan) - _MAX_DISPLAY_STEPS} more steps")
    else:
        parts.append("\nNo remaining steps.")
    return "\n".join(parts)


def _advance_step(current_idx: int, verdict: str) -> int:
    """Return the next step index based on reflector verdict."""
    if verdict in {"continue", "replan"}:
        return current_idx + 1
    return current_idx


def reflector_node(state: dict, llm: LLMClient) -> dict:
    """Observe the latest step result and decide the next action.

    Returns state updates including:
    - next_action: where to route next (executor/planner/output)
    - completed: whether the task is done
    - reflections: new reflection entry appended via Annotated reducer
    - loop_count: incremented iteration counter
    """
    current_idx = state.get("current_step", 0)
    plan = state.get("plan", [])
    tool_results = state.get("tool_results", [])

    # Guard: no plan or out of bounds → done
    if not plan or current_idx >= len(plan):
        return {
            "completed": True,
            "next_action": "done",
            "reflections": [{"verdict": "done", "reason": "Plan exhausted"}],
        }

    current_step = plan[current_idx]
    last_result = (
        tool_results[-1]
        if tool_results
        else {
            "tool_name": "unknown",
            "success": False,
            "error": "No results",
        }
    )
    recent_history = tool_results[:-1]
    remaining = plan[current_idx + 1 :]

    prompt = _build_reflect_prompt(current_step, last_result, recent_history, remaining)

    try:
        decision: ReflectDecision = llm.invoke(
            prompt=prompt,
            system=REFLECTOR_SYSTEM_PROMPT,
            schema=ReflectDecision,
        )
    except Exception as e:
        logger.warning("Reflector LLM call failed: %s", e)
        fallback_verdict = "continue" if last_result.get("success") else "done"
        decision = ReflectDecision(
            verdict=fallback_verdict,
            reason=f"LLM fallback: {e}",
        )

    # Enforce length limits (C7 L2)
    reason = (
        decision.reason[:_MAX_REASON_LENGTH]
        if len(decision.reason) > _MAX_REASON_LENGTH
        else decision.reason
    )
    suggestion = None
    if decision.suggestion:
        suggestion = (
            decision.suggestion[:_MAX_SUGGESTION_LENGTH]
            if len(decision.suggestion) > _MAX_SUGGESTION_LENGTH
            else decision.suggestion
        )

    reflection_entry: dict = {
        "step_index": current_idx,
        "verdict": decision.verdict,
        "reason": reason,
        "suggestion": suggestion,
        "severity": decision.severity,
    }

    # Phase 1: interrupt falls through to done
    next_action = decision.verdict if decision.verdict != "interrupt" else "done"

    # Advance current_step only when moving forward (not retry/done)
    new_step = _advance_step(current_idx, decision.verdict)

    return {
        "next_action": next_action,
        "completed": next_action == "done",
        "current_step": new_step,
        "plan": plan,  # pass through for stream display (next-step preview)
        "reflections": [reflection_entry],
        "loop_count": state.get("loop_count", 0) + 1,
    }
