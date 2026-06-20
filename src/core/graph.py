"""Graph assembly: build and compile the LangGraph."""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..core.registry import ToolRegistry
from ..core.state import CodingAgentState
from ..llm.client import LLMClient
from ..nodes.chat import chat_node
from ..nodes.executor import executor_node
from ..nodes.output import output_node
from ..nodes.planner import planner_node
from ..nodes.reflector import reflector_node
from ..nodes.router import intent_router_node

logger = logging.getLogger(__name__)


def _reflect_route(state: CodingAgentState) -> str:
    """Route after reflector. Enforces safety caps before LLM verdict.

    C2 Tier 1: Per-step retry cap (cheapest check).
    C2 Tier 2: Global loop cap.
    Default: delegate to reflector's next_action.
    """
    # C2 Tier 1: per-step retry cap
    reflections = state.get("reflections", [])
    if reflections:
        current_idx = state.get("current_step", 0)
        retries_on_current = sum(
            1
            for r in reflections[-5:]
            if r.get("step_index") == current_idx and r["verdict"] == "retry"
        )
        max_retries = state.get("max_retries", 2)
        if retries_on_current >= max_retries:
            logger.info(
                "Per-step retry cap reached (%d/%d) for step %d, forcing replan",
                retries_on_current,
                max_retries,
                current_idx,
            )
            return "replan"

    # C2 Tier 2: global loop cap
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 30)
    if loop_count >= max_loops:
        logger.info(
            "Global loop cap reached (%d/%d), forcing output",
            loop_count,
            max_loops,
        )
        return "done"

    # Default: use reflector's verdict
    return state.get("next_action", "done")


def should_continue(state: CodingAgentState) -> str:
    """Route to the next node based on next_action."""
    # Circuit breaker for task mode
    if state.get("mode") == "task":
        if state.get("step_attempts", 0) > state.get("max_steps", 20):
            return "output"
    return state.get("next_action", "end")


def build_graph(
    llm: LLMClient,
    registry: ToolRegistry,
    checkpointer: MemorySaver | None = None,
) -> StateGraph:
    """Build the coding agent state graph.

    Topology:
        START → intent_router
                    │
                    ├── (chat) → chat_node → END
                    │
                    └── (task) → planner → executor → reflector ─┐
                                    ↑              │           │
                              continue/retry       │     done → output → END
                                    │         replan → planner
                                    └─────────────────┘
    """
    builder = StateGraph(CodingAgentState)

    # Add nodes — inject LLM and registry via closure
    builder.add_node("intent_router", lambda s: intent_router_node(s, llm))
    builder.add_node("chat", lambda s: chat_node(s, llm))
    builder.add_node("planner", lambda s: planner_node(s, llm))
    builder.add_node("executor", lambda s: executor_node(s, llm, registry))
    builder.add_node("reflector", lambda s: reflector_node(s, llm))
    builder.add_node("output", lambda s: output_node(s, llm))

    # Start → intent router
    builder.add_edge(START, "intent_router")

    # Intent router → chat or task mode
    builder.add_conditional_edges(
        "intent_router",
        should_continue,
        {
            "chat": "chat",
            "planner": "planner",
        },
    )

    # Chat → end
    builder.add_edge("chat", END)

    # Task pipeline: planner → executor → reflector → (executor|planner|output)
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "reflector")

    builder.add_conditional_edges(
        "reflector",
        _reflect_route,
        {
            "continue": "executor",
            "retry": "executor",
            "replan": "planner",
            "done": "output",
        },
    )

    builder.add_edge("output", END)

    return builder.compile(checkpointer=checkpointer)
