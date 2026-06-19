"""Graph assembly: build and compile the LangGraph."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..core.state import CodingAgentState
from ..llm.client import LLMClient
from ..nodes.chat import chat_node
from ..nodes.executor import executor_node
from ..nodes.output import output_node
from ..nodes.planner import planner_node
from ..nodes.router import intent_router_node


def should_continue(state: CodingAgentState) -> str:
    """Route to the next node based on next_action."""
    # Circuit breaker for task mode
    if state.get("mode") == "task":
        if state.get("step_attempts", 0) > state.get("max_steps", 20):
            return "output"
    return state.get("next_action", "end")


def build_graph(llm: LLMClient, checkpointer: MemorySaver | None = None) -> StateGraph:
    """Build the coding agent state graph.

    Topology:
        START → intent_router
                    │
                    ├── (chat) → chat_node → END
                    │
                    └── (task) → planner → executor → output → END
    """
    builder = StateGraph(CodingAgentState)

    # Add nodes — inject LLM via closure
    builder.add_node("intent_router", lambda s: intent_router_node(s, llm))
    builder.add_node("chat", lambda s: chat_node(s, llm))
    builder.add_node("planner", lambda s: planner_node(s, llm))
    builder.add_node("executor", lambda s: executor_node(s, llm))
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

    # Task pipeline
    builder.add_conditional_edges(
        "planner",
        should_continue,
        {"executor": "executor", "output": "output"},
    )

    builder.add_conditional_edges(
        "executor",
        should_continue,
        {
            "executor": "executor",
            "output": "output",
        },
    )

    builder.add_edge("output", END)

    return builder.compile(checkpointer=checkpointer)
