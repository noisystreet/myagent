"""CLI entry point for the coding agent."""

import argparse
import logging
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from .core.config import AgentConfig
from .core.graph import build_graph
from .core.state import CodingAgentState
from .core.stream import run_streaming
from .llm.client import LLMClient

load_dotenv()

logger = logging.getLogger(__name__)


def _new_state(workspace: str, max_steps: int) -> CodingAgentState:
    """Create a fresh initial state."""
    return {
        "messages": [],
        "plan": [],
        "current_step": 0,
        "max_steps": max_steps,
        "workspace": workspace,
        "tool_results": [],
        "errors": [],
        "retry_count": 0,
        "step_attempts": 0,
        "mode": "task",
        "next_action": "intent_router",
        "final_output": None,
    }


def run_once(graph, state: CodingAgentState, task: str, config: dict) -> CodingAgentState:
    """Run a single turn with streaming: append user message, invoke graph, return final state."""
    state["messages"].append(HumanMessage(content=task))
    logger.info("Running: %s", task)
    final_state = state
    for event in run_streaming(graph, state, config):
        _display_stream_event(event)
        final_state = event.data
    return final_state


def main():
    args = _parse_args()
    config = _build_config(args)
    llm, graph, workspace = _setup(config)
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    prompt = " ".join(args.prompt) if args.prompt else None
    if prompt:
        state = _new_state(str(workspace), config.max_steps)
        state = run_once(graph, state, prompt, thread_config)
        _display_result(state)
    else:
        _interactive_loop(graph, str(workspace), config.max_steps)


def _parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="myagent — a LangGraph-based programming agent")
    parser.add_argument(
        "prompt", nargs="*", help="Task description. If omitted, enters interactive mode."
    )
    parser.add_argument(
        "--workspace", "-w", default=".", help="Working directory (default: current dir)"
    )
    parser.add_argument(
        "--model", "-m", default=None, help="LLM model (default: from LLM_MODEL env)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=20, help="Maximum execution steps (default: 20)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def _build_config(args):
    """Build AgentConfig from parsed args."""
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    return AgentConfig(
        model=args.model or AgentConfig.model,
        workspace=args.workspace,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )


def _setup(config):
    """Initialize LLM, graph, and workspace."""
    llm = LLMClient(model_name=config.model, api_key=config.api_key, base_url=config.base_url)
    graph = build_graph(llm, MemorySaver())
    workspace = Path(config.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    logger.info("Workspace: %s | Model: %s", workspace, config.model)
    return llm, graph, workspace


def _display_result(state: CodingAgentState):
    """Print execution result."""
    output = state.get("final_output") or "No output generated."
    print("\n" + "=" * 60 + f"\n{output}\n" + "=" * 60)
    mode = state.get("mode", "task")
    steps = state.get("step_attempts", 0)
    errors = len(state.get("errors", []))
    print(f"\nMode: {mode} | Steps: {steps} | Errors: {errors}")


def _display_stream_event(event):
    """Render a single streaming event to the terminal."""
    node = event.node
    kind = event.kind
    content = event.content

    if kind == "start":
        print(f"\n── [{node}] {content} ──")
    elif kind == "interim":
        if content:
            print(f"\n[{node}]\n{content}")
    elif kind == "tool_result":
        if content:
            print(f"\n{content}")
    elif kind == "error":
        print(f"\n⚠ [{node}] Error:\n{content}")
    elif kind == "end":
        print(f"\n{'=' * 60}\n{content}\n{'=' * 60}")


def _interactive_loop(graph, workspace: str, max_steps: int):
    """Run interactive REPL with cross-turn memory."""
    state = _new_state(workspace, max_steps)
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    msg_count = 0

    print("myagent interactive mode. Type 'exit' to quit. Type 'new' for a fresh session.")
    print("=" * 60)

    while True:
        try:
            task = input("\n❯ ").strip()
            if not task:
                continue
            if task.lower() in {"exit", "quit", "q"}:
                break
            if task.lower() == "new":
                state = _new_state(workspace, max_steps)
                thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                msg_count = 0
                print("--- New session started ---")
                continue

            state = run_once(graph, state, task, thread_config)
            msg_count += 1
            mode = state.get("mode", "?")
            errors = len(state.get("errors", []))
            msg_pairs = len(state.get("messages", [])) // 2
            summary = f"─── Turn {msg_count} | Msgs: {msg_pairs} pairs | Mode: {mode}"
            if errors:
                summary += f" | ⚠ {errors} error(s)"
            print(summary + " ───")

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            logger.exception("Error: %s", e)
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
