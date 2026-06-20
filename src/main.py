"""CLI entry point for the coding agent."""

import argparse
import logging
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from .core.config import AgentConfig
from .core.display import (
    display_banner,
    display_result,
    display_stream_event,
    display_turn_summary,
)
from .core.graph import build_graph
from .core.registry import ToolRegistry
from .core.state import CodingAgentState
from .core.stream import Spinner, run_streaming
from .llm.client import LLMClient
from .tools import ALL_TOOLS

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
    spinner = Spinner("Thinking...")
    spinner.start()
    for event in run_streaming(graph, state, config):
        spinner.stop()
        display_stream_event(event)
        final_state = event.data
        spinner = Spinner("Thinking...")
        spinner.start()
    spinner.stop()
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
        display_result(state)
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v INFO, -vv DEBUG)",
    )
    return parser.parse_args()


_LOG_LEVELS: list[int] = [logging.WARNING, logging.INFO, logging.DEBUG]


def _build_config(args):
    """Build AgentConfig from parsed args."""
    log_level = _LOG_LEVELS[min(args.verbose, len(_LOG_LEVELS) - 1)]
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
    registry = ToolRegistry(ALL_TOOLS)
    graph = build_graph(llm, registry, MemorySaver())
    workspace = Path(config.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    logger.info("Workspace: %s | Model: %s", workspace, config.model)
    return llm, graph, workspace


def _interactive_loop(graph, workspace: str, max_steps: int):
    """Run interactive REPL with cross-turn memory."""
    state = _new_state(workspace, max_steps)
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    msg_count = 0

    display_banner()

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
            display_turn_summary(msg_count, msg_pairs, mode, errors)

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            logger.exception("Error: %s", e)
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
