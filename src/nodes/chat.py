"""Chat node: pure conversation without tool calling."""

from langchain_core.messages import AIMessage, SystemMessage

from ..core.state import CodingAgentState
from ..llm.client import LLMClient

CHAT_SYSTEM_PROMPT = """\
You are a helpful programming assistant. Answer the user's questions clearly and concisely.
You can explain code concepts, answer questions, and provide guidance.
You do NOT have access to the user's files or the ability to execute commands in this mode.
You CAN see the full conversation history — use it to maintain context!
"""


def chat_node(state: CodingAgentState, llm: LLMClient) -> dict:
    """Respond to the user conversationally with full history."""
    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)]
    messages.extend(state["messages"])

    response = llm.invoke(messages=messages)

    response_text = str(response)

    return {
        "messages": [AIMessage(content=response_text)],
        "final_output": response_text,
        "next_action": "end",
    }
