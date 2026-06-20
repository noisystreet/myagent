"""Chat node: structured conversation with code block rendering."""

from langchain_core.messages import AIMessage, SystemMessage

from ..core.state import CodingAgentState
from ..llm.client import LLMClient
from ..llm.schemas import ChatResponse

CHAT_SYSTEM_PROMPT = """\
You are a helpful programming assistant. Answer the user's questions clearly and concisely.
You can explain code concepts, answer questions, and provide guidance.
You do NOT have access to the user's files or the ability to execute commands in this mode.
You CAN see the full conversation history — use it to maintain context!

When providing code examples, always specify the programming language.
Keep explanations and code separate for best rendering.
"""


def _format_chat_response(response: ChatResponse) -> str:
    """Format a structured ChatResponse as markdown text."""
    parts: list[str] = []

    if response.explanation:
        parts.append(response.explanation.strip())

    for block in response.code_blocks:
        if block.caption:
            parts.append(f"\n**{block.caption}**\n")
        lang = block.language or ""
        parts.append(f"\n```{lang}\n{block.code.strip()}\n```\n")

    if response.references:
        parts.append("\n**References:**")
        for ref in response.references:
            parts.append(f"- {ref}")

    return "\n".join(parts).strip()


def chat_node(state: CodingAgentState, llm: LLMClient) -> dict:
    """Respond to the user conversationally with structured output."""
    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)]
    messages.extend(state["messages"])

    response = llm.invoke(messages=messages, schema=ChatResponse)

    if isinstance(response, ChatResponse):
        response_text = _format_chat_response(response)
    else:
        response_text = str(response)

    return {
        "messages": [AIMessage(content=response_text)],
        "final_output": response_text,
        "next_action": "end",
    }
