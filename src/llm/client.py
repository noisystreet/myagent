"""LLM client with retry, structured output, and model switching."""

import json
import logging
import os
import re
import time
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# Sentinel for models that don't support structured output
_MODELS_WITHOUT_STRUCTURED_OUTPUT: set[str] = set()


class LLMClient:
    """LLM invocation wrapper with retry and structured output."""

    def __init__(
        self,
        model: BaseChatModel | None = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_retries: int = 3,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model_name = model_name
        self._model = model or self._build_model(
            model_name,
            temperature,
            api_key,
            base_url,
        )
        self.max_retries = max_retries

    @staticmethod
    def _resolve_key(model_name: str, api_key: str | None) -> str | None:
        """Resolve API key: explicit parameter > LLM_API_KEY env var."""
        if api_key:
            return api_key
        return os.getenv("LLM_API_KEY")

    @staticmethod
    def _build_model(
        model_name: str,
        temperature: float,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> BaseChatModel:
        """Build a LangChain chat model from name."""
        kwargs = {"model": model_name, "temperature": temperature}
        effective_key = LLMClient._resolve_key(model_name, api_key)
        if effective_key:
            kwargs["api_key"] = effective_key
        if base_url:
            kwargs["base_url"] = base_url

        if model_name.startswith("claude"):
            return ChatAnthropic(**kwargs)
        # Default: OpenAI or OpenAI-compatible (DeepSeek, Ollama, vLLM, etc.)
        return ChatOpenAI(**kwargs)

    def invoke(
        self,
        prompt: str | None = None,
        system: str = "",
        schema: type[BaseModel] | None = None,
        messages: list | None = None,
        tools: list[dict] | None = None,
    ) -> str | BaseModel | Any:
        """Invoke LLM with optional structured output or tool binding.

        Falls back to text + JSON parsing when the model doesn't
        support native structured output.

        Args:
            prompt: The user/human message (ignored if messages is provided).
            system: Optional system message prepended when using prompt.
            schema: Optional Pydantic model for structured output.
            messages: Full message list (overrides prompt + system).
            tools: Optional list of tool definitions for function calling
                   (OpenAI/DeepSeek compatible format).

        Returns:
            Content string, a Pydantic instance if schema provided,
            or an AIMessage if tools are used.
        """
        last_error = None
        skip_schema = self.model_name in _MODELS_WITHOUT_STRUCTURED_OUTPUT

        for attempt in range(self.max_retries):
            try:
                msgs = self._build_messages(prompt, system, messages)
                return self._model_call(msgs, schema, skip_schema, tools)

            except Exception as e:
                last_error = e

                if schema and _is_unsupported_format_error(e):
                    result = self._try_json_fallback(prompt, system, schema, messages)
                    if result is not None:
                        return result
                    continue

                logger.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)

        # Final attempt: plain text + JSON extraction
        if schema:
            result = self._try_json_fallback(prompt, system, schema, messages)
            if result is not None:
                return result

        raise last_error  # type: ignore

    @staticmethod
    def _build_messages(prompt: str | None, system: str, messages: list | None) -> list:
        """Build message list from prompt/system or use provided messages."""
        if messages is not None:
            return messages
        msgs = []
        if system:
            msgs.append(SystemMessage(system))
        msgs.append(HumanMessage(prompt or ""))
        return msgs

    def _model_call(
        self,
        msgs: list,
        schema: type[BaseModel] | None,
        skip_schema: bool,
        tools: list[dict] | None,
    ) -> str | BaseModel | Any:
        """Execute model call with optional structured output or tool binding."""
        if tools:
            return self._model.bind_tools(tools).invoke(msgs)  # type: ignore
        if schema and not skip_schema:
            # AI-generated
            # Try native structured output first (OpenAI-style)
            try:
                return self._model.with_structured_output(schema).invoke(msgs)  # type: ignore
            except Exception:
                pass
            # Fall back to response_format=json_object (DeepSeek-compatible)
            if _supports_json_mode(self.model_name):
                return self._call_with_json_mode(msgs, schema)
        return self._model.invoke(msgs).content  # type: ignore

    def _call_with_json_mode(
        self,
        msgs: list,
        schema: type[BaseModel],
    ) -> BaseModel:
        """Call model with response_format=json_object and parse into Pydantic schema.

        Used for DeepSeek and other OpenAI-compatible models that support
        JSON mode but not LangChain's with_structured_output().
        """
        schema_props = {}
        for field_name, field in schema.model_fields.items():
            schema_props[field_name] = _describe_type(field.annotation)

        json_instruction = (
            f"Respond with valid JSON only, matching this schema:\n"
            f"{json.dumps(schema_props, indent=2, ensure_ascii=False)}\n\n"
            f"Wrap the JSON in ```json ... ``` markers."
        )

        # Append JSON instruction to system message or prepend one
        enriched_msgs = list(msgs)
        for i, msg in enumerate(enriched_msgs):
            if isinstance(msg, SystemMessage):
                enriched_msgs[i] = SystemMessage(content=msg.content + "\n\n" + json_instruction)
                break
        else:
            enriched_msgs.insert(0, SystemMessage(content=json_instruction))

        response = self._model.invoke(  # type: ignore
            enriched_msgs,
            response_format={"type": "json_object"},
        ).content

        json_str = _extract_json(str(response))
        parsed = json.loads(json_str)
        return schema(**parsed)

    def _try_json_fallback(
        self,
        prompt: str | None,
        system: str,
        schema: type[BaseModel],
        messages: list | None = None,
    ) -> BaseModel | None:
        """Attempt structured output via JSON fallback. Returns None on failure."""
        logger.warning(
            "Model '%s' doesn't support structured output. Falling back to text + JSON parsing.",
            self.model_name,
        )
        _MODELS_WITHOUT_STRUCTURED_OUTPUT.add(self.model_name)
        try:
            return self._invoke_with_json_fallback(
                prompt or "",
                system,
                schema,
                messages,
            )
        except Exception:
            # Final attempt with plain text + JSON extraction
            try:
                msgs = self._build_messages(prompt, system, messages)
                text = self._model.invoke(msgs).content  # type: ignore
                json_str = _extract_json(str(text))
                parsed = json.loads(json_str)
                return schema(**parsed)
            except Exception:
                return None

    def _invoke_with_json_fallback(
        self,
        prompt: str,
        system: str,
        schema: type[BaseModel],
        messages: list | None = None,
    ) -> BaseModel:
        """Fallback: ask the LLM to output JSON in plain text, then parse."""
        # Build a description of the expected JSON schema
        schema_props = {}
        for field_name, field in schema.model_fields.items():
            schema_props[field_name] = _describe_type(field.annotation)

        json_instruction = f"""
Respond with valid JSON only, matching this schema:
{json.dumps(schema_props, indent=2, ensure_ascii=False)}

Wrap the JSON in ```json ... ``` markers.
"""

        if messages is not None:
            msgs = list(messages)
            for i, msg in enumerate(msgs):
                if isinstance(msg, SystemMessage):
                    msgs[i] = SystemMessage(
                        content=msg.content + "\n\nIMPORTANT: " + json_instruction
                    )
                    break
            else:
                msgs.insert(0, SystemMessage(content="IMPORTANT: " + json_instruction))
        else:
            msgs = []
            if system:
                msgs.append(SystemMessage(system + "\n\nIMPORTANT: " + json_instruction))
            else:
                msgs.append(SystemMessage(json_instruction))
            msgs.append(HumanMessage(prompt))

        response = self._model.invoke(msgs).content  # type: ignore

        # Parse JSON from the response
        json_str = _extract_json(str(response))
        parsed = json.loads(json_str)
        return schema(**parsed)

    @property
    def model(self) -> BaseChatModel:
        return self._model


def _supports_json_mode(model_name: str) -> bool:
    """Check if model supports response_format=json_object (DeepSeek-style).

    DeepSeek and most OpenAI-compatible APIs support this parameter.
    """
    return any(
        model_name.startswith(prefix) for prefix in ("deepseek", "qwen", "glm", "yi-", "moonshot")
    )


def _is_unsupported_format_error(e: Exception) -> bool:
    """Check if the error is about unsupported response_format."""
    msg = str(e).lower()
    return any(
        kw in msg
        for kw in (
            "response_format",
            "bad request",
            "400",
            "this response_format type is unavailable",
            "not supported",
            "not implement",
        )
    )


def _describe_type(annotation: Any) -> str:
    """Convert a type annotation to a readable description."""
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        args = getattr(annotation, "__args__", [str])
        return f"list of {_describe_type(args[0])}"
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    return str(annotation)


def _extract_json(text: str) -> str:
    """Extract JSON from text, handling ```json ... ``` markers."""
    # Try to find JSON code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to find {...} or [...] directly
    for delim in ("{", "["):
        start = text.find(delim)
        if start >= 0:
            end = text.rfind("}" if delim == "{" else "]")
            if end > start:
                return text[start : end + 1]
    return text.strip()
