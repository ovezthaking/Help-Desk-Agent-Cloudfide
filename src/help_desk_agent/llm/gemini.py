"""Google Gemini LLM client using the official google-genai SDK."""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from help_desk_agent.config import get_gemini_api_key, get_gemini_model
from help_desk_agent.models import LLMResponse, ToolCallRequest

_FALLBACK_MODELS = (
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
)
_RETRYABLE_ERROR_CODES = ("404", "429", "503", "504", "UNAVAILABLE")


class GeminiLLMClient:
    """LLM client backed by Google Gemini API with native tool calling."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        resolved_key = api_key or get_gemini_api_key()
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in your environment "
                "or in a .env file (see .env.example)."
            )
        self.model = model or get_gemini_model()
        self.temperature = temperature
        self.client = genai.Client(
            api_key=resolved_key,
            http_options=types.HttpOptions(timeout=30000),
        )

    def _build_tool_config(
        self,
        tools: list[dict[str, Any]],
    ) -> list[Any]:
        declarations = [
            types.FunctionDeclaration(
                name=schema["name"],
                description=schema["description"],
                parameters=schema["parameters"],
            )
            for schema in tools
        ]
        return [types.Tool(function_declarations=declarations)]

    def _convert_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[types.Content]:
        contents: list[types.Content] = []
        for msg in messages:
            role = msg["role"]
            if role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=str(msg["content"]))],
                    )
                )
            elif role == "assistant":
                raw_parts = msg.get("raw_model_parts")
                if raw_parts:
                    contents.append(
                        types.Content(role="model", parts=raw_parts)
                    )
                else:
                    parts: list[types.Part] = []
                    if msg.get("content"):
                        parts.append(
                            types.Part.from_text(text=str(msg["content"]))
                        )
                    parts.extend(
                        types.Part.from_function_call(
                            name=tc["name"],
                            args=tc.get("args", {}),
                        )
                        for tc in msg.get("tool_calls", [])
                    )
                    contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                tool_parts = [
                    types.Part.from_function_response(
                        name=item["name"],
                        response=item["result"],
                    )
                    for item in msg.get("results", [])
                ]
                contents.append(types.Content(role="user", parts=tool_parts))
        return contents

    def generate(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Call Gemini generate_content with fallback on transient 503/504."""
        contents = self._convert_messages(messages)
        gemini_tools = self._build_tool_config(tools) if tools else None

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            tools=gemini_tools,
        )

        candidates = list(dict.fromkeys((self.model, *_FALLBACK_MODELS)))

        response = None
        last_exc: Exception | None = None
        for candidate_model in candidates:
            try:
                response = self.client.models.generate_content(
                    model=candidate_model,
                    contents=contents,
                    config=config,
                )
                self.model = candidate_model
                break
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if any(code in err_str for code in _RETRYABLE_ERROR_CODES):
                    continue
                raise

        if response is None:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(
                "Gemini API error: failed to generate "
                + "content with all candidate models."
            )

        return self._parse_response(response)

    def _parse_response(self, response: Any) -> LLMResponse:
        tool_calls: list[ToolCallRequest] = []
        text_parts: list[str] = []
        raw_parts: list[types.Part] = []

        if response.candidates and response.candidates[0].content:
            candidate_parts = response.candidates[0].content.parts or []
            raw_parts = list(candidate_parts)
            for part in candidate_parts:
                if part.function_call:
                    fc = part.function_call
                    args_dict = dict(fc.args) if fc.args else {}
                    tool_calls.append(
                        ToolCallRequest(
                            name=fc.name or "",
                            args=args_dict,
                            call_id=getattr(fc, "id", None),
                        )
                    )
                elif part.text and not getattr(part, "thought", False):
                    text_parts.append(part.text)

        combined_text = "\n".join(text_parts).strip() if text_parts else None
        if not combined_text and not tool_calls and response.text:
            combined_text = response.text.strip()

        return LLMResponse(
            text=combined_text,
            tool_calls=tool_calls,
            raw_model_parts=raw_parts if raw_parts else None,
        )
