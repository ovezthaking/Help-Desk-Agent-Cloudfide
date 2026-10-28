"""Fake LLM client for deterministic offline testing of the agent loop."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from help_desk_agent.models import LLMResponse

ResponseHandler = Callable[[list[dict[str, Any]]], LLMResponse]


class FakeLLMClient:
    """Deterministic fake model that needs no network and no API key.

    Can be initialized with either a fixed sequence of LLMResponse objects
    or a custom callable handler that inspects the conversation history.
    """

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        handler: ResponseHandler | None = None,
    ) -> None:
        self._responses: deque[LLMResponse] = deque(responses or ())
        self._handler = handler
        self.call_history: list[list[dict[str, Any]]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Return the next scripted LLMResponse or invoke the handler."""
        # Record a shallow snapshot of messages for test assertions
        self.call_history.append([dict(m) for m in messages])

        if self._handler is not None:
            return self._handler(messages)

        if not self._responses:
            return LLMResponse(text="No more scripted responses available.")

        return self._responses.popleft()
