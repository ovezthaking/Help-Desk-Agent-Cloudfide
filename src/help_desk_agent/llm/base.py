"""Base protocol for LLM clients."""

from __future__ import annotations

from typing import Any, Protocol

from help_desk_agent.models import LLMResponse


class LLMClient(Protocol):
    """Protocol implemented by real (Gemini) and fake (test) LLM clients."""

    def generate(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Generate the next assistant turn given conversation messages."""
        ...
