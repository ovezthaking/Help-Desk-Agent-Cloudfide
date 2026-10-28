"""Core tool-calling agent loop and conversation memory management."""

from __future__ import annotations

from typing import Any

from help_desk_agent.config import STAFF_EMAIL, STAFF_PHONE
from help_desk_agent.llm.base import LLMClient
from help_desk_agent.models import AgentResponse, ToolCallRecord
from help_desk_agent.prompts import SYSTEM_PROMPT
from help_desk_agent.tools.registry import TOOL_SCHEMAS, ToolRegistry


class HelpDeskAgent:
    """CityWheels Help Desk AI Agent with tool-calling loop and memory."""

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = 6,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.messages: list[dict[str, Any]] = []

    def reset_memory(self) -> None:
        """Clear the multi-turn conversation history."""
        self.messages.clear()

    def ask(self, question: str) -> AgentResponse:
        """Process a customer question through the tool-calling loop."""
        if not question or not question.strip():
            return AgentResponse(
                answer="Please provide a question so I can assist you.",
                tool_calls=[],
                steps=0,
            )

        self.messages.append({"role": "user", "content": question.strip()})
        turn_tool_calls: list[ToolCallRecord] = []

        for step in range(1, self.max_steps + 1):
            llm_response = self.llm.generate(
                messages=self.messages,
                system_prompt=self.system_prompt,
                tools=TOOL_SCHEMAS,
            )

            if not llm_response.tool_calls:
                final_text = (
                    llm_response.text
                    or "I could not generate an answer. Please contact "
                    f"CityWheels staff at {STAFF_EMAIL} or {STAFF_PHONE}."
                )
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": final_text,
                        "raw_model_parts": llm_response.raw_model_Parts,
                    }
                )
                return AgentResponse(
                    answer=final_text,
                    tool_calls=turn_tool_calls,
                    steps=step,
                )

            # Record assistant tool-call turn in history
            self.messages.append(
                {
                    "role": "assistant",
                    "content": llm_response.text,
                    "tool_calls": [
                        {"name": tc.name, "args": tc.args}
                        for tc in llm_response.tool_calls
                    ],
                    "raw_model_parts": llm_response.raw_model_Parts,
                }
            )

            # Execute each requested tool and collect results
            step_results: list[dict[str, Any]] = []
            for tc in llm_response.tool_calls:
                tool_output = self.tool_registry.execute(
                    name=tc.name,
                    args=tc.args,
                )
                record = ToolCallRecord(
                    name=tc.name,
                    args=tc.args,
                    result=tool_output,
                )
                turn_tool_calls.append(record)
                step_results.append(
                    {
                        "name": tc.name,
                        "args": tc.args,
                        "result": tool_output,
                    }
                )

            self.messages.append(
                {
                    "role": "tool",
                    "results": step_results,
                }
            )

        fallback_msg = (
            "I reached the maximum number of reasoning steps while "
            "processing your request. Please contact the CityWheels staff at "
            f"{STAFF_EMAIL} or {STAFF_PHONE} for assistance."
        )
        self.messages.append({"role": "assistant", "content": fallback_msg})
        return AgentResponse(
            answer=fallback_msg,
            tool_calls=turn_tool_calls,
            steps=self.max_steps,
        )
