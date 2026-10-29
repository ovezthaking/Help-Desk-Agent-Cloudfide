"""Offline tests for the HelpDeskAgent loop using FakeLLMClient."""

from __future__ import annotations

from typing import Any

from help_desk_agent.agent import HelpDeskAgent
from help_desk_agent.llm.fake import FakeLLMClient
from help_desk_agent.models import LLMResponse, ToolCallRequest
from help_desk_agent.tools.registry import ToolRegistry


def test_agent_loop_single_tool_call_refund() -> None:
    """Test that agent executes calculate_refund and passes result to LLM."""

    def handler(messages: list[dict[str, Any]]) -> LLMResponse:
        last_msg = messages[-1]
        if last_msg["role"] == "user":
            return LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        name="calculate_refund",
                        args={"booking_id": "CW-1002"},
                    )
                ]
            )
        # Verify the tool result was actually passed back to the model
        assert last_msg["role"] == "tool"
        tool_res = last_msg["results"][0]["result"]
        assert tool_res["booking_id"] == "CW-1002"
        assert tool_res["rental_refund_eur"] == 40.0
        assert tool_res["deposit_refund_eur"] == 300.0
        assert tool_res["total_refund_eur"] == 340.0
        return LLMResponse(
            text=(
                "If you cancel booking CW-1002 now, you will get back "
                "40.00 EUR for the rental (50%) plus your 300.00 EUR "
                "deposit, for a total of 340.00 EUR. The 2.00 EUR booking "
                "fee is non-refundable."
            )
        )

    fake_llm = FakeLLMClient(handler=handler)
    agent = HelpDeskAgent(
        llm=fake_llm,
        tool_registry=ToolRegistry(search_mode="bm25"),
    )

    response = agent.ask("I want to cancel CW-1002. How much do I get back?")
    assert response.steps == 2
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "calculate_refund"
    assert response.tool_calls[0].result["total_refund_eur"] == 340.0
    assert "340.00 EUR" in response.answer


def test_agent_loop_multi_step_tool_calls() -> None:
    """Test agent calling search_policies then get_booking across steps."""
    scripted = [
        LLMResponse(
            tool_calls=[
                ToolCallRequest(
                    name="search_policies",
                    args={"query": "late return fee"},
                )
            ]
        ),
        LLMResponse(
            tool_calls=[
                ToolCallRequest(
                    name="get_booking",
                    args={"booking_id": "CW-1008"},
                )
            ]
        ),
        LLMResponse(
            text=(
                "Booking CW-1008 is a city bike (4 EUR/hour). A late return "
                "is charged at 1.5x the hourly rate (6 EUR per started hour "
                "of delay), taken from the 50 EUR deposit."
            )
        ),
    ]
    fake_llm = FakeLLMClient(responses=scripted)
    agent = HelpDeskAgent(
        llm=fake_llm,
        tool_registry=ToolRegistry(search_mode="bm25"),
    )

    res = agent.ask("What if I return booking CW-1008 late?")
    assert res.steps == 3
    assert [tc.name for tc in res.tool_calls] == [
        "search_policies",
        "get_booking",
    ]
    assert "6 EUR" in res.answer


def test_agent_loop_unknown_booking_does_not_crash() -> None:
    """Test agent handling a non-existent booking gracefully."""
    scripted = [
        LLMResponse(
            tool_calls=[
                ToolCallRequest(
                    name="get_booking",
                    args={"booking_id": "CW-9999"},
                )
            ]
        ),
        LLMResponse(text="I could not find booking CW-9999 in our system."),
    ]
    fake_llm = FakeLLMClient(responses=scripted)
    agent = HelpDeskAgent(
        llm=fake_llm,
        tool_registry=ToolRegistry(search_mode="bm25"),
    )

    res = agent.ask("Can you check booking CW-9999?")
    assert res.steps == 2
    assert res.tool_calls[0].result["found"] is False
    assert "CW-9999" in res.answer


def test_agent_loop_unknown_tool_graceful_error() -> None:
    """Test that an unknown tool requested by the LLM does not crash."""
    scripted = [
        LLMResponse(
            tool_calls=[
                ToolCallRequest(
                    name="non_existent_tool",
                    args={"foo": "bar"},
                )
            ]
        ),
        LLMResponse(text="Sorry, I cannot perform that action."),
    ]
    fake_llm = FakeLLMClient(responses=scripted)
    agent = HelpDeskAgent(
        llm=fake_llm,
        tool_registry=ToolRegistry(search_mode="bm25"),
    )

    res = agent.ask("Do something weird")
    assert res.steps == 2
    assert "error" in res.tool_calls[0].result


def test_agent_loop_max_steps_guard() -> None:
    """Test that infinite tool calls stop at max_steps and return fallback."""
    infinite_calls = [
        LLMResponse(
            tool_calls=[
                ToolCallRequest(
                    name="search_policies",
                    args={"query": "opening hours"},
                )
            ]
        )
        for _ in range(5)
    ]
    fake_llm = FakeLLMClient(responses=infinite_calls)
    agent = HelpDeskAgent(
        llm=fake_llm,
        tool_registry=ToolRegistry(search_mode="bm25"),
        max_steps=3,
    )

    res = agent.ask("When do you open?")
    assert res.steps == 3
    assert len(res.tool_calls) == 3
    assert "help@citywheels.example" in res.answer


def test_agent_conversation_memory_follow_up() -> None:
    """Test multi-turn memory so customer can ask follow-up questions."""
    scripted = [
        LLMResponse(
            tool_calls=[
                ToolCallRequest(
                    name="calculate_refund",
                    args={"booking_id": "CW-1002"},
                )
            ]
        ),
        LLMResponse(
            text="Cancelling CW-1002 now gives 340 EUR back (40 + 300)."
        ),
        LLMResponse(
            tool_calls=[
                ToolCallRequest(
                    name="calculate_refund",
                    args={
                        "booking_id": "CW-1002",
                        "cancel_at": "2026-06-11T10:00:00+02:00",
                    },
                )
            ]
        ),
        LLMResponse(
            text=(
                "If you cancel CW-1002 tomorrow at 10:00 instead, it is less "
                "than 24h before start, so you only get the 300 EUR deposit."
            )
        ),
    ]
    fake_llm = FakeLLMClient(responses=scripted)
    agent = HelpDeskAgent(
        llm=fake_llm,
        tool_registry=ToolRegistry(search_mode="bm25"),
    )

    first = agent.ask("How much if I cancel CW-1002 now?")
    assert "340 EUR" in first.answer

    second = agent.ask("And what if I cancel tomorrow at 10:00 instead?")
    assert "300 EUR" in second.answer
    assert second.tool_calls[0].result["rental_refund_eur"] == 0.0
    assert second.tool_calls[0].result["deposit_refund_eur"] == 300.0
    # History contains both turns
    assert len(fake_llm.call_history[-1]) >= 5

    agent.reset_memory()
    assert not agent.messages
