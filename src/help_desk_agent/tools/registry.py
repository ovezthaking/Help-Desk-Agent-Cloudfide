"""Tool registry and schema definitions for the CityWheels Help Desk Agent."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from help_desk_agent.tools.bookings import get_booking
from help_desk_agent.tools.policies import search_policies
from help_desk_agent.tools.refunds import calculate_refund

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_policies",
        "description": (
            "Search the CityWheels shop policies, prices, cancellation "
            "rules, and FAQ documents in data/policies/ to find sections "
            "relevant to a customer's question."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": (
                        "Search query or keywords describing the policy, "
                        "price, rule, or FAQ topic to look up."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_booking",
        "description": (
            "Get the details of a single booking from data/bookings.json "
            "by its booking ID (for example 'CW-1001'). Returns a clear "
            "'not found' result when the booking does not exist."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "booking_id": {
                    "type": "STRING",
                    "description": (
                        "The booking identifier, e.g. 'CW-1001' or 'CW-1002'."
                    ),
                },
            },
            "required": ["booking_id"],
        },
    },
    {
        "name": "calculate_refund",
        "description": (
            "Calculate deterministically how much a customer gets back if "
            "they cancel a booking, following CityWheels cancellation rules. "
            "Always call this tool whenever a customer asks about cancelling "
            "a specific booking or how much money/refund they will get back."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "booking_id": {
                    "type": "STRING",
                    "description": (
                        "The booking identifier to calculate the refund for, "
                        "e.g. 'CW-1002'."
                    ),
                },
                "cancel_at": {
                    "type": "STRING",
                    "description": (
                        "Optional ISO 8601 timestamp when the cancellation "
                        "takes place (e.g. '2026-06-11T09:00:00+02:00'). "
                        "Omit this parameter to use the current time "
                        "(2026-06-10T09:00:00+02:00)."
                    ),
                },
            },
            "required": ["booking_id"],
        },
    },
]


class ToolRegistry:
    """Dispatches tool calls requested by the LLM to Python functions."""

    def __init__(
        self,
        policies_dir: Path | None = None,
        bookings_path: Path | None = None,
        search_mode: str = "auto",
    ) -> None:
        self.policies_dir = policies_dir
        self.bookings_path = bookings_path
        self.search_mode = search_mode
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {
            "search_policies": self._run_search_policies,
            "get_booking": self._run_get_booking,
            "calculate_refund": self._run_calculate_refund,
        }

    def register(
        self,
        name: str,
        func: Callable[..., dict[str, Any]],
    ) -> None:
        """Register an additional custom tool callable."""
        self._tools[name] = func

    def _run_search_policies(
        self,
        query: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return search_policies(
            query=query,
            top_k=int(kwargs.get("top_k", 4)),
            mode=self.search_mode,
            policies_dir=self.policies_dir,
        )

    def _run_get_booking(
        self,
        booking_id: str = "",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return get_booking(
            booking_id=booking_id,
            bookings_path=self.bookings_path,
        )

    def _run_calculate_refund(
        self,
        booking_id: str = "",
        cancel_at: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return calculate_refund(
            booking_id=booking_id,
            cancel_at=cancel_at,
            bookings_path=self.bookings_path,
        )

    def execute(
        self,
        name: str,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a registered tool safely and return a dictionary result."""
        call_args = args or {}
        handler = self._tools.get(name)
        if handler is None:
            return {
                "error": f"Unknown tool '{name}'.",
                "available_tools": list(self._tools.keys()),
            }
        try:
            return handler(**call_args)
        except Exception as exc:
            return {
                "error": f"Tool '{name}' failed with error: {exc}",
            }
