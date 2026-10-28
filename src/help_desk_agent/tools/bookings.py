"""Booking lookup tool for CityWheels Help Desk Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from help_desk_agent.config import BOOKINGS_PATH
from help_desk_agent.models import Booking


def load_bookings(bookings_path: Path | None = None) -> list[Booking]:
    """Load and validate all bookings from the JSON file."""
    path = bookings_path or BOOKINGS_PATH
    if not path.exists():
        return []
    raw_data = json.loads(path.read_text(encoding="utf-8"))
    return [Booking.model_validate(item) for item in raw_data]


def get_booking(
    booking_id: str,
    bookings_path: Path | None = None,
) -> dict[str, Any]:
    """Return the details of one booking from data/bookings.json.

    Returns a clear 'not found' result when the booking does not exist.
    """
    if not booking_id or not isinstance(booking_id, str):
        return {
            "found": False,
            "booking_id": str(booking_id or ""),
            "message": "Booking ID was not provided or is invalid.",
        }

    normalized_id = booking_id.strip().upper()
    bookings = load_bookings(bookings_path=bookings_path)

    for booking in bookings:
        if booking.id.upper() == normalized_id:
            return {
                "found": True,
                "booking": booking.model_dump(),
            }

    return {
        "found": False,
        "booking_id": normalized_id,
        "message": f"Booking '{booking_id.strip()}' was not found.",
    }
