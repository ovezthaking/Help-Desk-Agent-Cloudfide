"""Booking lookup tool for CityWheels Help Desk Agent."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from help_desk_agent.config import BOOKINGS_PATH
from help_desk_agent.models import Booking


@lru_cache(maxsize=8)
def _load_bookings_index_cached(
    resolved_path: Path,
    mtime_ns: int,
) -> dict[str, Booking]:
    """Load and index bookings by uppercase ID, cached by file mtime."""
    del mtime_ns  # Used as part of the cache key for invalidation
    raw_data = json.loads(resolved_path.read_text(encoding="utf-8"))
    return {
        booking.id.upper(): booking
        for item in raw_data
        if (booking := Booking.model_validate(item))
    }


def _get_bookings_index(
    bookings_path: Path | None = None,
) -> dict[str, Booking]:
    """Return the uppercase-ID-keyed booking dictionary for the given path."""
    path = (bookings_path or BOOKINGS_PATH).resolve()
    if not path.exists():
        return {}
    return _load_bookings_index_cached(path, path.stat().st_mtime_ns)


def load_bookings(bookings_path: Path | None = None) -> list[Booking]:
    """Load and validate all bookings from the JSON file."""
    return list(_get_bookings_index(bookings_path).values())


def find_booking_model(
    booking_id: str,
    bookings_path: Path | None = None,
) -> Booking | None:
    """Look up a validated Booking model in O(1) time by booking ID."""
    if not booking_id or not isinstance(booking_id, str):
        return None
    normalized_id = booking_id.strip().upper()
    return _get_bookings_index(bookings_path).get(normalized_id)


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
    booking = _get_bookings_index(bookings_path).get(normalized_id)
    if booking is not None:
        return {
            "found": True,
            "booking": booking.model_dump(),
        }

    return {
        "found": False,
        "booking_id": normalized_id,
        "message": f"Booking '{booking_id.strip()}' was not found.",
    }
