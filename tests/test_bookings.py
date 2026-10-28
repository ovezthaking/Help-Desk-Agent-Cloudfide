"""Unit tests for the get_booking tool."""

from __future__ import annotations

from pathlib import Path

from help_desk_agent.tools.bookings import get_booking, load_bookings


def test_load_bookings_count() -> None:
    """Verify that all 10 sample bookings load properly."""
    bookings = load_bookings()
    assert len(bookings) == 10
    assert bookings[0].id == "CW-1001"


def test_get_booking_existing() -> None:
    """Verify retrieving an existing booking by exact ID."""
    res = get_booking("CW-1002")
    assert res["found"] is True
    booking = res["booking"]
    assert booking["id"] == "CW-1002"
    assert booking["status"] == "confirmed"
    assert booking["bike_type"] == "e-bike"
    assert booking["bikes"] == 2
    assert booking["rental_price_eur"] == 80.0
    assert booking["booking_fee_eur"] == 2.0
    assert booking["deposit_eur"] == 300.0
    assert booking["paid_with"] == "card"


def test_get_booking_case_insensitive_and_whitespace() -> None:
    """Verify booking lookup normalizes lowercase and surrounding spaces."""
    res = get_booking("  cw-1004  ")
    assert res["found"] is True
    assert res["booking"]["id"] == "CW-1004"
    assert res["booking"]["paid_with"] == "voucher"
    assert res["booking"]["add_ons"] == ["child_seat"]


def test_get_booking_not_found() -> None:
    """Verify an unknown booking number returns a clear not-found dict."""
    res = get_booking("CW-9999")
    assert res["found"] is False
    assert res["booking_id"] == "CW-9999"
    assert "not found" in res["message"].lower()


def test_get_booking_empty_or_invalid_id() -> None:
    """Verify empty string does not crash the tool."""
    res = get_booking("")
    assert res["found"] is False
    assert "not provided" in res["message"].lower()


def test_get_booking_missing_file(tmp_path: Path) -> None:
    """Verify graceful handling when bookings.json does not exist."""
    missing = tmp_path / "nonexistent.json"
    res = get_booking("CW-1001", bookings_path=missing)
    assert res["found"] is False
