"""Unit tests for every rule and boundary condition of calculate_refund."""

from __future__ import annotations

from help_desk_agent.tools.refunds import calculate_refund


def test_refund_more_than_48_hours_before_start() -> None:
    """CW-1001 starts 2026-06-13 10:00 (73h after fixed current time).

    Rule: >= 48h -> 100% rental refund + 100% deposit, 0% booking fee.
    """
    res = calculate_refund("CW-1001")
    assert res["found"] is True
    assert res["can_cancel"] is True
    assert res["hours_before_start"] == 73.0
    assert res["refund_percentage"] == 100
    assert res["rental_refund_eur"] == 40.0
    assert res["deposit_refund_eur"] == 50.0
    assert res["booking_fee_refund_eur"] == 0.0
    assert res["total_refund_eur"] == 90.0
    assert res["paid_with"] == "card"


def test_refund_boundary_exactly_48_hours_before_start() -> None:
    """CW-1007 starts 2026-06-12 09:00 (exactly 48.0h after 2026-06-10 09:00).

    Rule: '48 hours or more before the start' -> 100% rental refund.
    """
    res = calculate_refund("CW-1007")
    assert res["found"] is True
    assert res["can_cancel"] is True
    assert res["hours_before_start"] == 48.0
    assert res["refund_percentage"] == 100
    assert res["rental_refund_eur"] == 36.0
    assert res["deposit_refund_eur"] == 150.0
    assert res["booking_fee_refund_eur"] == 0.0
    assert res["total_refund_eur"] == 186.0


def test_refund_between_24_and_48_hours_before_start() -> None:
    """CW-1002 starts 2026-06-11 18:00 (33h after fixed current time).

    Rule: 24h <= t < 48h -> 50% rental refund (40 EUR) + deposit (300 EUR).
    Booking fee (2 EUR) is never refunded.
    """
    res = calculate_refund("CW-1002")
    assert res["found"] is True
    assert res["can_cancel"] is True
    assert res["hours_before_start"] == 33.0
    assert res["refund_percentage"] == 50
    assert res["rental_refund_eur"] == 40.0
    assert res["deposit_refund_eur"] == 300.0
    assert res["booking_fee_refund_eur"] == 0.0
    assert res["total_refund_eur"] == 340.0


def test_refund_boundary_exactly_24_hours_before_start() -> None:
    """CW-1010 starts 2026-06-11 09:00 (exactly 24.0h after 2026-06-10 09:00).

    Rule: 'Less than 48 hours, but 24 hours or more' -> 50% rental refund.
    """
    res = calculate_refund("CW-1010")
    assert res["found"] is True
    assert res["can_cancel"] is True
    assert res["hours_before_start"] == 24.0
    assert res["refund_percentage"] == 50
    assert res["rental_refund_eur"] == 20.0
    assert res["deposit_refund_eur"] == 150.0
    assert res["booking_fee_refund_eur"] == 0.0
    assert res["total_refund_eur"] == 170.0


def test_refund_less_than_24_hours_before_start() -> None:
    """CW-1003 starts 2026-06-10 15:00 (6h after fixed current time).

    Rule: < 24h -> 0% rental refund (0 EUR), but deposit (50 EUR) IS refunded!
    """
    res = calculate_refund("CW-1003")
    assert res["found"] is True
    assert res["can_cancel"] is True
    assert res["hours_before_start"] == 6.0
    assert res["refund_percentage"] == 0
    assert res["rental_refund_eur"] == 0.0
    assert res["deposit_refund_eur"] == 50.0
    assert res["booking_fee_refund_eur"] == 0.0
    assert res["total_refund_eur"] == 50.0


def test_refund_paid_with_voucher() -> None:
    """CW-1004 is paid with a voucher and starts 2026-06-20 09:00 (240h away).

    Rule: Voucher bookings are refunded as a new voucher of the same value,
    never as money. Deposit is refunded the same way it was paid.
    """
    res = calculate_refund("CW-1004")
    assert res["found"] is True
    assert res["can_cancel"] is True
    assert res["paid_with"] == "voucher"
    assert res["refund_percentage"] == 100
    assert res["rental_refund_eur"] == 23.0
    assert res["deposit_refund_eur"] == 50.0
    assert res["total_refund_eur"] == 73.0
    assert "new voucher" in res["refund_method_explanation"].lower()
    assert "never as cash" in res["refund_method_explanation"].lower()


def test_refund_already_cancelled_booking() -> None:
    """CW-1005 has status 'cancelled' -> cannot be cancelled, 0 EUR refund."""
    res = calculate_refund("CW-1005")
    assert res["found"] is True
    assert res["status"] == "cancelled"
    assert res["can_cancel"] is False
    assert res["rental_refund_eur"] == 0.0
    assert res["deposit_refund_eur"] == 0.0
    assert res["total_refund_eur"] == 0.0


def test_refund_completed_booking() -> None:
    """CW-1006 has status 'completed' -> cannot be cancelled, 0 EUR refund."""
    res = calculate_refund("CW-1006")
    assert res["found"] is True
    assert res["status"] == "completed"
    assert res["can_cancel"] is False
    assert res["rental_refund_eur"] == 0.0
    assert res["deposit_refund_eur"] == 0.0
    assert res["total_refund_eur"] == 0.0


def test_refund_in_progress_booking() -> None:
    """CW-1008 has status 'in_progress' -> cannot be cancelled, 0 EUR."""
    res = calculate_refund("CW-1008")
    assert res["found"] is True
    assert res["status"] == "in_progress"
    assert res["can_cancel"] is False
    assert res["rental_refund_eur"] == 0.0
    assert res["deposit_refund_eur"] == 0.0
    assert res["total_refund_eur"] == 0.0


def test_refund_unknown_booking() -> None:
    """Unknown booking ID does not crash and returns found=False."""
    res = calculate_refund("CW-9999")
    assert res["found"] is False
    assert res["can_cancel"] is False
    assert res["total_refund_eur"] == 0.0
    assert "not found" in res["explanation"].lower()


def test_refund_with_custom_cancel_at_time() -> None:
    """Test follow-up scenario: cancelling CW-1002 tomorrow at 10:00 instead.

    CW-1002 starts 2026-06-11 18:00. If cancelled at 2026-06-11 10:00 (8h
    before start), rental refund drops from 50% to 0%, leaving only 300 EUR
    deposit.
    """
    res = calculate_refund(
        "CW-1002",
        cancel_at="2026-06-11T10:00:00+02:00",
    )
    assert res["found"] is True
    assert res["hours_before_start"] == 8.0
    assert res["refund_percentage"] == 0
    assert res["rental_refund_eur"] == 0.0
    assert res["deposit_refund_eur"] == 300.0
    assert res["total_refund_eur"] == 300.0
