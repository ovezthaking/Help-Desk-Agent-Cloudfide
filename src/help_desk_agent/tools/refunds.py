"""Deterministic refund calculation tool for CityWheels Help Desk Agent."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from help_desk_agent.config import FIXED_CURRENT_TIME
from help_desk_agent.models import Booking, RefundResult
from help_desk_agent.tools.bookings import get_booking


def _parse_cancel_time(
    cancel_at: str | datetime | None,
    reference_tz: Any,
) -> datetime:
    """Parse cancellation timestamp, defaulting to FIXED_CURRENT_TIME."""
    if cancel_at is None:
        return FIXED_CURRENT_TIME
    if isinstance(cancel_at, datetime):
        if cancel_at.tzinfo is None:
            return cancel_at.replace(tzinfo=reference_tz)
        return cancel_at

    cleaned = cancel_at.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=reference_tz)
    return parsed


def calculate_refund(
    booking_id: str,
    cancel_at: str | None = None,
    bookings_path: Path | None = None,
) -> dict[str, Any]:
    """Calculate how much a customer gets back if they cancel a booking.

    All amounts are calculated deterministically in Python following
    data/policies/cancellations.md.
    """
    lookup = get_booking(booking_id=booking_id, bookings_path=bookings_path)
    if not lookup.get("found"):
        result = RefundResult(
            found=False,
            booking_id=str(booking_id or "").strip().upper(),
            can_cancel=False,
            explanation=lookup.get(
                "message", f"Booking '{booking_id}' was not found."
            ),
        )
        return result.model_dump()

    booking = Booking.model_validate(lookup["booking"])
    start_dt = datetime.fromisoformat(booking.start)

    try:
        current_dt = _parse_cancel_time(cancel_at, start_dt.tzinfo)
    except ValueError:
        current_dt = FIXED_CURRENT_TIME

    hours_before = round(
        (start_dt - current_dt).total_seconds() / 3600.0,
        2,
    )

    # Rule: Bookings with status in_progress, completed or cancelled
    # cannot be cancelled, and nothing is refunded for them.
    if booking.status in ("in_progress", "completed", "cancelled"):
        result = RefundResult(
            found=True,
            booking_id=booking.id,
            status=booking.status,
            can_cancel=False,
            cancelled_at=current_dt.isoformat(),
            booking_start=booking.start,
            hours_before_start=hours_before,
            rental_price_eur=booking.rental_price_eur,
            refund_percentage=0,
            rental_refund_eur=0.0,
            deposit_refund_eur=0.0,
            booking_fee_eur=booking.booking_fee_eur,
            booking_fee_refund_eur=0.0,
            total_refund_eur=0.0,
            paid_with=booking.paid_with,
            refund_method_explanation=(
                "No refund is issued because the booking cannot be cancelled."
            ),
            explanation=(
                f"Booking {booking.id} has status '{booking.status}'. "
                "Bookings with status 'in_progress', 'completed', or "
                "'cancelled' cannot be cancelled, and nothing (0.00 EUR) "
                "is refunded."
            ),
        )
        return result.model_dump()

    # Status is 'confirmed'
    if hours_before >= 48.0:
        refund_pct = 100
        rental_refund = round(booking.rental_price_eur, 2)
        window_desc = (
            f"{hours_before:g} hours before start "
            "(48 hours or more -> 100% rental price refund)"
        )
    elif hours_before >= 24.0:
        refund_pct = 50
        rental_refund = round(booking.rental_price_eur * 0.5, 2)
        window_desc = (
            f"{hours_before:g} hours before start "
            "(between 24 and 48 hours -> 50% rental price refund)"
        )
    else:
        refund_pct = 0
        rental_refund = 0.0
        window_desc = (
            f"{hours_before:g} hours before start "
            "(less than 24 hours -> 0% rental price refund)"
        )

    # Deposit is always refunded in full when a confirmed booking is cancelled.
    deposit_refund = round(booking.deposit_eur, 2)
    # Booking fee is never refunded.
    booking_fee_refund = 0.0
    total_refund = round(rental_refund + deposit_refund, 2)

    if booking.paid_with == "card":
        method_explanation = (
            "Paid by card: refunded to the same card within 5 working days."
        )
    else:
        method_explanation = (
            "Paid with voucher: rental price refund "
            f"({rental_refund:.2f} EUR) is refunded as a new voucher of the "
            "same value (never as cash/money). The deposit "
            f"({deposit_refund:.2f} EUR) is refunded the same way it was paid."
        )

    explanation = (
        f"Booking {booking.id} (status: confirmed) cancelled at "
        f"{current_dt.isoformat()}, which is {window_desc}. "
        f"Rental refund: {rental_refund:.2f} EUR ({refund_pct}% of "
        f"{booking.rental_price_eur:.2f} EUR). "
        f"Deposit refund: {deposit_refund:.2f} EUR (100% of "
        f"{booking.deposit_eur:.2f} EUR). "
        f"Booking fee of {booking.booking_fee_eur:.2f} EUR is non-refundable "
        "(0.00 EUR). "
        f"Total value returned: {total_refund:.2f} EUR. {method_explanation}"
    )

    result = RefundResult(
        found=True,
        booking_id=booking.id,
        status=booking.status,
        can_cancel=True,
        cancelled_at=current_dt.isoformat(),
        booking_start=booking.start,
        hours_before_start=hours_before,
        rental_price_eur=booking.rental_price_eur,
        refund_percentage=refund_pct,
        rental_refund_eur=rental_refund,
        deposit_refund_eur=deposit_refund,
        booking_fee_eur=booking.booking_fee_eur,
        booking_fee_refund_eur=booking_fee_refund,
        total_refund_eur=total_refund,
        paid_with=booking.paid_with,
        refund_method_explanation=method_explanation,
        explanation=explanation,
    )
    return result.model_dump()
