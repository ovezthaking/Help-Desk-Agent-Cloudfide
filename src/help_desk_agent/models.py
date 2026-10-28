"""Domain and tool data models for the CityWheels Help Desk Agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

BookingStatus = Literal["confirmed", "in_progress", "completed", "cancelled"]
BikeType = Literal["city", "e-bike", "kids"]
PaymentMethod = Literal["card", "voucher"]


class Booking(BaseModel):
    """Represents a single bike rental booking from bookings.json."""

    id: str
    status: BookingStatus
    bike_type: BikeType
    bikes: int
    start: str
    end: str
    add_ons: list[str] = Field(default_factory=list)
    rental_price_eur: float
    booking_fee_eur: float
    deposit_eur: float
    paid_with: PaymentMethod


class RefundResult(BaseModel):
    """Deterministic refund calculation output for calculate_refund tool."""

    found: bool
    booking_id: str
    status: BookingStatus | None = None
    can_cancel: bool = False
    cancelled_at: str | None = None
    booking_start: str | None = None
    hours_before_start: float | None = None
    rental_price_eur: float = 0.0
    refund_percentage: int = 0
    rental_refund_eur: float = 0.0
    deposit_refund_eur: float = 0.0
    booking_fee_eur: float = 0.0
    booking_fee_refund_eur: float = 0.0
    total_refund_eur: float = 0.0
    paid_with: PaymentMethod | None = None
    refund_method_explanation: str = ""
    explanation: str


class PolicyChunk(BaseModel):
    """A section of a policy markdown document."""

    chunk_id: str
    source_file: str
    document_title: str
    section_title: str
    content: str
    score: float = 0.0


class ToolCallRequest(BaseModel):
    """Represents a tool call requested by the LLM."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class ToolCallRecord(BaseModel):
    """Records a tool call and its result during an agent turn."""

    name: str
    args: dict[str, Any]
    result: dict[str, Any]


class LLMResponse(BaseModel):
    """Standardized response from an LLM client."""

    text: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    raw_model_parts: Any = None


class AgentResponse(BaseModel):
    """Final response returned by the HelpDeskAgent."""

    answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    steps: int = 0
