"""Configuration and constants for the CityWheels Help Desk Agent."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.getenv("CITYWHEELS_DATA_DIR", str(PROJECT_ROOT / "data")))
POLICIES_DIR = DATA_DIR / "policies"
BOOKINGS_PATH = DATA_DIR / "bookings.json"
EMBEDDINGS_CACHE_PATH = DATA_DIR / ".embeddings_cache.json"

# Fixed current date/time specified by the CityWheels assignment:
# 2026-06-10 09:00, Europe/Warsaw time (UTC+02:00)
FIXED_CURRENT_TIME_ISO = "2026-06-10T09:00:00+02:00"
FIXED_CURRENT_TIME = datetime.fromisoformat(FIXED_CURRENT_TIME_ISO)
FIXED_CURRENT_TIME_DISPLAY = "2026-06-10 09:00, Europe/Warsaw time (UTC+02:00)"

STAFF_EMAIL = "help@citywheels.example"
STAFF_PHONE = "+48 000 000 000"


def get_gemini_api_key() -> str | None:
    """Return the Gemini API key from the environment, if configured."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and key.strip() and key.strip() != "your-gemini-api-key-here":
        return key.strip()
    return None


def get_gemini_model() -> str:
    """Return the configured Gemini generation model name."""
    return os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()


def get_gemini_embedding_model() -> str:
    """Return the configured Gemini embedding model name."""
    return os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
