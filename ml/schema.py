from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "baseline_model.json"
SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
METRICS_PATH = ARTIFACTS_DIR / "training_metrics.json"

MODEL_VERSION = "baseline-logreg-v2"
MODEL_FEATURES = [
    "price",
    "price_to_user_mean",
    "hour_of_day",
    "is_view",
    "is_click",
    "is_cart",
    "session_event_index",
    "session_click_count",
    "session_cart_count",
    "user_total_clicks",
    "user_total_carts",
    "user_total_purchases",
]
METADATA_FIELDS = ["event_id", "user_id", "timestamp"]
REQUIRED_PREDICTION_FIELDS = METADATA_FIELDS + MODEL_FEATURES
FEATURE_DESCRIPTIONS = {
    "price": "Current event price.",
    "price_to_user_mean": "Current price divided by the user's historical average price.",
    "hour_of_day": "Hour extracted from event timestamp.",
    "is_view": "Current event is a product view.",
    "is_click": "Current event is a click.",
    "is_cart": "Current event is an add-to-cart action.",
    "session_event_index": "Position of the event within the current session.",
    "session_click_count": "Clicks observed in the current session up to this event.",
    "session_cart_count": "Cart additions observed in the current session up to this event.",
    "user_total_clicks": "Historical clicks for the user before the current session.",
    "user_total_carts": "Historical cart additions for the user before the current session.",
    "user_total_purchases": "Historical completed purchases for the user.",
}


def parse_hour_of_day(timestamp: str) -> int:
    normalized = timestamp.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).hour


def normalize_prediction_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing_fields = [field for field in REQUIRED_PREDICTION_FIELDS if field not in payload]
    if missing_fields:
        raise KeyError(f"Missing prediction fields: {', '.join(missing_fields)}")

    normalized = {
        "event_id": str(payload["event_id"]),
        "user_id": int(payload["user_id"]),
        "timestamp": str(payload["timestamp"]),
        "price": float(payload["price"]),
        "price_to_user_mean": float(payload["price_to_user_mean"]),
        "hour_of_day": int(payload["hour_of_day"]),
        "is_view": int(payload["is_view"]),
        "is_click": int(payload["is_click"]),
        "is_cart": int(payload["is_cart"]),
        "session_event_index": int(payload["session_event_index"]),
        "session_click_count": int(payload["session_click_count"]),
        "session_cart_count": int(payload["session_cart_count"]),
        "user_total_clicks": int(payload["user_total_clicks"]),
        "user_total_carts": int(payload["user_total_carts"]),
        "user_total_purchases": int(payload["user_total_purchases"]),
    }
    return normalized


def build_schema_document() -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "model_features": MODEL_FEATURES,
        "prediction_request": {
            "event_id": "string",
            "user_id": "integer",
            "timestamp": "ISO-8601 datetime",
            "price": "float",
            "price_to_user_mean": "float",
            "hour_of_day": "0-23",
            "is_view": "0|1",
            "is_click": "0|1",
            "is_cart": "0|1",
            "session_event_index": "integer >= 1",
            "session_click_count": "integer >= 0",
            "session_cart_count": "integer >= 0",
            "user_total_clicks": "integer >= 0",
            "user_total_carts": "integer >= 0",
            "user_total_purchases": "integer >= 0",
        },
        "prediction_response": {
            "event_id": "string",
            "user_id": "integer",
            "purchase_probability": "float",
            "model_version": "string",
        },
        "feature_descriptions": FEATURE_DESCRIPTIONS,
    }
