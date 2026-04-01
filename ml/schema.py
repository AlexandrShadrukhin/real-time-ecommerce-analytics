from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "baseline_model.json"
SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
METRICS_PATH = ARTIFACTS_DIR / "training_metrics.json"

MODEL_VERSION = "baseline-logreg-v1"
MODEL_FEATURES = ["price", "is_view", "is_click", "is_cart", "hour_of_day"]
METADATA_FIELDS = ["event_id", "user_id", "timestamp"]
REQUIRED_PREDICTION_FIELDS = METADATA_FIELDS + MODEL_FEATURES


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
        "is_view": int(payload["is_view"]),
        "is_click": int(payload["is_click"]),
        "is_cart": int(payload["is_cart"]),
        "hour_of_day": int(payload["hour_of_day"]),
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
            "is_view": "0|1",
            "is_click": "0|1",
            "is_cart": "0|1",
            "hour_of_day": "0-23",
        },
        "prediction_response": {
            "event_id": "string",
            "user_id": "integer",
            "purchase_probability": "float",
            "model_version": "string",
        },
    }
