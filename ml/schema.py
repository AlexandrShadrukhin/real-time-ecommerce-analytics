from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
MODEL_PATH = PROJECT_ROOT / "ml" / "model" / "model.cbm"
SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"

MODEL_VERSION = "catboost-v1"
MODEL_FEATURES = [
    "n_views",
    "n_carts",
    "uniq_products",
    "uniq_categories",
    "mean_price",
    "max_price",
    "price_std",
    "uniq_brands",
    "missing_brand",
    "missing_category_code",
    "prefix_duration_sec",
    "has_cart_in_prefix",
    "cart_view_ratio",
    "same_product_repeat_count",
    "same_category_repeat_count",
    "event_1_is_cart",
    "event_2_is_cart",
    "event_3_is_cart",
    "time_1_to_2_sec",
    "time_2_to_3_sec",
]

METADATA_FIELDS = ["event_id", "user_id", "session_id", "timestamp"]
REQUIRED_PREDICTION_FIELDS = METADATA_FIELDS + MODEL_FEATURES

FEATURE_DESCRIPTIONS = {
    "n_views": "Number of product views in the current session prefix.",
    "n_carts": "Number of add-to-cart events in the current session prefix.",
    "uniq_products": "Count of unique products in the current session prefix.",
    "uniq_categories": "Count of unique category codes in the current session prefix.",
    "mean_price": "Average product price in the current session prefix.",
    "max_price": "Maximum product price in the current session prefix.",
    "price_std": "Standard deviation of product prices in the current session prefix.",
    "uniq_brands": "Count of unique brands in the current session prefix.",
    "missing_brand": "Number of events with missing brand in the current session prefix.",
    "missing_category_code": "Number of events with missing category code in the current session prefix.",
    "prefix_duration_sec": "Duration of the current session prefix in seconds.",
    "has_cart_in_prefix": "Whether there is at least one cart event in the current session prefix.",
    "cart_view_ratio": "Ratio of cart events to view events in the current session prefix.",
    "same_product_repeat_count": "How many repeated product visits happened in the current session prefix.",
    "same_category_repeat_count": "How many repeated category visits happened in the current session prefix.",
    "event_1_is_cart": "Whether the first event in the session is add-to-cart.",
    "event_2_is_cart": "Whether the second event in the session is add-to-cart.",
    "event_3_is_cart": "Whether the third event in the session is add-to-cart.",
    "time_1_to_2_sec": "Time between the first and second events in seconds.",
    "time_2_to_3_sec": "Time between the second and third events in seconds.",
}


def normalize_prediction_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing_fields = [field for field in REQUIRED_PREDICTION_FIELDS if field not in payload]
    if missing_fields:
        raise KeyError(f"Missing prediction fields: {', '.join(missing_fields)}")

    normalized = {
        "event_id": str(payload["event_id"]),
        "user_id": int(payload["user_id"]),
        "session_id": str(payload["session_id"]),
        "timestamp": str(payload["timestamp"]),
    }

    for feature_name in MODEL_FEATURES:
        normalized[feature_name] = float(payload[feature_name])

    return normalized


def build_schema_document() -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "model_features": MODEL_FEATURES,
        "prediction_request": {
            "event_id": "string",
            "user_id": "integer",
            "session_id": "string",
            "timestamp": "ISO-8601 datetime",
            **{feature_name: "float" for feature_name in MODEL_FEATURES},
        },
        "prediction_response": {
            "event_id": "string",
            "user_id": "integer",
            "session_id": "string",
            "purchase_probability": "float",
            "model_version": "string",
        },
        "feature_descriptions": FEATURE_DESCRIPTIONS,
    }