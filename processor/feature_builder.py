from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from typing import Any


def initialize_state() -> dict[str, Any]:
    return {
        "sessions": {},
    }


def _parse_ts(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _ensure_session_state(state: dict[str, Any], session_id: str, user_id: int) -> dict[str, Any]:
    sessions = state["sessions"]
    if session_id not in sessions:
        sessions[session_id] = {
            "user_id": user_id,
            "events": [],
        }
    return sessions[session_id]


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def build_features(event: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    user_id = int(event["user_id"])
    session_id = str(event["session_id"])

    session_state = _ensure_session_state(state, session_id, user_id)
    session_state["events"].append(event)

    prefix_events: list[dict[str, Any]] = session_state["events"]
    event_types = [str(item["event_type"]) for item in prefix_events]
    prices = [float(item["price"]) for item in prefix_events]

    product_ids = [item.get("product_id") for item in prefix_events if item.get("product_id") is not None]
    category_codes = [item.get("category_code") for item in prefix_events if item.get("category_code")]
    brands = [item.get("brand") for item in prefix_events if item.get("brand")]

    timestamps = [_parse_ts(str(item["timestamp"])) for item in prefix_events]

    n_views = sum(1 for item in event_types if item == "view")
    n_carts = sum(1 for item in event_types if item == "add_to_cart")

    uniq_products = len(set(product_ids))
    uniq_categories = len(set(category_codes))
    uniq_brands = len(set(brands))

    mean_price = sum(prices) / len(prices)
    max_price = max(prices)
    price_std = _population_std(prices)

    missing_brand = sum(1 for item in prefix_events if not item.get("brand"))
    missing_category_code = sum(1 for item in prefix_events if not item.get("category_code"))

    prefix_duration_sec = 0.0
    if len(timestamps) >= 2:
        prefix_duration_sec = (timestamps[-1] - timestamps[0]).total_seconds()

    has_cart_in_prefix = 1.0 if n_carts > 0 else 0.0
    cart_view_ratio = float(n_carts / n_views) if n_views > 0 else float(n_carts)

    product_counts = Counter(product_ids)
    category_counts = Counter(category_codes)

    same_product_repeat_count = sum(count - 1 for count in product_counts.values() if count > 1)
    same_category_repeat_count = sum(count - 1 for count in category_counts.values() if count > 1)

    event_1_is_cart = 1.0 if len(event_types) >= 1 and event_types[0] == "add_to_cart" else 0.0
    event_2_is_cart = 1.0 if len(event_types) >= 2 and event_types[1] == "add_to_cart" else 0.0
    event_3_is_cart = 1.0 if len(event_types) >= 3 and event_types[2] == "add_to_cart" else 0.0

    time_1_to_2_sec = 0.0
    time_2_to_3_sec = 0.0
    if len(timestamps) >= 2:
        time_1_to_2_sec = max((timestamps[1] - timestamps[0]).total_seconds(), 0.0)
    if len(timestamps) >= 3:
        time_2_to_3_sec = max((timestamps[2] - timestamps[1]).total_seconds(), 0.0)

    return {
        "event_id": str(event["event_id"]),
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": str(event["timestamp"]),
        "n_views": float(n_views),
        "n_carts": float(n_carts),
        "uniq_products": float(uniq_products),
        "uniq_categories": float(uniq_categories),
        "mean_price": float(mean_price),
        "max_price": float(max_price),
        "price_std": float(price_std),
        "uniq_brands": float(uniq_brands),
        "missing_brand": float(missing_brand),
        "missing_category_code": float(missing_category_code),
        "prefix_duration_sec": float(prefix_duration_sec),
        "has_cart_in_prefix": float(has_cart_in_prefix),
        "cart_view_ratio": float(cart_view_ratio),
        "same_product_repeat_count": float(same_product_repeat_count),
        "same_category_repeat_count": float(same_category_repeat_count),
        "event_1_is_cart": float(event_1_is_cart),
        "event_2_is_cart": float(event_2_is_cart),
        "event_3_is_cart": float(event_3_is_cart),
        "time_1_to_2_sec": float(time_1_to_2_sec),
        "time_2_to_3_sec": float(time_2_to_3_sec),
    }