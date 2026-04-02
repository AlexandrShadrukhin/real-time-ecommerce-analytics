from __future__ import annotations

from collections import defaultdict
from typing import Any

from ml.schema import parse_hour_of_day


def initialize_state() -> dict[str, Any]:
    """
    Создать in-memory state для MVP pipeline.
    """
    return {
        "sessions": {},
        "users": defaultdict(
            lambda: {
                "total_clicks": 0,
                "total_carts": 0,
                "total_purchases": 0,
                "price_sum": 0.0,
                "price_count": 0,
            }
        ),
    }


def _ensure_session_state(state: dict[str, Any], session_id: str, user_id: int) -> dict[str, Any]:
    sessions = state["sessions"]
    if session_id not in sessions:
        sessions[session_id] = {
            "user_id": user_id,
            "event_index": 0,
            "click_count": 0,
            "cart_count": 0,
        }
    return sessions[session_id]


def build_features(event: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """
    Построить feature payload, совместимый с ml.schema.REQUIRED_PREDICTION_FIELDS.
    Счётчики сессии учитывают текущее событие.
    Исторические user_total_* считаются ДО текущего события.
    """
    user_id = int(event["user_id"])
    session_id = str(event["session_id"])
    event_type = str(event["event_type"])
    price = float(event["price"])
    timestamp = str(event["timestamp"])

    session_state = _ensure_session_state(state, session_id, user_id)
    user_state = state["users"][user_id]

    # История пользователя до текущего события
    user_total_clicks = int(user_state["total_clicks"])
    user_total_carts = int(user_state["total_carts"])
    user_total_purchases = int(user_state["total_purchases"])

    # Средняя цена пользователя до текущего события
    if user_state["price_count"] > 0:
        user_mean_price = user_state["price_sum"] / user_state["price_count"]
    else:
        user_mean_price = price

    price_to_user_mean = price / user_mean_price if user_mean_price > 0 else 1.0

    # Обновляем сессионные счётчики с учётом текущего события
    session_state["event_index"] += 1
    if event_type == "click":
        session_state["click_count"] += 1
    if event_type == "add_to_cart":
        session_state["cart_count"] += 1

    payload = {
        "event_id": str(event["event_id"]),
        "user_id": user_id,
        "timestamp": timestamp,
        "price": price,
        "price_to_user_mean": round(price_to_user_mean, 6),
        "hour_of_day": parse_hour_of_day(timestamp),
        "is_view": int(event_type == "view"),
        "is_click": int(event_type == "click"),
        "is_cart": int(event_type == "add_to_cart"),
        "session_event_index": int(session_state["event_index"]),
        "session_click_count": int(session_state["click_count"]),
        "session_cart_count": int(session_state["cart_count"]),
        "user_total_clicks": user_total_clicks,
        "user_total_carts": user_total_carts,
        "user_total_purchases": user_total_purchases,
    }

    # Обновляем историю пользователя после построения признаков
    if event_type == "click":
        user_state["total_clicks"] += 1
    elif event_type == "add_to_cart":
        user_state["total_carts"] += 1
    elif event_type == "purchase":
        user_state["total_purchases"] += 1

    user_state["price_sum"] += price
    user_state["price_count"] += 1

    return payload