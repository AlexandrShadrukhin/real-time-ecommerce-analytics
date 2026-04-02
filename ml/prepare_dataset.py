from __future__ import annotations

import math
import random
import uuid
from collections import Counter
from datetime import datetime, timedelta
from typing import Any


def _build_user_profiles(user_count: int, rng: random.Random) -> dict[int, dict[str, float]]:
    profiles = {}
    for user_id in range(1, user_count + 1):
        profiles[user_id] = {
            "engagement": rng.uniform(0.8, 1.25),
            "cart_affinity": rng.uniform(0.6, 1.35),
            "purchase_affinity": rng.uniform(-0.75, 0.95),
            "preferred_price": rng.uniform(600.0, 3200.0),
            "preferred_hour": rng.randint(9, 22),
        }
    return profiles


def _sample_session_length(profile: dict[str, float], rng: random.Random) -> int:
    base_length = rng.randint(3, 8)
    if profile["engagement"] > 1.1:
        base_length += 1
    return max(2, min(base_length, 10))


def _sample_price(profile: dict[str, float], rng: random.Random) -> float:
    deviation = rng.uniform(0.55, 1.6)
    return round(profile["preferred_price"] * deviation, 2)


def _sample_event_type(
    *,
    step: int,
    session_length: int,
    session_clicks: int,
    session_carts: int,
    profile: dict[str, float],
    rng: random.Random,
) -> str:
    progress = step / max(session_length, 1)
    view_weight = max(0.2, 1.15 - progress * 0.55)
    click_weight = 0.55 + progress * 0.75 * profile["engagement"]
    cart_weight = 0.18 + progress * 0.95 * profile["cart_affinity"]

    if session_clicks == 0:
        click_weight += 0.18
    if session_carts > 0:
        cart_weight += 0.28
        click_weight += 0.1

    event_types = ["view", "click", "add_to_cart"]
    weights = [view_weight, click_weight, cart_weight]
    return rng.choices(event_types, weights=weights, k=1)[0]


def _sigmoid(value: float) -> float:
    bounded = max(min(value, 35.0), -35.0)
    return 1.0 / (1.0 + math.exp(-bounded))


def _build_training_row(
    *,
    user_id: int,
    timestamp: datetime,
    event_type: str,
    price: float,
    price_to_user_mean: float,
    session_event_index: int,
    session_click_count: int,
    session_cart_count: int,
    user_total_clicks: int,
    user_total_carts: int,
    user_total_purchases: int,
    label: int,
) -> dict[str, Any]:
    return {
        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
        "user_id": user_id,
        "timestamp": timestamp.isoformat(),
        "event_type": event_type,
        "price": float(price),
        "price_to_user_mean": round(price_to_user_mean, 4),
        "hour_of_day": timestamp.hour,
        "is_view": int(event_type == "view"),
        "is_click": int(event_type == "click"),
        "is_cart": int(event_type == "add_to_cart"),
        "session_event_index": session_event_index,
        "session_click_count": session_click_count,
        "session_cart_count": session_cart_count,
        "user_total_clicks": user_total_clicks,
        "user_total_carts": user_total_carts,
        "user_total_purchases": user_total_purchases,
        "label": int(label),
    }


def build_synthetic_dataset(
    sample_count: int = 5000,
    *,
    seed: int = 42,
    user_count: int = 350,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    user_profiles = _build_user_profiles(user_count, rng)
    user_state = {
        user_id: {
            "clicks": 0,
            "carts": 0,
            "purchases": 0,
            "mean_price": profile["preferred_price"],
        }
        for user_id, profile in user_profiles.items()
    }

    rows: list[dict[str, Any]] = []
    base_datetime = datetime(2026, 4, 1, 8, 0, 0)

    while len(rows) < sample_count:
        user_id = rng.randint(1, user_count)
        profile = user_profiles[user_id]
        state = user_state[user_id]

        session_length = _sample_session_length(profile, rng)
        hour_of_day = int((profile["preferred_hour"] + rng.randint(-3, 3)) % 24)
        session_start = base_datetime + timedelta(
            minutes=rng.randint(0, 60 * 24 * 14),
            hours=hour_of_day - base_datetime.hour,
        )

        session_events: list[tuple[str, float, datetime]] = []
        session_click_count = 0
        session_cart_count = 0

        for step in range(1, session_length + 1):
            event_type = _sample_event_type(
                step=step,
                session_length=session_length,
                session_clicks=session_click_count,
                session_carts=session_cart_count,
                profile=profile,
                rng=rng,
            )
            price = _sample_price(profile, rng)
            timestamp = session_start + timedelta(minutes=step * rng.randint(1, 4))
            session_events.append((event_type, price, timestamp))

            session_click_count += int(event_type == "click")
            session_cart_count += int(event_type == "add_to_cart")

        running_clicks = 0
        running_carts = 0
        session_had_purchase = False
        for step, (event_type, price, timestamp) in enumerate(session_events, start=1):
            running_clicks += int(event_type == "click")
            running_carts += int(event_type == "add_to_cart")
            price_to_user_mean = float(price) / max(state["mean_price"], 1.0)

            row_logit = -4.2
            row_logit += 0.16 * step
            row_logit += 0.45 * running_clicks
            row_logit += 0.95 * running_carts
            row_logit += 0.1 * min(state["clicks"], 12)
            row_logit += 0.28 * min(state["carts"], 8)
            row_logit += 0.38 * min(state["purchases"], 5)
            row_logit += 0.16 if 18 <= timestamp.hour <= 22 else -0.08
            row_logit -= 0.9 * max(price_to_user_mean - 1.0, 0.0)
            row_logit += 0.18 * max(1.0 - price_to_user_mean, 0.0)
            row_logit += 0.18 * int(event_type == "click")
            row_logit += 0.48 * int(event_type == "add_to_cart")
            row_logit += profile["purchase_affinity"]
            row_logit += rng.uniform(-0.22, 0.22)

            label = int(rng.random() < _sigmoid(row_logit))
            session_had_purchase = session_had_purchase or bool(label)

            rows.append(
                _build_training_row(
                    user_id=user_id,
                    timestamp=timestamp,
                    event_type=event_type,
                    price=price,
                    price_to_user_mean=price_to_user_mean,
                    session_event_index=step,
                    session_click_count=running_clicks,
                    session_cart_count=running_carts,
                    user_total_clicks=state["clicks"],
                    user_total_carts=state["carts"],
                    user_total_purchases=state["purchases"],
                    label=label,
                )
            )
            if len(rows) >= sample_count:
                break

        state["clicks"] += session_click_count
        state["carts"] += session_cart_count
        state["purchases"] += int(session_had_purchase)
        session_average_price = sum(price for _, price, _ in session_events) / len(session_events)
        state["mean_price"] = round((state["mean_price"] * 0.85) + (session_average_price * 0.15), 2)

    return rows[:sample_count]


def summarize_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Dataset is empty.")

    event_counter = Counter(row["event_type"] for row in rows)
    label_counter = Counter(row["label"] for row in rows)

    return {
        "row_count": len(rows),
        "unique_users": len({row["user_id"] for row in rows}),
        "positive_rate": label_counter[1] / len(rows),
        "average_price": round(sum(row["price"] for row in rows) / len(rows), 2),
        "average_session_depth": round(
            sum(row["session_event_index"] for row in rows) / len(rows),
            2,
        ),
        "average_user_clicks": round(
            sum(row["user_total_clicks"] for row in rows) / len(rows),
            2,
        ),
        "average_user_carts": round(
            sum(row["user_total_carts"] for row in rows) / len(rows),
            2,
        ),
        "event_mix": {
            event_type: round(count / len(rows), 4)
            for event_type, count in sorted(event_counter.items())
        },
    }
