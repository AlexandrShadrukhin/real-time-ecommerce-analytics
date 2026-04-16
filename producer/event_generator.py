from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Any

EVENT_TYPES = ["view", "view", "view", "add_to_cart", "click"]
CATEGORY_CODES = [
    "electronics.smartphones",
    "electronics.laptops",
    "home.kitchen",
    "fashion.shoes",
    "fashion.tshirts",
    "books.fiction",
]
BRANDS = ["apple", "samsung", "nike", "adidas", "xiaomi", "lenovo", None]
DEVICE_TYPES = ["mobile", "desktop", "tablet"]
SOURCES = ["search", "catalog", "recommendation", "ads"]


def _sample_price(category_code: str | None) -> float:
    if not category_code:
        return round(random.uniform(100, 3000), 2)

    if category_code.startswith("electronics"):
        return round(random.uniform(1000, 6000), 2)
    if category_code.startswith("fashion"):
        return round(random.uniform(300, 2500), 2)
    if category_code.startswith("books"):
        return round(random.uniform(100, 1500), 2)
    return round(random.uniform(200, 4000), 2)


def generate_event(
    user_id: int | None = None,
    session_id: str | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    event_timestamp = timestamp or datetime.utcnow()
    category_code = random.choice(CATEGORY_CODES + [None] * 1)
    brand = random.choice(BRANDS)

    return {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id if user_id is not None else random.randint(1, 20),
        "session_id": session_id or f"session_{uuid.uuid4().hex[:8]}",
        "product_id": random.randint(1, 200),
        "event_type": random.choice(EVENT_TYPES),
        "timestamp": event_timestamp.isoformat(),
        "price": _sample_price(category_code),
        "category_code": category_code,
        "brand": brand,
        "device_type": random.choice(DEVICE_TYPES),
        "source": random.choice(SOURCES),
    }


def generate_session_events(
    user_id: int | None = None,
    session_length: int = 5,
    start_time: datetime | None = None,
) -> list[dict[str, Any]]:
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    base_time = start_time or datetime.utcnow()
    resolved_user_id = user_id if user_id is not None else random.randint(1, 20)

    events: list[dict[str, Any]] = []
    current_time = base_time

    for step in range(session_length):
        event = generate_event(
            user_id=resolved_user_id,
            session_id=session_id,
            timestamp=current_time,
        )

        if step == 0:
            event["event_type"] = "view"
        elif step == 1 and random.random() < 0.25:
            event["event_type"] = "add_to_cart"

        events.append(event)
        current_time += timedelta(seconds=random.randint(5, 60))

    return events


def generate_event_stream(
    session_count: int = 3,
    min_session_length: int = 3,
    max_session_length: int = 7,
) -> list[dict[str, Any]]:
    all_events: list[dict[str, Any]] = []
    current_time = datetime.utcnow()

    for _ in range(session_count):
        session_events = generate_session_events(
            user_id=random.randint(1, 20),
            session_length=random.randint(min_session_length, max_session_length),
            start_time=current_time,
        )
        all_events.extend(session_events)
        current_time += timedelta(minutes=random.randint(1, 5))

    return all_events


if __name__ == "__main__":
    for item in generate_event_stream(session_count=2):
        print(item)