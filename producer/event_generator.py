from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Any

EVENT_TYPES = ["view", "click", "add_to_cart", "purchase"]
PRODUCT_CATEGORIES = ["electronics", "books", "fashion", "home"]
DEVICE_TYPES = ["mobile", "desktop", "tablet"]
SOURCES = ["search", "catalog", "recommendation", "ads"]


def generate_event(
    user_id: int | None = None,
    session_id: str | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """
    Сгенерировать одно пользовательское событие, пригодное для построения
    признаков под ML-контракт.
    """
    event_timestamp = timestamp or datetime.utcnow()

    return {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id if user_id is not None else random.randint(1, 20),
        "session_id": session_id or f"session_{uuid.uuid4().hex[:8]}",
        "product_id": random.randint(1, 500),
        "event_type": random.choice(EVENT_TYPES),
        "timestamp": event_timestamp.isoformat(),
        "price": round(random.uniform(100, 5000), 2),
        "category": random.choice(PRODUCT_CATEGORIES),
        "device_type": random.choice(DEVICE_TYPES),
        "source": random.choice(SOURCES),
    }


def generate_session_events(
    user_id: int | None = None,
    session_length: int = 5,
    start_time: datetime | None = None,
) -> list[dict[str, Any]]:
    """
    Сгенерировать последовательность событий в рамках одной сессии.
    """
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    base_time = start_time or datetime.utcnow()
    resolved_user_id = user_id if user_id is not None else random.randint(1, 20)

    events: list[dict[str, Any]] = []
    current_time = base_time

    weighted_event_types = ["view", "view", "click", "click", "add_to_cart", "purchase"]

    for _ in range(session_length):
        event = generate_event(
            user_id=resolved_user_id,
            session_id=session_id,
            timestamp=current_time,
        )
        event["event_type"] = random.choice(weighted_event_types)
        events.append(event)

        current_time += timedelta(seconds=random.randint(5, 45))

    return events


def generate_event_stream(
    session_count: int = 3,
    min_session_length: int = 3,
    max_session_length: int = 7,
) -> list[dict[str, Any]]:
    """
    Сгенерировать поток событий из нескольких пользовательских сессий.
    """
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