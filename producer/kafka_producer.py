from __future__ import annotations

import json
import time
from typing import Any

from kafka import KafkaProducer

from config.settings import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC
from producer.event_generator import generate_event_stream


def create_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda key: str(key).encode("utf-8"),
    )


def send_events(session_count: int = 3, sleep_seconds: float = 0.5) -> None:
    producer = create_producer()
    events = generate_event_stream(session_count=session_count)

    print(f"Sending {len(events)} events to Kafka topic '{KAFKA_TOPIC}'...\n")

    for index, event in enumerate(events, start=1):
        future = producer.send(
            KAFKA_TOPIC,
            key=event["user_id"],
            value=event,
        )
        metadata = future.get(timeout=10)

        print(
            f"[{index}] sent event_id={event['event_id']} "
            f"user_id={event['user_id']} "
            f"event_type={event['event_type']} "
            f"topic={metadata.topic} partition={metadata.partition} offset={metadata.offset}"
        )

        time.sleep(sleep_seconds)

    producer.flush()
    producer.close()
    print("\nKafka producer finished sending events.")


if __name__ == "__main__":
    send_events()