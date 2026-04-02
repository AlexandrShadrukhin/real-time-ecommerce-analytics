from __future__ import annotations

import json
from typing import Any

from kafka import KafkaConsumer

from config.settings import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_CONSUMER_GROUP,
    KAFKA_TOPIC,
)
from ml.predict import predict
from processor.feature_builder import build_features, initialize_state
from storage.clickhouse_client import (
    FEATURE_PAYLOADS_PATH,
    PREDICTIONS_PATH,
    RAW_EVENTS_PATH,
    initialize_clickhouse,
    save_feature_payload,
    save_prediction,
    save_raw_event,
)


def create_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_CONSUMER_GROUP,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        key_deserializer=lambda key: key.decode("utf-8") if key else None,
    )


def process_one_event(event: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    feature_payload = build_features(event, state)
    prediction = predict(feature_payload)

    return {
        "event": event,
        "features": feature_payload,
        "prediction": prediction,
    }


def consume_stream() -> None:
    initialize_clickhouse()

    state = initialize_state()
    consumer = create_consumer()

    print(f"Listening to Kafka topic '{KAFKA_TOPIC}'...")
    print("Artifacts will be saved to:")
    print(f"  - {RAW_EVENTS_PATH}")
    print(f"  - {FEATURE_PAYLOADS_PATH}")
    print(f"  - {PREDICTIONS_PATH}")
    print("ClickHouse sink is enabled.\n")

    try:
        for index, message in enumerate(consumer, start=1):
            event = message.value
            result = process_one_event(event, state)

            save_raw_event(result["event"])
            save_feature_payload(result["features"])
            save_prediction(result["prediction"], event_timestamp=result["event"]["timestamp"])

            print(f"===== Kafka Message #{index} =====")
            print(
                f"topic={message.topic} partition={message.partition} offset={message.offset}"
            )

            print("Raw event:")
            print(json.dumps(result["event"], indent=2, ensure_ascii=False))

            print("Feature payload:")
            print(json.dumps(result["features"], indent=2, ensure_ascii=False))

            print("Prediction:")
            print(json.dumps(result["prediction"], indent=2, ensure_ascii=False))
            print()

    except KeyboardInterrupt:
        print("\nConsumer stopped by user.")
    finally:
        consumer.close()


if __name__ == "__main__":
    consume_stream()