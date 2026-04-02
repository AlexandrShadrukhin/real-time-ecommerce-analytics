from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import clickhouse_connect

from config.settings import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USERNAME,
)

REPORTS_DIR = Path("artifacts/reports")
RAW_EVENTS_PATH = REPORTS_DIR / "raw_events.jsonl"
FEATURE_PAYLOADS_PATH = REPORTS_DIR / "feature_payloads.jsonl"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.jsonl"
SCHEMA_SQL_PATH = Path("storage/clickhouse_schema.sql")


def ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_reports_dir()
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_clickhouse_client():
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USERNAME,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
    )


def initialize_clickhouse() -> None:
    if not SCHEMA_SQL_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_SQL_PATH}")

    client = get_clickhouse_client()
    try:
        sql_text = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
        statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]
        for statement in statements:
            client.command(statement)
    finally:
        client.close()


def save_raw_event(event: dict[str, Any]) -> None:
    append_jsonl(RAW_EVENTS_PATH, event)

    client = get_clickhouse_client()
    try:
        client.insert(
            "raw_events",
            [[
                str(event["event_id"]),
                int(event["user_id"]),
                str(event["session_id"]),
                int(event["product_id"]),
                str(event["event_type"]),
                str(event["timestamp"]),
                float(event["price"]),
                str(event.get("category", "")),
                str(event.get("device_type", "")),
                str(event.get("source", "")),
            ]],
            column_names=[
                "event_id",
                "user_id",
                "session_id",
                "product_id",
                "event_type",
                "timestamp",
                "price",
                "category",
                "device_type",
                "source",
            ],
        )
    finally:
        client.close()


def save_feature_payload(payload: dict[str, Any]) -> None:
    append_jsonl(FEATURE_PAYLOADS_PATH, payload)

    client = get_clickhouse_client()
    try:
        client.insert(
            "feature_payloads",
            [[
                str(payload["event_id"]),
                int(payload["user_id"]),
                str(payload["timestamp"]),
                float(payload["price"]),
                float(payload["price_to_user_mean"]),
                int(payload["hour_of_day"]),
                int(payload["is_view"]),
                int(payload["is_click"]),
                int(payload["is_cart"]),
                int(payload["session_event_index"]),
                int(payload["session_click_count"]),
                int(payload["session_cart_count"]),
                int(payload["user_total_clicks"]),
                int(payload["user_total_carts"]),
                int(payload["user_total_purchases"]),
            ]],
            column_names=[
                "event_id",
                "user_id",
                "timestamp",
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
            ],
        )
    finally:
        client.close()


def save_prediction(prediction: dict[str, Any], event_timestamp: str) -> None:
    append_jsonl(PREDICTIONS_PATH, prediction)

    client = get_clickhouse_client()
    try:
        client.insert(
            "predictions",
            [[
                str(prediction["event_id"]),
                int(prediction["user_id"]),
                float(prediction["purchase_probability"]),
                str(prediction["model_version"]),
                str(event_timestamp),
            ]],
            column_names=[
                "event_id",
                "user_id",
                "purchase_probability",
                "model_version",
                "timestamp",
            ],
        )
    finally:
        client.close()