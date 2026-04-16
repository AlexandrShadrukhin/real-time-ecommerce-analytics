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
                event.get("category_code"),
                event.get("brand"),
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
                "category_code",
                "brand",
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
                str(payload["session_id"]),
                str(payload["timestamp"]),
                float(payload["n_views"]),
                float(payload["n_carts"]),
                float(payload["uniq_products"]),
                float(payload["uniq_categories"]),
                float(payload["mean_price"]),
                float(payload["max_price"]),
                float(payload["price_std"]),
                float(payload["uniq_brands"]),
                float(payload["missing_brand"]),
                float(payload["missing_category_code"]),
                float(payload["prefix_duration_sec"]),
                float(payload["has_cart_in_prefix"]),
                float(payload["cart_view_ratio"]),
                float(payload["same_product_repeat_count"]),
                float(payload["same_category_repeat_count"]),
                float(payload["event_1_is_cart"]),
                float(payload["event_2_is_cart"]),
                float(payload["event_3_is_cart"]),
                float(payload["time_1_to_2_sec"]),
                float(payload["time_2_to_3_sec"]),
            ]],
            column_names=[
                "event_id",
                "user_id",
                "session_id",
                "timestamp",
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
                str(prediction["session_id"]),
                float(prediction["purchase_probability"]),
                str(prediction["model_version"]),
                str(event_timestamp),
            ]],
            column_names=[
                "event_id",
                "user_id",
                "session_id",
                "purchase_probability",
                "model_version",
                "timestamp",
            ],
        )
    finally:
        client.close()