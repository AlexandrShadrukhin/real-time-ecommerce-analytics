from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from ml.predict import predict
from ml.schema import METRICS_PATH, MODEL_VERSION, SCHEMA_PATH


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_payload(
    *,
    price: float,
    user_average_price: float,
    scenario: str,
    hour_of_day: int,
    session_event_index: int,
    session_click_count: int,
    session_cart_count: int,
    user_total_clicks: int,
    user_total_carts: int,
    user_total_purchases: int,
) -> dict:
    scenario_flags = {
        "view": {"is_view": 1, "is_click": 0, "is_cart": 0},
        "click": {"is_view": 0, "is_click": 1, "is_cart": 0},
        "add_to_cart": {"is_view": 0, "is_click": 0, "is_cart": 1},
    }
    return {
        "event_id": f"dashboard-{scenario}-{hour_of_day}",
        "user_id": 1,
        "timestamp": datetime.utcnow().isoformat(),
        "price": float(price),
        "price_to_user_mean": round(float(price) / max(float(user_average_price), 1.0), 4),
        "hour_of_day": int(hour_of_day),
        "session_event_index": int(session_event_index),
        "session_click_count": int(session_click_count),
        "session_cart_count": int(session_cart_count),
        "user_total_clicks": int(user_total_clicks),
        "user_total_carts": int(user_total_carts),
        "user_total_purchases": int(user_total_purchases),
        **scenario_flags[scenario],
    }


def _scenario_comparison(base_payload: dict) -> dict[str, float]:
    outputs = {}
    scenario_map = {
        "view": {"is_view": 1, "is_click": 0, "is_cart": 0},
        "click": {"is_view": 0, "is_click": 1, "is_cart": 0},
        "add_to_cart": {"is_view": 0, "is_click": 0, "is_cart": 1},
    }
    for scenario, flags in scenario_map.items():
        prediction = predict({**base_payload, **flags})
        outputs[scenario] = prediction["purchase_probability"]
    return outputs


st.set_page_config(page_title="ML Purchase Monitor", layout="wide")

metrics = _load_json(METRICS_PATH)
schema = _load_json(SCHEMA_PATH)

st.title("Purchase Probability Dashboard")
st.caption("ML baseline artifacts and interactive scoring for the MVP model.")

left_column, right_column = st.columns([1.15, 1.0])

with left_column:
    st.subheader("Baseline Metrics")
    if metrics:
        validation = metrics.get("validation_metrics", {})
        training = metrics.get("training_metrics", {})
        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Validation Accuracy",
            f"{validation.get('accuracy', 0.0):.1%}",
        )
        metric_columns[1].metric(
            "Best Validation F1",
            f"{validation.get('best_f1_score', validation.get('f1_score', 0.0)):.1%}",
        )
        metric_columns[2].metric(
            "Validation Log Loss",
            f"{validation.get('log_loss', 0.0):.3f}",
        )
        metric_columns[3].metric(
            "Best Threshold",
            f"{validation.get('best_threshold', 0.5):.2f}",
        )

        st.write(
            {
                "model_version": metrics.get("model_version", MODEL_VERSION),
                "sample_count": metrics.get("sample_count", 0),
                "train_rows": metrics.get("train_rows", 0),
                "validation_rows": metrics.get("validation_rows", 0),
                "dataset_summary": metrics.get("dataset_summary", {}),
                "training_metrics": training,
                "validation_metrics": validation,
            }
        )

        st.caption("Top feature weights")
        st.dataframe(metrics.get("feature_weights", []), use_container_width=True)
    else:
        st.warning("Artifacts not found yet. Run `python -m ml.train_model` to generate metrics.")

with right_column:
    st.subheader("Feature Contract")
    if schema:
        st.json(schema)
    else:
        st.info("Feature schema artifact is not available yet.")

st.divider()
st.subheader("Interactive Scoring")

controls, preview = st.columns([1.0, 1.15])

with controls:
    scenario = st.radio(
        "Behavior snapshot",
        options=["view", "click", "add_to_cart"],
        format_func=lambda value: value.replace("_", " ").title(),
        horizontal=True,
    )
    price = st.slider("Price", min_value=100, max_value=5000, value=1200, step=50)
    user_average_price = st.slider(
        "User average price",
        min_value=200,
        max_value=5000,
        value=1600,
        step=50,
    )
    hour_of_day = st.slider("Hour of day", min_value=0, max_value=23, value=18)
    session_event_index = st.slider("Session depth", min_value=1, max_value=10, value=3)
    session_click_count = st.slider("Session clicks so far", min_value=0, max_value=10, value=1)
    session_cart_count = st.slider("Session carts so far", min_value=0, max_value=6, value=0)
    user_total_clicks = st.slider("User historical clicks", min_value=0, max_value=40, value=5)
    user_total_carts = st.slider("User historical carts", min_value=0, max_value=20, value=1)
    user_total_purchases = st.slider("User historical purchases", min_value=0, max_value=10, value=0)
    payload = _build_payload(
        price=price,
        user_average_price=user_average_price,
        scenario=scenario,
        hour_of_day=hour_of_day,
        session_event_index=session_event_index,
        session_click_count=session_click_count,
        session_cart_count=session_cart_count,
        user_total_clicks=user_total_clicks,
        user_total_carts=user_total_carts,
        user_total_purchases=user_total_purchases,
    )
    try:
        prediction = predict(payload)
        probability = prediction["purchase_probability"]
        st.metric("Purchase Probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Price to user mean: {payload['price_to_user_mean']:.2f}")
    except FileNotFoundError:
        prediction = {"detail": "Train the model first with `python -m ml.train_model`."}
        st.warning(prediction["detail"])

with preview:
    if "purchase_probability" in prediction:
        st.caption("Scenario comparison with the same context")
        st.json(_scenario_comparison(payload))
    st.caption("Payload sent to the scorer")
    st.code(json.dumps(payload, indent=2), language="json")
    st.caption("Prediction response")
    st.json(prediction)
