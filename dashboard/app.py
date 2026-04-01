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


def _build_payload(price: float, scenario: str, hour_of_day: int) -> dict:
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
        "hour_of_day": int(hour_of_day),
        **scenario_flags[scenario],
    }


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
        metric_columns = st.columns(3)
        metric_columns[0].metric(
            "Validation Accuracy",
            f"{validation.get('accuracy', 0.0):.1%}",
        )
        metric_columns[1].metric(
            "Validation Log Loss",
            f"{validation.get('log_loss', 0.0):.3f}",
        )
        metric_columns[2].metric(
            "Positive Rate",
            f"{validation.get('positive_rate', 0.0):.1%}",
        )

        st.write(
            {
                "model_version": metrics.get("model_version", MODEL_VERSION),
                "sample_count": metrics.get("sample_count", 0),
                "train_rows": metrics.get("train_rows", 0),
                "validation_rows": metrics.get("validation_rows", 0),
                "training_metrics": training,
                "validation_metrics": validation,
            }
        )
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
    hour_of_day = st.slider("Hour of day", min_value=0, max_value=23, value=18)
    payload = _build_payload(price=price, scenario=scenario, hour_of_day=hour_of_day)
    try:
        prediction = predict(payload)
        probability = prediction["purchase_probability"]
        st.metric("Purchase Probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
    except FileNotFoundError:
        prediction = {"detail": "Train the model first with `python -m ml.train_model`."}
        st.warning(prediction["detail"])

with preview:
    st.caption("Payload sent to the scorer")
    st.code(json.dumps(payload, indent=2), language="json")
    st.caption("Prediction response")
    st.json(prediction)
