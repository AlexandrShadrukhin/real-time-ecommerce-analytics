from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from ml.model import explain_feature_contributions
from ml.predict import predict
from ml.predict import load_model_artifact
from ml.schema import FEATURE_DESCRIPTIONS, METRICS_PATH, MODEL_VERSION, SCHEMA_PATH


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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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


def _replace_behavior_flags(payload: dict, scenario: str) -> dict:
    scenario_flags = {
        "view": {"is_view": 1, "is_click": 0, "is_cart": 0},
        "click": {"is_view": 0, "is_click": 1, "is_cart": 0},
        "add_to_cart": {"is_view": 0, "is_click": 0, "is_cart": 1},
    }
    return {**payload, **scenario_flags[scenario]}


def _scenario_comparison(base_payload: dict) -> dict[str, float]:
    outputs = {}
    for scenario in ["view", "click", "add_to_cart"]:
        prediction = predict(_replace_behavior_flags(base_payload, scenario))
        outputs[scenario] = prediction["purchase_probability"]
    return outputs


def _score_band(score: float, threshold: float) -> tuple[str, str]:
    if score >= threshold + 0.15:
        return "High intent", "User already looks close to conversion."
    if score >= threshold:
        return "Conversion zone", "This user is above the current action threshold."
    if score >= max(threshold - 0.15, 0.1):
        return "Watchlist", "Worth nudging, but signal is still mixed."
    return "Low intent", "No strong purchase signal yet."


def _recommended_action(score: float, threshold: float) -> tuple[str, str]:
    if score >= threshold + 0.15:
        return "Push conversion", "Show a strong purchase CTA, cart reminder, or limited-time incentive."
    if score >= threshold:
        return "Assist decision", "Surface recommendations, delivery info, or social proof near checkout."
    if score >= max(threshold - 0.15, 0.1):
        return "Nurture intent", "Keep the user warm with product details, comparison blocks, or soft promos."
    return "Do not force", "Collect more behavior first instead of spending an aggressive intervention."


def _feature_label(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


def _top_driver_rows(payload: dict, artifact: dict) -> tuple[list[dict], list[dict]]:
    explanation = explain_feature_contributions(payload, artifact)
    contribution_rows = []
    for row in explanation["contributions"]:
        contribution_rows.append(
            {
                "feature": _feature_label(row["feature"]),
                "value": round(row["raw_value"], 3),
                "impact": round(row["contribution"], 4),
                "description": FEATURE_DESCRIPTIONS.get(row["feature"], ""),
            }
        )

    positives = [row for row in contribution_rows if row["impact"] > 0]
    negatives = [row for row in contribution_rows if row["impact"] < 0]
    positives.sort(key=lambda row: row["impact"], reverse=True)
    negatives.sort(key=lambda row: row["impact"])
    return positives[:4], negatives[:4]


def _scenario_rows(base_payload: dict, current_score: float) -> list[dict]:
    rows = []
    for scenario, score in _scenario_comparison(base_payload).items():
        rows.append(
            {
                "scenario": scenario.replace("_", " ").title(),
                "probability": round(score, 4),
                "delta_vs_current": round(score - current_score, 4),
            }
        )
    rows.sort(key=lambda row: row["probability"], reverse=True)
    return rows


metrics = _load_json(METRICS_PATH)
schema = _load_json(SCHEMA_PATH)
artifact = load_model_artifact() if Path(METRICS_PATH).exists() and Path(SCHEMA_PATH).exists() else None

st.set_page_config(page_title="Purchase Decision Console", layout="wide", initial_sidebar_state="expanded")

st.title("Purchase Decision Console")
st.caption("Use this screen to judge whether the current user state is worth an intervention.")

with st.sidebar:
    st.header("Simulation Controls")
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

if metrics and schema and artifact:
    validation = metrics.get("validation_metrics", {})
    threshold = float(validation.get("best_threshold", 0.5))
    prediction = predict(payload)
    probability = prediction["purchase_probability"]
    band_title, band_text = _score_band(probability, threshold)
    action_title, action_text = _recommended_action(probability, threshold)
    positives, negatives = _top_driver_rows(payload, artifact)
    scenario_rows = _scenario_rows(payload, probability)

    summary_columns = st.columns(4)
    summary_columns[0].metric("Purchase Probability", f"{probability:.1%}", delta=f"{probability - threshold:+.1%} vs threshold")
    summary_columns[1].metric("Action Threshold", f"{threshold:.2f}")
    summary_columns[2].metric("Decision Band", band_title)
    summary_columns[3].metric("Model Version", prediction["model_version"])

    decision_left, decision_right = st.columns([1.05, 0.95])
    with decision_left:
        st.subheader("What Should We Do?")
        st.info(f"**{action_title}**\n\n{action_text}")
        st.caption(band_text)
        context_columns = st.columns(4)
        context_columns[0].metric("Price", f"{payload['price']:.0f}")
        context_columns[1].metric("Price vs user mean", f"{payload['price_to_user_mean']:.2f}x")
        context_columns[2].metric("Session depth", str(payload["session_event_index"]))
        context_columns[3].metric("Historical purchases", str(payload["user_total_purchases"]))

    with decision_right:
        st.subheader("Scenario Comparison")
        st.dataframe(scenario_rows, width="stretch", hide_index=True)
        best_scenario = scenario_rows[0]["scenario"] if scenario_rows else "n/a"
        st.caption(f"Best immediate behavior in the same context: **{best_scenario}**")

    drivers_left, drivers_right = st.columns(2)
    with drivers_left:
        st.subheader("What Pushes The Score Up")
        st.dataframe(positives, width="stretch", hide_index=True)
    with drivers_right:
        st.subheader("What Holds The Score Down")
        st.dataframe(negatives, width="stretch", hide_index=True)

    with st.expander("Model Health", expanded=True):
        health_columns = st.columns(4)
        health_columns[0].metric("Validation Accuracy", f"{validation.get('accuracy', 0.0):.1%}")
        health_columns[1].metric("Validation F1", f"{validation.get('f1_score', 0.0):.1%}")
        health_columns[2].metric("Validation Log Loss", f"{validation.get('log_loss', 0.0):.3f}")
        health_columns[3].metric("Positive Rate", f"{validation.get('positive_rate', 0.0):.1%}")

        model_left, model_right = st.columns([1.05, 0.95])
        with model_left:
            st.caption("Dataset summary")
            st.json(metrics.get("dataset_summary", {}))
        with model_right:
            st.caption("Top feature weights")
            st.dataframe(metrics.get("feature_weights", []), width="stretch", hide_index=True)

    with st.expander("Technical Details"):
        tech_left, tech_right = st.columns(2)
        with tech_left:
            st.caption("Payload sent to the scorer")
            st.code(json.dumps(payload, indent=2), language="json")
            st.caption("Prediction response")
            st.json(prediction)
        with tech_right:
            st.caption("Feature contract")
            st.json(schema)
else:
    st.warning("Artifacts not found yet. Run `python -m ml.train_model` to generate metrics.")
