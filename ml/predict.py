from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from catboost import CatBoostClassifier

from ml.schema import MODEL_FEATURES, MODEL_PATH, MODEL_VERSION, normalize_prediction_payload


@lru_cache(maxsize=1)
def load_model(model_path: str | Path | None = None) -> CatBoostClassifier:
    resolved_path = Path(model_path or MODEL_PATH)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model file was not found at {resolved_path}. "
            f"Expected CatBoost model.cbm."
        )

    model = CatBoostClassifier()
    model.load_model(resolved_path)
    return model


def predict_proba(payload: Mapping[str, Any], model_path: str | Path | None = None) -> float:
    normalized_payload = normalize_prediction_payload(payload)
    model = load_model(model_path)

    feature_vector = [[normalized_payload[feature_name] for feature_name in MODEL_FEATURES]]
    probability = model.predict_proba(feature_vector)[0][1]
    return float(probability)


def predict(payload: Mapping[str, Any], model_path: str | Path | None = None) -> dict[str, Any]:
    normalized_payload = normalize_prediction_payload(payload)
    score = predict_proba(normalized_payload, model_path=model_path)

    return {
        "event_id": normalized_payload["event_id"],
        "user_id": normalized_payload["user_id"],
        "session_id": normalized_payload["session_id"],
        "purchase_probability": round(score, 6),
        "model_version": MODEL_VERSION,
        "features": {feature_name: normalized_payload[feature_name] for feature_name in MODEL_FEATURES},
    }