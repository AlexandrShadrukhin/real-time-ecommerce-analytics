from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from ml.model import score_features
from ml.schema import MODEL_FEATURES, MODEL_PATH, MODEL_VERSION, normalize_prediction_payload


@lru_cache(maxsize=1)
def load_model_artifact(model_path: str | Path | None = None) -> dict[str, Any]:
    artifact_path = Path(model_path or MODEL_PATH)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact was not found at {artifact_path}. Run `python -m ml.train_model` first."
        )
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def predict_proba(payload: Mapping[str, Any], model_path: str | Path | None = None) -> float:
    normalized_payload = normalize_prediction_payload(payload)
    artifact = load_model_artifact(model_path)
    model_features = {feature_name: normalized_payload[feature_name] for feature_name in MODEL_FEATURES}
    return score_features(model_features, artifact)


def predict(payload: Mapping[str, Any], model_path: str | Path | None = None) -> dict[str, Any]:
    normalized_payload = normalize_prediction_payload(payload)
    score = predict_proba(normalized_payload, model_path=model_path)
    return {
        "event_id": normalized_payload["event_id"],
        "user_id": normalized_payload["user_id"],
        "purchase_probability": round(score, 6),
        "model_version": MODEL_VERSION,
        "features": {feature_name: normalized_payload[feature_name] for feature_name in MODEL_FEATURES},
    }
