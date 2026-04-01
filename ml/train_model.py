from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from ml.model import evaluate_model, train_logistic_regression
from ml.schema import ARTIFACTS_DIR, METRICS_PATH, MODEL_FEATURES, MODEL_PATH, MODEL_VERSION, SCHEMA_PATH, build_schema_document


def _sample_training_row(rng: random.Random) -> dict[str, Any]:
    event_type = rng.choices(
        population=["view", "click", "add_to_cart"],
        weights=[0.58, 0.27, 0.15],
        k=1,
    )[0]
    price = rng.randint(100, 5000)
    hour_of_day = rng.randint(0, 23)

    is_view = int(event_type == "view")
    is_click = int(event_type == "click")
    is_cart = int(event_type == "add_to_cart")

    logit = -4.2
    logit += 0.65 * is_view
    logit += 1.25 * is_click
    logit += 2.4 * is_cart
    logit += 0.00018 * price
    logit += 0.15 if 18 <= hour_of_day <= 23 else -0.05
    logit += rng.uniform(-0.35, 0.35)

    purchase_probability = 1.0 / (1.0 + math.exp(-logit))
    label = int(rng.random() < purchase_probability)

    return {
        "price": float(price),
        "is_view": is_view,
        "is_click": is_click,
        "is_cart": is_cart,
        "hour_of_day": hour_of_day,
        "label": label,
    }


def build_synthetic_dataset(sample_count: int = 5000, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [_sample_training_row(rng) for _ in range(sample_count)]


def _split_train_validation(
    rows: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)
    split_index = int(len(shuffled_rows) * (1 - validation_ratio))
    return shuffled_rows[:split_index], shuffled_rows[split_index:]


def train_and_save_model(
    *,
    output_dir: Path | None = None,
    sample_count: int = 5000,
    validation_ratio: float = 0.2,
    seed: int = 42,
    epochs: int = 500,
    learning_rate: float = 0.08,
) -> dict[str, Any]:
    artifacts_dir = Path(output_dir or ARTIFACTS_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_synthetic_dataset(sample_count=sample_count, seed=seed)
    train_rows, validation_rows = _split_train_validation(dataset, validation_ratio, seed)

    artifact = train_logistic_regression(
        train_rows,
        MODEL_FEATURES,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    artifact["model_version"] = MODEL_VERSION

    validation_metrics = evaluate_model(validation_rows, artifact)
    training_metrics = evaluate_model(train_rows, artifact)
    metrics_payload = {
        "model_version": MODEL_VERSION,
        "sample_count": sample_count,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
    }

    model_path = artifacts_dir / MODEL_PATH.name
    schema_path = artifacts_dir / SCHEMA_PATH.name
    metrics_path = artifacts_dir / METRICS_PATH.name

    model_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    schema_path.write_text(json.dumps(build_schema_document(), indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return {
        "artifact": artifact,
        "metrics": metrics_payload,
        "paths": {
            "model": str(model_path),
            "schema": str(schema_path),
            "metrics": str(metrics_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline purchase model.")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    args = parser.parse_args()

    result = train_and_save_model(
        sample_count=args.samples,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
