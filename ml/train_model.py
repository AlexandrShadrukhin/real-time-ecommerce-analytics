from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.model import evaluate_model, train_logistic_regression
from ml.prepare_dataset import build_synthetic_dataset, summarize_dataset
from ml.schema import ARTIFACTS_DIR, METRICS_PATH, MODEL_FEATURES, MODEL_PATH, MODEL_VERSION, SCHEMA_PATH, build_schema_document


def _split_train_validation(
    rows: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled_rows = list(rows)
    import random

    random.Random(seed).shuffle(shuffled_rows)
    split_index = int(len(shuffled_rows) * (1 - validation_ratio))
    return shuffled_rows[:split_index], shuffled_rows[split_index:]


def _feature_weights(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "feature": feature_name,
                "weight": round(weight, 6),
                "abs_weight": round(abs(weight), 6),
            }
            for feature_name, weight in zip(artifact["feature_names"], artifact["weights"])
        ],
        key=lambda item: item["abs_weight"],
        reverse=True,
    )


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
    dataset_summary = summarize_dataset(dataset)
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
        "dataset_summary": dataset_summary,
        "feature_weights": _feature_weights(artifact),
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
