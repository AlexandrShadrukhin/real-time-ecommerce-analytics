from __future__ import annotations

import math
from typing import Any, Iterable


def sigmoid(value: float) -> float:
    bounded = max(min(value, 35.0), -35.0)
    return 1.0 / (1.0 + math.exp(-bounded))


def _split_dataset(
    rows: Iterable[dict[str, Any]],
    feature_names: list[str],
) -> tuple[list[list[float]], list[int]]:
    matrix: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        matrix.append([float(row[name]) for name in feature_names])
        labels.append(int(row["label"]))
    return matrix, labels


def _fit_scaler(matrix: list[list[float]]) -> dict[str, list[float]]:
    if not matrix:
        raise ValueError("Training matrix is empty.")

    column_count = len(matrix[0])
    means = []
    scales = []

    for column_index in range(column_count):
        column = [row[column_index] for row in matrix]
        mean_value = sum(column) / len(column)
        variance = sum((value - mean_value) ** 2 for value in column) / len(column)
        means.append(mean_value)
        scales.append(math.sqrt(variance) or 1.0)

    return {"means": means, "scales": scales}


def _apply_scaler(
    matrix: list[list[float]],
    scaler: dict[str, list[float]],
) -> list[list[float]]:
    transformed = []
    for row in matrix:
        transformed.append(
            [
                (value - scaler["means"][index]) / scaler["scales"][index]
                for index, value in enumerate(row)
            ]
        )
    return transformed


def train_logistic_regression(
    rows: Iterable[dict[str, Any]],
    feature_names: list[str],
    *,
    epochs: int = 500,
    learning_rate: float = 0.08,
) -> dict[str, Any]:
    matrix, labels = _split_dataset(rows, feature_names)
    scaler = _fit_scaler(matrix)
    normalized_matrix = _apply_scaler(matrix, scaler)

    weights = [0.0 for _ in feature_names]
    bias = 0.0
    row_count = len(normalized_matrix)

    for _ in range(epochs):
        gradient = [0.0 for _ in feature_names]
        bias_gradient = 0.0

        for row, label in zip(normalized_matrix, labels):
            prediction = sigmoid(sum(weight * value for weight, value in zip(weights, row)) + bias)
            error = prediction - label
            for index, value in enumerate(row):
                gradient[index] += error * value
            bias_gradient += error

        for index in range(len(weights)):
            weights[index] -= learning_rate * (gradient[index] / row_count)
        bias -= learning_rate * (bias_gradient / row_count)

    return {
        "weights": weights,
        "bias": bias,
        "feature_names": feature_names,
        "scaler": scaler,
        "training_rows": row_count,
    }


def score_features(features: dict[str, float], artifact: dict[str, Any]) -> float:
    normalized_row = []
    for index, feature_name in enumerate(artifact["feature_names"]):
        value = float(features[feature_name])
        mean_value = artifact["scaler"]["means"][index]
        scale_value = artifact["scaler"]["scales"][index]
        normalized_row.append((value - mean_value) / scale_value)

    linear_score = artifact["bias"]
    linear_score += sum(
        weight * value for weight, value in zip(artifact["weights"], normalized_row)
    )
    return sigmoid(linear_score)


def evaluate_model(rows: Iterable[dict[str, Any]], artifact: dict[str, Any]) -> dict[str, float]:
    scored_rows = list(rows)
    if not scored_rows:
        raise ValueError("Evaluation dataset is empty.")

    predictions = [score_features(row, artifact) for row in scored_rows]
    labels = [int(row["label"]) for row in scored_rows]

    accuracy = sum(
        int((prediction >= 0.5) == bool(label))
        for prediction, label in zip(predictions, labels)
    ) / len(labels)

    log_loss = 0.0
    for prediction, label in zip(predictions, labels):
        bounded = min(max(prediction, 1e-9), 1 - 1e-9)
        log_loss += -(label * math.log(bounded) + (1 - label) * math.log(1 - bounded))

    average_score = sum(predictions) / len(predictions)
    positive_rate = sum(labels) / len(labels)

    return {
        "accuracy": accuracy,
        "log_loss": log_loss / len(labels),
        "average_score": average_score,
        "positive_rate": positive_rate,
    }
