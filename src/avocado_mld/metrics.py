from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.union1d(y_true, y_pred)
    if labels.size == 0:
        return 0.0
    scores: list[float] = []
    for label in labels:
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        false_negative = np.sum((y_true == label) & (y_pred != label))
        denominator = (2 * true_positive) + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return float(np.mean(scores))


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return 0.0
    min_rating = int(min(y_true.min(), y_pred.min()))
    max_rating = int(max(y_true.max(), y_pred.max()))
    num_ratings = max_rating - min_rating + 1
    if num_ratings <= 1:
        return 1.0

    shifted_true = y_true - min_rating
    shifted_pred = y_pred - min_rating
    confusion = np.zeros((num_ratings, num_ratings), dtype=float)
    for actual, predicted in zip(shifted_true, shifted_pred):
        confusion[actual, predicted] += 1.0

    hist_true = np.bincount(shifted_true, minlength=num_ratings).astype(float)
    hist_pred = np.bincount(shifted_pred, minlength=num_ratings).astype(float)
    expected = np.outer(hist_true, hist_pred) / y_true.size

    grid = np.arange(num_ratings, dtype=float)
    weights = ((grid[:, None] - grid[None, :]) ** 2) / float((num_ratings - 1) ** 2)

    denominator = float(np.sum(weights * expected))
    if denominator == 0.0:
        return 1.0
    numerator = float(np.sum(weights * confusion))
    return 1.0 - (numerator / denominator)


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0:
        return 0.0
    return float(np.mean(np.abs(actual - predicted)))


def mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0:
        return 0.0
    return float(np.mean((actual - predicted) ** 2))


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0:
        return 0.0
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return 1.0 - (residual / total)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = ((start + end - 1) / 2.0) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def monotonic_violation_rate(z: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    violations = sum(1 for left, right in pairs if z[right] < z[left])
    return violations / len(pairs)


def rank_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0:
        return 0.0
    actual_ranks = _average_ranks(actual)
    predicted_ranks = _average_ranks(predicted)
    actual_centered = actual_ranks - actual_ranks.mean()
    predicted_centered = predicted_ranks - predicted_ranks.mean()
    denominator = float(
        np.sqrt(np.sum(actual_centered**2) * np.sum(predicted_centered**2))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.sum(actual_centered * predicted_centered) / denominator)
