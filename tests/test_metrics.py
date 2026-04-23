from pathlib import Path
import unittest
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from avocado_mld.metrics import (
    accuracy_score,
    macro_f1_score,
    mean_absolute_error,
    mean_squared_error,
    monotonic_violation_rate,
    quadratic_weighted_kappa,
    r2_score,
    rank_correlation,
)


class MetricTests(unittest.TestCase):
    def test_classification_metrics_match_expected_values(self) -> None:
        y_true = np.array([0, 1, 2, 2], dtype=int)
        y_pred = np.array([0, 2, 2, 1], dtype=int)

        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.5, places=6)
        self.assertAlmostEqual(macro_f1_score(y_true, y_pred), 0.5, places=6)
        self.assertAlmostEqual(quadratic_weighted_kappa(y_true, y_pred), 0.6363636363636364, places=6)

    def test_regression_metrics_match_expected_values(self) -> None:
        actual = np.array([1.0, 2.0, 3.0], dtype=float)
        predicted = np.array([1.0, 2.0, 4.0], dtype=float)

        self.assertAlmostEqual(mean_absolute_error(actual, predicted), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(mean_squared_error(actual, predicted), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(r2_score(actual, predicted), 0.5, places=6)

    def test_monotonic_violation_rate_counts_inversions(self) -> None:
        z = np.array([0.5, 0.3, 0.8], dtype=float)
        pairs = [(0, 1), (1, 2), (0, 2)]
        rate = monotonic_violation_rate(z, pairs)
        self.assertAlmostEqual(rate, 1 / 3, places=6)

    def test_rank_correlation_returns_one_for_perfect_order(self) -> None:
        actual = np.array([4, 3, 2, 1], dtype=float)
        predicted = np.array([4, 3, 2, 1], dtype=float)
        self.assertAlmostEqual(rank_correlation(actual, predicted), 1.0, places=6)

    def test_rank_correlation_returns_negative_one_for_reverse_order(self) -> None:
        actual = np.array([1, 2, 3, 4], dtype=float)
        predicted = np.array([4, 3, 2, 1], dtype=float)
        self.assertAlmostEqual(rank_correlation(actual, predicted), -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
