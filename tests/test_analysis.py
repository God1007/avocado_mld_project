from pathlib import Path
import unittest
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from avocado_mld.analysis import (
    build_results_frame,
    plot_metric_comparison,
    plot_stage_confusion,
    summarize_group_metrics,
    summarize_results_frame,
)


class AnalysisTests(unittest.TestCase):
    def test_build_results_frame_flattens_nested_result_dict(self) -> None:
        frame = build_results_frame(
            {
                "stage_classifier": {"accuracy": 0.31, "qwk": 0.12, "mae": 7.4, "spearman": 0.28},
                "mld": {"accuracy": 0.43, "qwk": 0.52, "mae": 5.8, "spearman": 0.71},
            }
        )

        self.assertEqual(frame["model"].tolist(), ["stage_classifier", "mld"])
        self.assertEqual(
            list(frame.columns),
            ["model", "accuracy", "qwk", "mae", "spearman"],
        )

    def test_summarize_results_frame_reports_best_qwk(self) -> None:
        frame = pd.DataFrame(
            {
                "model": ["baseline", "mld"],
                "qwk": [0.70, 0.82],
                "mae": [2.4, 1.8],
            }
        )

        summary = summarize_results_frame(frame)

        self.assertEqual(summary["best_model"], "mld")
        self.assertAlmostEqual(summary["best_qwk"], 0.82, places=6)

    def test_summarize_group_metrics_builds_rows_per_storage_condition(self) -> None:
        predictions = pd.DataFrame(
            {
                "model": ["mld", "mld", "mld", "mld"],
                "storage_condition": ["ambient", "ambient", "cold", "cold"],
                "stage_true": [0, 1, 0, 1],
                "stage_pred": [0, 1, 0, 0],
                "remaining_true": [9.0, 7.0, 10.0, 8.0],
                "remaining_pred": [8.0, 7.0, 10.0, 7.0],
                "latent": [0.1, 0.3, 0.05, 0.25],
                "day": [1, 3, 1, 3],
                "fruit_id": ["A1", "A1", "C1", "C1"],
            }
        )

        summary = summarize_group_metrics(predictions, group_col="storage_condition")

        self.assertEqual(summary["storage_condition"].tolist(), ["ambient", "cold"])
        self.assertEqual(summary["sample_count"].tolist(), [2, 2])
        self.assertIn("accuracy", summary.columns)
        self.assertIn("mae", summary.columns)

    def test_plot_helpers_write_output_files(self) -> None:
        tmp_dir = Path("tmp_analysis_outputs")
        tmp_dir.mkdir(exist_ok=True)
        results_frame = pd.DataFrame(
            {
                "model": ["stage_classifier", "mld"],
                "accuracy": [0.31, 0.43],
                "qwk": [0.12, 0.52],
                "mae": [7.4, 5.8],
                "spearman": [0.28, 0.71],
            }
        )
        predictions = pd.DataFrame(
            {
                "stage_true": [0, 1, 2, 1],
                "stage_pred": [0, 1, 1, 2],
            }
        )

        metric_path = Path(plot_metric_comparison(results_frame, tmp_dir / "metric_comparison.png"))
        confusion_path = Path(plot_stage_confusion(predictions, tmp_dir / "stage_confusion.png"))

        self.assertTrue(metric_path.exists())
        self.assertTrue(confusion_path.exists())

        for path in [metric_path, confusion_path]:
            path.unlink()
        tmp_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
