from pathlib import Path
import shutil
import unittest
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from avocado_mld.train import save_run_artifacts


class TrainOutputTests(unittest.TestCase):
    def test_save_run_artifacts_writes_metrics_history_and_predictions(self) -> None:
        output_dir = Path("tmp_train_outputs")
        metrics = {"accuracy": 0.42, "qwk": 0.51}
        history = [{"epoch": 1, "loss": 1.2, "val_qwk": 0.4}]
        predictions = pd.DataFrame(
            {
                "model": ["mld", "mld"],
                "stage_true": [0, 1],
                "stage_pred": [0, 1],
                "remaining_true": [9.0, 7.0],
                "remaining_pred": [8.5, 7.1],
                "latent": [0.12, 0.34],
            }
        )

        artifacts = save_run_artifacts(
            output_dir=output_dir,
            metrics=metrics,
            history=history,
            predictions=predictions,
            model_state=None,
        )

        self.assertTrue(Path(artifacts["metrics_path"]).exists())
        self.assertTrue(Path(artifacts["history_path"]).exists())
        self.assertTrue(Path(artifacts["predictions_path"]).exists())
        self.assertEqual(pd.read_csv(artifacts["predictions_path"]).shape[0], 2)

        shutil.rmtree(output_dir)


if __name__ == "__main__":
    unittest.main()
