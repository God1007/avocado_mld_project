from pathlib import Path
import unittest
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from avocado_mld.config import TrainingConfig
from avocado_mld.metadata import (
    build_metadata_frame,
    filter_existing_images,
    make_instance_split,
    normalize_official_record,
)


class TrainingConfigTests(unittest.TestCase):
    def test_training_config_defaults(self) -> None:
        config = TrainingConfig()
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.batch_size, 16)


class MetadataTests(unittest.TestCase):
    def test_normalize_official_record_builds_project_metadata_row(self) -> None:
        record = {
            "File Name": "T20_d05_001_b_3",
            "Time Stamp": "2022-04-08 15:58:29",
            "Storage Group": "T20",
            "Sample": "1",
            "Day of Experiment": "5",
            "Ripening Index Classification": "3",
        }

        row = normalize_official_record(record, image_dir="data/Avocado Ripening Dataset")

        self.assertEqual(row["image_path"], "data/Avocado Ripening Dataset/T20_d05_001_b_3.jpg")
        self.assertEqual(row["fruit_id"], "T20_001")
        self.assertEqual(row["storage_condition"], "T20")
        self.assertEqual(row["day"], 5)
        self.assertEqual(row["stage"], 3)
        self.assertEqual(row["view"], "b")

    def test_build_metadata_frame_derives_remaining_days(self) -> None:
        tmp_dir = Path("tmp_test_metadata")
        tmp_dir.mkdir(exist_ok=True)
        csv_path = tmp_dir / "metadata.csv"
        frame = pd.DataFrame(
            {
                "image_path": ["a.jpg", "b.jpg", "c.jpg"],
                "fruit_id": ["F1", "F1", "F1"],
                "storage_condition": ["cold", "cold", "cold"],
                "day": [1, 2, 4],
                "stage": [1, 2, 4],
                "view": ["front", "front", "front"],
            }
        )
        frame.to_csv(csv_path, index=False)

        result = build_metadata_frame(csv_path)

        self.assertEqual(result["remaining_days"].tolist(), [3, 2, 0])
        self.assertEqual(result["stage_index"].tolist(), [0, 1, 3])
        csv_path.unlink()
        tmp_dir.rmdir()

    def test_filter_existing_images_removes_missing_paths(self) -> None:
        tmp_dir = Path("tmp_test_images")
        tmp_dir.mkdir(exist_ok=True)
        existing_path = tmp_dir / "exists.jpg"
        existing_path.write_bytes(b"ok")
        frame = pd.DataFrame(
            {
                "image_path": [existing_path.as_posix(), (tmp_dir / "missing.jpg").as_posix()],
                "fruit_id": ["F1", "F1"],
                "storage_condition": ["cold", "cold"],
                "day": [1, 2],
                "stage": [1, 2],
                "view": ["a", "b"],
            }
        )

        filtered = filter_existing_images(frame)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["image_path"], existing_path.as_posix())
        existing_path.unlink()
        tmp_dir.rmdir()

    def test_make_instance_split_keeps_fruit_ids_disjoint(self) -> None:
        frame = pd.DataFrame(
            {
                "image_path": [f"img_{i}.jpg" for i in range(8)],
                "fruit_id": ["F1", "F1", "F2", "F2", "F3", "F3", "F4", "F4"],
                "storage_condition": ["cold"] * 8,
                "day": [1, 2, 1, 2, 1, 2, 1, 2],
                "stage": [1, 2, 1, 2, 1, 2, 1, 2],
                "view": ["front"] * 8,
                "remaining_days": [1] * 8,
                "stage_index": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        train_df, val_df, test_df = make_instance_split(frame, random_state=7)

        train_ids = set(train_df["fruit_id"])
        val_ids = set(val_df["fruit_id"])
        test_ids = set(test_df["fruit_id"])

        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))


if __name__ == "__main__":
    unittest.main()
