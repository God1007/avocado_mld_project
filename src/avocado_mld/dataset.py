from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AvocadoDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: str | Path = ".", image_size: int = 224):
        self.frame = frame.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        image = Image.open(image_path).convert("RGB")
        return {
            "image": self.transform(image),
            "stage_index": int(row["stage_index"]),
            "remaining_days": float(row["remaining_days"]),
            "image_path": str(row["image_path"]),
            "fruit_id": str(row["fruit_id"]),
            "storage_condition": str(row["storage_condition"]),
            "day": int(row["day"]),
            "view": str(row["view"]),
        }


def collate_batch(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "image": torch.stack([item["image"] for item in items]),
        "stage_index": torch.tensor([item["stage_index"] for item in items], dtype=torch.long),
        "remaining_days": torch.tensor([item["remaining_days"] for item in items], dtype=torch.float32),
        "image_path": [item["image_path"] for item in items],
        "fruit_id": [item["fruit_id"] for item in items],
        "storage_condition": [item["storage_condition"] for item in items],
        "day": torch.tensor([item["day"] for item in items], dtype=torch.long),
        "view": [item["view"] for item in items],
    }
