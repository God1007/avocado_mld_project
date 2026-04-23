from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "image_path",
    "fruit_id",
    "storage_condition",
    "day",
    "stage",
    "view",
]

XML_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def normalize_official_record(record: dict[str, str], image_dir: str | Path) -> dict[str, object]:
    stem = str(record["File Name"]).strip()
    sample = int(record["Sample"])
    storage_group = str(record["Storage Group"]).strip()
    day = int(record["Day of Experiment"])
    stage = int(record["Ripening Index Classification"])
    view = stem.split("_")[-2]
    image_path = Path(image_dir) / f"{stem}.jpg"
    return {
        "image_path": image_path.as_posix(),
        "fruit_id": f"{storage_group}_{sample:03d}",
        "storage_condition": storage_group,
        "day": day,
        "stage": stage,
        "view": view,
        "timestamp": str(record.get("Time Stamp", "")).strip(),
        "file_stem": stem,
    }


def _load_shared_strings(zf: ZipFile) -> list[str]:
    shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in shared_root.findall("a:si", XML_NS):
        texts = [node.text or "" for node in item.iterfind(".//a:t", XML_NS)]
        values.append("".join(texts))
    return values


def _iter_sheet_rows(zf: ZipFile, sheet_path: str) -> list[list[str]]:
    shared = _load_shared_strings(zf)
    sheet_root = ET.fromstring(zf.read(sheet_path))
    rows: list[list[str]] = []
    for row in sheet_root.find("a:sheetData", XML_NS):
        values: list[str] = []
        for cell in row.findall("a:c", XML_NS):
            cell_type = cell.attrib.get("t")
            value_node = cell.find("a:v", XML_NS)
            value = value_node.text if value_node is not None else ""
            if cell_type == "s" and value != "":
                value = shared[int(value)]
            values.append(value)
        rows.append(values)
    return rows


def build_metadata_from_official_xlsx(
    xlsx_path: str | Path,
    image_dir: str | Path,
) -> pd.DataFrame:
    with ZipFile(xlsx_path) as zf:
        rows = _iter_sheet_rows(zf, "xl/worksheets/sheet1.xml")
    header = rows[0]
    records = [dict(zip(header, row)) for row in rows[1:] if any(row)]
    normalized = [normalize_official_record(record, image_dir=image_dir) for record in records]
    frame = pd.DataFrame(normalized)
    return frame.sort_values(["fruit_id", "day", "view"]).reset_index(drop=True)


def filter_existing_images(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame["image_path"].map(lambda value: Path(value).exists())
    return frame.loc[mask].reset_index(drop=True)


def export_official_metadata_csv(
    xlsx_path: str | Path,
    image_dir: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    frame = build_metadata_from_official_xlsx(xlsx_path, image_dir=image_dir)
    frame = filter_existing_images(frame)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def build_metadata_frame(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path).copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame["day"] = frame["day"].astype(int)
    frame["stage"] = frame["stage"].astype(int)
    frame["stage_index"] = frame["stage"] - 1
    max_day = frame.groupby("fruit_id")["day"].transform("max")
    frame["remaining_days"] = max_day - frame["day"]
    return frame.sort_values(["fruit_id", "day", "view"]).reset_index(drop=True)


def make_instance_split(
    frame: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fruit_ids = sorted(frame["fruit_id"].unique().tolist())
    rng = np.random.default_rng(random_state)
    shuffled = fruit_ids.copy()
    rng.shuffle(shuffled)

    def split_ids(values: list[str], ratio: float) -> tuple[list[str], list[str]]:
        if len(values) <= 1:
            return values, []
        holdout = int(round(len(values) * ratio))
        holdout = max(1, min(holdout, len(values) - 1))
        return values[holdout:], values[:holdout]

    train_ids, test_ids = split_ids(shuffled, test_size)
    train_ids, val_ids = split_ids(train_ids, val_size)
    train_df = frame[frame["fruit_id"].isin(train_ids)].reset_index(drop=True)
    val_df = frame[frame["fruit_id"].isin(val_ids)].reset_index(drop=True)
    test_df = frame[frame["fruit_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df
