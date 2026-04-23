from __future__ import annotations

import pandas as pd


def build_temporal_pairs(frame: pd.DataFrame) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for _, group in frame.groupby("fruit_id", sort=False):
        ordered = group.sort_values("day")
        indices = ordered.index.tolist()
        days = ordered["day"].tolist()
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                if days[left] < days[right]:
                    pairs.append((indices[left], indices[right]))
    return pairs


def build_view_pairs(frame: pd.DataFrame) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for _, group in frame.groupby(["fruit_id", "day"], sort=False):
        indices = group.index.tolist()
        if len(indices) < 2:
            continue
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                pairs.append((indices[left], indices[right]))
    return pairs
