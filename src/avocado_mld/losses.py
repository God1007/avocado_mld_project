from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_monotonicity_loss(
    z: torch.Tensor,
    pairs: list[tuple[int, int]],
    margin: float,
) -> torch.Tensor:
    if not pairs:
        return z.new_tensor(0.0)
    penalties = [F.relu(margin - (z[right] - z[left])) for left, right in pairs]
    return torch.stack(penalties).mean()


def view_consistency_loss(z: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return z.new_tensor(0.0)
    penalties = [torch.abs(z[left] - z[right]) for left, right in pairs]
    return torch.stack(penalties).mean()
