from pathlib import Path
import unittest
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from avocado_mld.losses import temporal_monotonicity_loss, view_consistency_loss


class LossTests(unittest.TestCase):
    def test_temporal_monotonicity_loss_is_zero_for_ordered_latents(self) -> None:
        z = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float32)
        pairs = [(0, 1), (1, 2)]
        loss = temporal_monotonicity_loss(z, pairs, margin=0.05)
        self.assertTrue(torch.isclose(loss, torch.tensor(0.0)))

    def test_view_consistency_loss_penalizes_mismatch(self) -> None:
        z = torch.tensor([0.2, 0.5], dtype=torch.float32)
        pairs = [(0, 1)]
        loss = view_consistency_loss(z, pairs)
        self.assertAlmostEqual(loss.item(), 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
