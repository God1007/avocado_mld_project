from pathlib import Path
import unittest
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from avocado_mld.models import MLDModel


class ModelSmokeTests(unittest.TestCase):
    def test_mld_model_returns_scalar_latent_and_logits(self) -> None:
        model = MLDModel(num_stages=5)
        batch = torch.randn(2, 3, 224, 224)
        outputs = model(batch)
        self.assertEqual(tuple(outputs["latent"].shape), (2,))
        self.assertEqual(tuple(outputs["ordinal_logits"].shape), (2, 4))
        self.assertEqual(tuple(outputs["remaining_days"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
