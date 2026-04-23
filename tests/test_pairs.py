from pathlib import Path
import unittest
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from avocado_mld.pairs import build_temporal_pairs, build_view_pairs


class PairBuilderTests(unittest.TestCase):
    def test_build_temporal_pairs_orders_same_fruit_by_day(self) -> None:
        frame = pd.DataFrame(
            {
                "fruit_id": ["F1", "F1", "F1", "F2"],
                "day": [1, 3, 2, 1],
                "view": ["front", "front", "front", "front"],
            }
        )

        pairs = build_temporal_pairs(frame)

        self.assertEqual(pairs, [(0, 2), (0, 1), (2, 1)])

    def test_build_view_pairs_matches_same_fruit_and_day(self) -> None:
        frame = pd.DataFrame(
            {
                "fruit_id": ["F1", "F1", "F1", "F2"],
                "day": [1, 1, 2, 1],
                "view": ["front", "back", "front", "front"],
            }
        )

        pairs = build_view_pairs(frame)

        self.assertEqual(pairs, [(0, 1)])


if __name__ == "__main__":
    unittest.main()
