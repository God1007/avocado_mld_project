from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TrainImportTests(unittest.TestCase):
    def test_train_module_imports(self) -> None:
        from avocado_mld import train

        self.assertTrue(callable(train.evaluate_mld))
        self.assertTrue(callable(train.train_mld_model))


if __name__ == "__main__":
    unittest.main()
