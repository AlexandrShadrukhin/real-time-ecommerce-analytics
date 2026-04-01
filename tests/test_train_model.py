import json
import unittest
from pathlib import Path

from ml.train_model import train_and_save_model
from tests.support import workspace_temp_dir


class TrainModelTests(unittest.TestCase):
    def test_training_writes_model_schema_and_metrics(self):
        with workspace_temp_dir() as temp_dir:
            output_dir = Path(temp_dir)
            result = train_and_save_model(output_dir=output_dir, sample_count=700, epochs=200)

            model_path = Path(result["paths"]["model"])
            schema_path = Path(result["paths"]["schema"])
            metrics_path = Path(result["paths"]["metrics"])

            self.assertTrue(model_path.exists())
            self.assertTrue(schema_path.exists())
            self.assertTrue(metrics_path.exists())

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertGreater(metrics["validation_metrics"]["accuracy"], 0.7)
            self.assertGreater(metrics["validation_metrics"]["best_f1_score"], 0.45)
            self.assertLess(metrics["validation_metrics"]["log_loss"], 0.7)
            self.assertIn("dataset_summary", metrics)
            self.assertTrue(metrics["feature_weights"])
