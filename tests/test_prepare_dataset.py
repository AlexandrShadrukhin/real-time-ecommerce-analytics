import unittest

from ml.prepare_dataset import build_synthetic_dataset, summarize_dataset
from ml.schema import MODEL_FEATURES


class PrepareDatasetTests(unittest.TestCase):
    def test_generated_rows_contain_all_model_features(self):
        rows = build_synthetic_dataset(sample_count=50, seed=7, user_count=10)

        self.assertEqual(len(rows), 50)
        for feature_name in MODEL_FEATURES:
            self.assertIn(feature_name, rows[0])
        self.assertIn(rows[0]["event_type"], {"view", "click", "add_to_cart"})
        self.assertIn(rows[0]["label"], {0, 1})

    def test_dataset_summary_contains_event_mix(self):
        rows = build_synthetic_dataset(sample_count=80, seed=11, user_count=12)
        summary = summarize_dataset(rows)

        self.assertEqual(summary["row_count"], 80)
        self.assertIn("event_mix", summary)
        self.assertGreater(summary["positive_rate"], 0.0)
