import unittest
from pathlib import Path

from ml.predict import load_model_artifact, predict, predict_proba
from ml.train_model import train_and_save_model
from tests.support import workspace_temp_dir


class PredictTests(unittest.TestCase):
    def setUp(self):
        load_model_artifact.cache_clear()

    def test_predict_returns_probability_and_model_version(self):
        with workspace_temp_dir() as temp_dir:
            result = train_and_save_model(output_dir=Path(temp_dir), sample_count=600, epochs=180)
            model_path = result["paths"]["model"]
            payload = {
                "event_id": "evt-1",
                "user_id": 10,
                "timestamp": "2026-04-01T18:30:00",
                "price": 1200.0,
                "price_to_user_mean": 0.9,
                "hour_of_day": 18,
                "is_view": 0,
                "is_click": 1,
                "is_cart": 0,
                "session_event_index": 3,
                "session_click_count": 1,
                "session_cart_count": 0,
                "user_total_clicks": 8,
                "user_total_carts": 1,
                "user_total_purchases": 0,
            }

            prediction = predict(payload, model_path=model_path)

            self.assertEqual(prediction["event_id"], "evt-1")
            self.assertEqual(prediction["user_id"], 10)
            self.assertEqual(prediction["model_version"], "baseline-logreg-v2")
            self.assertGreaterEqual(prediction["purchase_probability"], 0.0)
            self.assertLessEqual(prediction["purchase_probability"], 1.0)

    def test_cart_snapshot_scores_higher_than_view_snapshot(self):
        with workspace_temp_dir() as temp_dir:
            result = train_and_save_model(output_dir=Path(temp_dir), sample_count=800, epochs=220)
            model_path = result["paths"]["model"]

            base_payload = {
                "event_id": "evt-2",
                "user_id": 11,
                "timestamp": "2026-04-01T19:00:00",
                "price": 1400.0,
                "price_to_user_mean": 0.95,
                "hour_of_day": 19,
                "session_event_index": 4,
                "session_click_count": 2,
                "session_cart_count": 0,
                "user_total_clicks": 12,
                "user_total_carts": 2,
                "user_total_purchases": 1,
            }

            view_score = predict_proba(
                {
                    **base_payload,
                    "is_view": 1,
                    "is_click": 0,
                    "is_cart": 0,
                },
                model_path=model_path,
            )
            cart_score = predict_proba(
                {
                    **base_payload,
                    "is_view": 0,
                    "is_click": 0,
                    "is_cart": 1,
                },
                model_path=model_path,
            )

            self.assertGreater(cart_score, view_score)
