import unittest

from ml.schema import normalize_prediction_payload, parse_hour_of_day


class SchemaTests(unittest.TestCase):
    def test_parse_hour_of_day_handles_iso_timestamp(self):
        self.assertEqual(parse_hour_of_day("2026-04-01T21:15:00"), 21)

    def test_normalize_prediction_payload_converts_numeric_types(self):
        payload = normalize_prediction_payload(
            {
                "event_id": 123,
                "user_id": "7",
                "timestamp": "2026-04-01T12:00:00",
                "price": "999.5",
                "price_to_user_mean": "1.2",
                "hour_of_day": "12",
                "is_view": "1",
                "is_click": 0,
                "is_cart": 0,
                "session_event_index": "3",
                "session_click_count": "2",
                "session_cart_count": 0,
                "user_total_clicks": "9",
                "user_total_carts": "2",
                "user_total_purchases": "1",
            }
        )

        self.assertEqual(payload["event_id"], "123")
        self.assertEqual(payload["user_id"], 7)
        self.assertEqual(payload["price"], 999.5)
        self.assertEqual(payload["price_to_user_mean"], 1.2)
        self.assertEqual(payload["is_view"], 1)
        self.assertEqual(payload["hour_of_day"], 12)
        self.assertEqual(payload["session_event_index"], 3)
        self.assertEqual(payload["user_total_purchases"], 1)
