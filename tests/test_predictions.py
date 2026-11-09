import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main


class TrendPredictionTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.feed = {
            "region": "IN",
            "count": 1,
            "trends": [{
                "title": "Sample trend",
                "summary": "A test topic",
                "category": "Technology",
                "source": "Google Trends + GDELT",
                "trend_score": 45,
                "trend_stage": "rising",
                "growth_delta": 10,
                "published_at": "2026-11-09T10:00:00+00:00",
            }],
        }
        self.history = [
            {"timestamp": "2026-11-09T10:00:00+00:00",
             "items": {"sample trend": {"score": 30, "title": "Sample trend"}}},
            {"timestamp": "2026-11-09T10:15:00+00:00",
             "items": {"sample trend": {"score": 35, "title": "Sample trend"}}},
            {"timestamp": "2026-11-09T10:30:00+00:00",
             "items": {"sample trend": {"score": 45, "title": "Sample trend"}}},
        ]

    def test_estimates_direction_from_existing_snapshots(self):
        with patch.object(main, "trends", new=AsyncMock(return_value=self.feed)), \
             patch.dict(main._trend_history, {"IN": self.history}):
            response = self.client.get("/api/predictions?region=IN&limit=5")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 1)
        prediction = body["predictions"][0]
        self.assertEqual(prediction["outlook"], "accelerating")
        self.assertEqual(prediction["prediction_status"], "estimated")
        self.assertEqual(prediction["observations"], 3)
        self.assertEqual(prediction["projected_score"], 52)
        self.assertEqual(prediction["confidence"], "moderate")

    def test_cold_start_reports_collecting_instead_of_guessing(self):
        with patch.object(main, "trends", new=AsyncMock(return_value=self.feed)), \
             patch.dict(main._trend_history, {"IN": []}):
            response = self.client.get("/api/predictions?region=IN")

        self.assertEqual(response.status_code, 200)
        prediction = response.json()["predictions"][0]
        self.assertEqual(prediction["prediction_status"], "collecting")
        self.assertIsNone(prediction["projected_score"])
        self.assertEqual(prediction["confidence"], "low")

    def test_limit_above_maximum_is_rejected(self):
        with patch.object(main, "trends", new=AsyncMock(return_value=self.feed)) as trends_mock:
            response = self.client.get("/api/predictions?limit=21")

        self.assertEqual(response.status_code, 422)
        trends_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
