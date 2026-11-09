import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class OpportunityFinderTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.feed = {
            "region": "IN",
            "count": 2,
            "trends": [
                {
                    "title": "Sample rising topic",
                    "summary": "A test trend",
                    "category": "Technology",
                    "source": "Google Trends + GDELT",
                    "trend_score": 80,
                    "trend_stage": "rising",
                    "growth_delta": None,
                    "published_at": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "title": "Older topic",
                    "summary": "Another test trend",
                    "category": "News",
                    "source": "GDELT",
                    "trend_score": 40,
                    "trend_stage": "stable",
                    "growth_delta": None,
                    "published_at": "2026-01-01T00:00:00+00:00",
                },
            ],
            "sources": {"Google Trends": "cached", "GDELT": "cached"},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def test_returns_ranked_opportunities_without_fetching_feeds(self):
        with patch.object(main, "trends", return_value=self.feed) as trends_mock:
            response = self.client.get("/api/opportunities?region=IN&limit=10")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["region"], "IN")
        self.assertEqual(body["count"], 2)
        self.assertEqual(body["opportunities"][0]["title"], "Sample rising topic")
        self.assertEqual(body["opportunities"][0]["opportunity_score"], 88)
        self.assertEqual(body["opportunities"][0]["competition_status"], "not_measured")
        self.assertIsNone(body["opportunities"][0]["signals"]["competition"])
        trends_mock.assert_called_once_with(region="IN", q=None)

    def test_limit_above_maximum_is_rejected(self):
        with patch.object(main, "trends", return_value=self.feed) as trends_mock:
            response = self.client.get("/api/opportunities?limit=21")

        self.assertEqual(response.status_code, 422)
        trends_mock.assert_not_called()

    def test_empty_feed_returns_empty_opportunities(self):
        empty_feed = {**self.feed, "trends": [], "count": 0}
        with patch.object(main, "trends", return_value=empty_feed):
            response = self.client.get("/api/opportunities")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["opportunities"], [])
        self.assertEqual(response.json()["count"], 0)


if __name__ == "__main__":
    unittest.main()
