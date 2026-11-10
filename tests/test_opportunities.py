import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

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

    def test_category_filter_returns_only_matching_creator_topics(self):
        category_feed = {
            **self.feed,
            "trends": [
                {
                    **self.feed["trends"][0],
                    "title": "Pawan Kalyan movie update",
                    "summary": "Entertainment news",
                },
                {
                    **self.feed["trends"][1],
                    "title": "India cricket match update",
                    "summary": "Sports news",
                },
            ],
        }
        with patch.object(main, "trends", return_value=category_feed), \
             patch.object(main, "fetch_creator_category_news", new=AsyncMock(return_value=[])):
            response = self.client.get("/api/opportunities?category=Entertainment")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["selected_category"], "Entertainment")
        self.assertEqual(len(body["opportunities"]), 1)
        self.assertEqual(body["opportunities"][0]["title"], "Pawan Kalyan movie update")

    def test_unknown_category_is_rejected(self):
        with patch.object(main, "trends", return_value=self.feed) as trends_mock:
            response = self.client.get("/api/opportunities?category=MadeUp")

        self.assertEqual(response.status_code, 422)
        trends_mock.assert_not_called()

    
    def test_category_specific_research_supplements_empty_category(self):
        category_feed = {
            **self.feed,
            "trends": [
                {
                    **self.feed["trends"][0],
                    "title": "Sample rising topic",
                    "summary": "A test trend",
                },
            ],
        }
        targeted = [{
            "title": "BGMI India esports tournament announced",
            "summary": "Recent gaming coverage",
            "source": "Google News category search",
            "source_url": "https://news.google.com/example",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "category": "Gaming",
            "creator_category": "Gaming",
            "trend_score": 70,
            "trend_stage": "new",
            "growth_delta": None,
        }]
        with patch.object(main, "trends", return_value=category_feed), \
             patch.object(main, "fetch_creator_category_news", new=AsyncMock(return_value=targeted)) as search_mock:
            response = self.client.get("/api/opportunities?category=Gaming")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["category_research"], "ok")
        self.assertEqual(body["category_research_count"], 1)
        self.assertEqual(len(body["opportunities"]), 1)
        self.assertEqual(body["opportunities"][0]["title"], targeted[0]["title"])
        self.assertEqual(body["opportunities"][0]["category"], "Gaming")
        search_mock.assert_awaited_once()

    def test_category_search_terms_exist_for_each_creator_category(self):
        self.assertEqual(set(main.CREATOR_CATEGORY_SEARCH_TERMS), set(main.CREATOR_CATEGORIES))
        self.assertTrue(all(main.CREATOR_CATEGORY_SEARCH_TERMS.values()))

    def test_empty_feed_returns_empty_opportunities(self):
        empty_feed = {**self.feed, "trends": [], "count": 0}
        with patch.object(main, "trends", return_value=empty_feed):
            response = self.client.get("/api/opportunities")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["opportunities"], [])
        self.assertEqual(response.json()["count"], 0)


if __name__ == "__main__":
    unittest.main()
