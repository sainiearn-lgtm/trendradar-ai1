from __future__ import annotations

import asyncio
import base64
import html
import json
import logging
import os
import re
import time
import subprocess
import sys
import tempfile
import wave
from array import array
from pathlib import Path
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse, quote, urljoin

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("trendradar")

APP_VERSION = "1.9.4"

# Read both keys. Empty keys are ignored; never log key values.
GEMINI_API_KEYS = [
    value.strip()
    for value in (
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("GEMINI_API_KEY_2", ""),
    )
    if value and value.strip()
]

# Optional Pixabay API key for stock photos. Never log this value.
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "").strip()
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY", "").strip()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
# Optional: set this in Render only if you have confirmed that the model is
# available to your Google AI Studio project. Otherwise leave it blank.
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "900"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "12"))
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "90"))
# Total attempts per model/key combination (1 to 3).
GEMINI_MAX_RETRIES = max(1, min(3, int(os.getenv("GEMINI_MAX_RETRIES", "2"))))
MAX_TRENDS = 35

REGIONS = {
    "IN": {"name": "India", "gdelt": "India", "youtube": "IN"},
    "US": {"name": "United States", "gdelt": "United States", "youtube": "US"},
    "GB": {"name": "United Kingdom", "gdelt": "United Kingdom", "youtube": "GB"},
    "CA": {"name": "Canada", "gdelt": "Canada", "youtube": "CA"},
    "AU": {"name": "Australia", "gdelt": "Australia", "youtube": "AU"},
    "DE": {"name": "Germany", "gdelt": "Germany", "youtube": "DE"},
}

app = FastAPI(title="TrendRadar AI API", version=APP_VERSION)
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_cache: dict[str, tuple[float, Any]] = {}
_rate: dict[str, list[float]] = {}
_rate_lock = asyncio.Lock()
_pixabay_search_cache: dict[str, tuple[float, list[dict]]] = {}
_pixabay_image_cache: dict[str, tuple[float, bytes, str]] = {}


def cached(key: str):
    item = _cache.get(key)
    if item and time.monotonic() - item[0] < CACHE_TTL:
        return item[1]
    return None


def put_cache(key: str, value: Any):
    _cache[key] = (time.monotonic(), value)


@app.middleware("http")
async def basic_rate_limit(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.method in {"GET", "POST"}:
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        async with _rate_lock:
            hits = [t for t in _rate.get(ip, []) if now - t < 60]
            if len(hits) >= 45:
                return JSONResponse(status_code=429, content={"detail": "Rate limit reached. Please wait a minute."})
            hits.append(now)
            _rate[ip] = hits
    return await call_next(request)


def clean_text(value: Any, limit: int = 500) -> str:
    text_value = html.unescape(re.sub(r"<[^>]*>", " ", str(value or "")))
    return re.sub(r"\s+", " ", text_value).strip()[:limit]


def safe_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None


def score_item(published: datetime | None, mentions: int, source_count: int, popularity: int = 0) -> int:
    age_hours = max(0.0, (datetime.now(timezone.utc) - (published or datetime.now(timezone.utc))).total_seconds() / 3600)
    recency = max(0, 35 - int(age_hours * 1.15))
    repetition = min(30, max(0, (mentions - 1) * 8))
    source_signal = min(20, source_count * 10)
    popularity_signal = min(15, max(0, popularity))
    return max(0, min(100, 30 + recency + repetition + source_signal + popularity_signal))


async def fetch_google_trends(client: httpx.AsyncClient, region: str) -> list[dict]:
    url = f"https://trends.google.com/trending/rss?geo={region}"
    response = await client.get(url, headers={"User-Agent": "TrendRadarAI/1.0"}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.text)
    rows = []
    for item in root.findall(".//item"):
        title = clean_text(item.findtext("title"), 180)
        link = clean_text(item.findtext("link"), 1000)
        desc = clean_text(item.findtext("description"), 450)
        pub = parse_date(item.findtext("pubDate"))
        if title and safe_http_url(link):
            # Google Trends RSS embeds related news items for many trending searches.
            # These explain the event/context behind a keyword without another API call.
            related_news = []
            for news_item in item.iter():
                if not str(news_item.tag).endswith("news_item"):
                    continue
                fields = {}
                for child in news_item.iter():
                    name = str(child.tag).split("}")[-1]
                    if name in {"news_item_title", "news_item_snippet", "news_item_url", "news_item_source"}:
                        fields[name] = clean_text(child.text, 500)
                news_url = fields.get("news_item_url", "")
                news_title = fields.get("news_item_title", "")
                if news_title:
                    related_news.append({
                        "title": news_title,
                        "snippet": fields.get("news_item_snippet", ""),
                        "source": fields.get("news_item_source", ""),
                        "url": news_url if safe_http_url(news_url) else "",
                    })
                if len(related_news) >= 5:
                    break
            reason_parts = [
                f"{entry['title']}: {entry['snippet']}".strip(": ")
                for entry in related_news if entry.get("title")
            ]
            rows.append({
                "title": title, "summary": desc or f"Google Trends search interest is rising for {title}.",
                "source": "Google Trends", "source_url": link,
                "published_at": (pub or datetime.now(timezone.utc)).isoformat(),
                "category": "Search trend", "mentions": 1, "popularity": 8,
                "related_news": related_news,
                "why_trending": "Google Trends is showing increased search interest. Related coverage: " + " | ".join(reason_parts) if reason_parts else "",
            })
    return rows


async def fetch_gdelt(client: httpx.AsyncClient, region: str) -> list[dict]:
    country = REGIONS[region]["gdelt"]
    # Restrict articles to publishers originating in the selected country.
    # A plain search for the word "India" also returns international stories
    # that merely mention India, which polluted the India trend feed.
    country_names = {"IN": "india", "US": "unitedstates", "GB": "unitedkingdom",
                     "CA": "canada", "AU": "australia", "DE": "germany"}
    country_codes = {"IN": {"india"}, "US": {"united states"},
                     "GB": {"united kingdom"}, "CA": {"canada"},
                     "AU": {"australia"}, "DE": {"germany"}}
    params = {
        "query": f"sourcecountry:{country_names.get(region, 'india')}",
        "mode": "ArtList", "format": "json",
        "sort": "HybridRel", "maxrecords": "50", "timespan": "24h",
    }
    response = await client.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    rows = []
    for article in data.get("articles", []) or []:
        source_country = clean_text(article.get("sourcecountry"), 80).casefold()
        if source_country not in country_codes.get(region, set()):
            continue
        title = clean_text(article.get("title"), 180)
        url = article.get("url", "")
        if not title or not safe_http_url(url):
            continue
        published = parse_date(article.get("seendate"))
        domain = clean_text(article.get("domain") or urlparse(url).netloc, 100)
        rows.append({
            "title": title,
            "summary": clean_text(article.get("socialimage") or f"Recent news coverage from {country}.", 450),
            "source": f"GDELT · {domain}", "source_url": url,
            "published_at": (published or datetime.now(timezone.utc)).isoformat(),
            "category": clean_text(article.get("tone"), 40) or "News",
            "mentions": 1, "popularity": 2,
        })
    return rows


async def fetch_youtube(client: httpx.AsyncClient, region: str) -> list[dict]:
    if not YOUTUBE_API_KEY:
        return []
    params = {"part": "snippet,statistics", "chart": "mostPopular", "regionCode": region,
              "maxResults": 25, "key": YOUTUBE_API_KEY}
    response = await client.get("https://www.googleapis.com/youtube/v3/videos", params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    rows = []
    for video in response.json().get("items", []):
        snippet = video.get("snippet", {})
        stats = video.get("statistics", {})
        title = clean_text(snippet.get("title"), 180)
        video_id = video.get("id")
        if not title or not video_id:
            continue
        views = int(stats.get("viewCount", 0) or 0)
        rows.append({
            "title": title, "summary": clean_text(snippet.get("description"), 450) or "A currently popular YouTube video in this region.",
            "source": "YouTube mostPopular", "source_url": f"https://www.youtube.com/watch?v={video_id}",
            "published_at": snippet.get("publishedAt") or datetime.now(timezone.utc).isoformat(),
            "category": clean_text(snippet.get("categoryId"), 40) or "Video",
            "mentions": 1, "popularity": min(15, int(views / 10_000_000)),
        })
    return rows


def merge_and_rank(rows: list[dict], region: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = re.sub(r"[^a-z0-9]+", " ", row["title"].lower()).strip()
        if key:
            groups.setdefault(key, []).append(row)
    output = []
    for key, group in groups.items():
        best = max(group, key=lambda x: parse_date(x.get("published_at")) or datetime.min.replace(tzinfo=timezone.utc))
        sources = {x["source"].split(" · ")[0] for x in group}
        pub = parse_date(best.get("published_at"))
        output.append({
            "id": str(abs(hash(key))), "title": best["title"], "summary": best["summary"],
            "source": " + ".join(sorted(sources)), "source_url": best["source_url"],
            "region": region, "category": best.get("category") or "General",
            "trend_score": score_item(pub, len(group), len(sources), max(x.get("popularity", 0) for x in group)),
            "published_at": best.get("published_at"), "mentions": len(group),
            "related_news": list({
                (news.get("title", ""), news.get("url", "")): news
                for row in group for news in (row.get("related_news") or [])
                if news.get("title")
            }.values())[:5],
            "why_trending": next((row.get("why_trending") for row in group if row.get("why_trending")), ""),
        })
    output.sort(key=lambda x: (x["trend_score"], x["published_at"] or ""), reverse=True)
    return output[:MAX_TRENDS]


# Lightweight, process-local trend history. No database, disk writes, background
# polling, or additional feed requests; history resets when the service restarts.
TREND_HISTORY_MAX_SNAPSHOTS = 96  # up to 24 hours when requests arrive every 15 min
_trend_history: dict[str, list[dict[str, Any]]] = {}


def record_and_classify_trends(rows: list[dict], region: str) -> list[dict]:
    """Compare a fresh feed result with the previous in-memory snapshot."""
    now = datetime.now(timezone.utc)
    history = _trend_history.setdefault(region, [])
    previous = history[-1] if history else {}
    previous_items = previous.get("items", {})
    first_seen = {}
    for snapshot in history:
        for key in snapshot.get("items", {}):
            first_seen.setdefault(key, snapshot.get("timestamp", now.isoformat()))

    classified = []
    snapshot_items = {}
    enough_history = len(history) >= 1
    for row in rows:
        key = re.sub(r"[^a-z0-9]+", " ", row.get("title", "").casefold()).strip()
        old = previous_items.get(key)
        score = int(row.get("trend_score", 0) or 0)
        delta = score - int(old.get("score", score)) if old else 0
        points = len(history) + 1
        if not enough_history:
            stage = "collecting"
        elif not old:
            stage = "new"
        elif delta <= -5:
            stage = "declining"
        elif score >= 80:
            stage = "viral"
        elif delta >= 5 and score < 70:
            stage = "emerging"
        elif delta >= 3:
            stage = "rising"
        else:
            stage = "stable"

        item = dict(row)
        item.update({
            "trend_stage": stage,
            "growth_delta": delta if old else None,
            "history_points": points,
        })
        classified.append(item)
        snapshot_items[key] = {"score": score, "title": row.get("title", "")}

    # Surface early momentum before already-established high-score topics.
    stage_order = {"emerging": 0, "rising": 1, "viral": 2, "new": 3,
                   "collecting": 4, "stable": 5, "declining": 6}
    classified.sort(key=lambda item: (
        stage_order.get(item.get("trend_stage"), 5),
        -int(item.get("trend_score", 0)),
    ))

    # Keep only compact scores/titles; no article bodies or URLs are duplicated.
    history.append({"timestamp": now.isoformat(), "items": snapshot_items})
    if len(history) > TREND_HISTORY_MAX_SNAPSHOTS:
        del history[:-TREND_HISTORY_MAX_SNAPSHOTS]
    return classified


@app.get("/")
async def root():
    return {"name": "TrendRadar AI API", "status": "online", "version": APP_VERSION, "docs": "/docs"}


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEYS),
        "gemini_keys_configured": len(GEMINI_API_KEYS),
        "gemini_model": GEMINI_MODEL,
        "youtube_configured": bool(YOUTUBE_API_KEY),
        "pollinations_configured": bool(POLLINATIONS_API_KEY),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/trends")
async def trends(region: str = "IN", q: str | None = None):
    region = region.upper()
    if region not in REGIONS:
        raise HTTPException(status_code=422, detail=f"Unsupported region. Choose: {', '.join(REGIONS)}")
    cache_key = f"trends:{region}"
    rows = cached(cache_key)
    source_status = {"Google Trends": "not_checked", "GDELT": "not_checked",
                     "YouTube": "not_configured" if not YOUTUBE_API_KEY else "not_checked"}
    if rows is None:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = [
                ("Google Trends", fetch_google_trends(client, region)),
                ("GDELT", fetch_gdelt(client, region)),
                ("YouTube", fetch_youtube(client, region)),
            ]
            results = await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
        combined = []
        for (name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                source_status[name] = "unavailable"
                logger.warning("%s feed failed for %s: %s", name, region, result)
            else:
                source_status[name] = "ok" if result else ("no_data" if name != "YouTube" or YOUTUBE_API_KEY else "not_configured")
                combined.extend(result)
        rows = merge_and_rank(combined, region)
        if rows:
            rows = record_and_classify_trends(rows, region)
        if not rows:
            raise HTTPException(status_code=503, detail={"message": "Trend feeds are temporarily unavailable or returned no results. Please retry shortly.", "sources": source_status})
        put_cache(cache_key, rows)
    else:
        source_status = {"Google Trends": "cached", "GDELT": "cached",
                         "YouTube": "cached" if YOUTUBE_API_KEY else "not_configured"}
    filtered = rows
    if q:
        query = clean_text(q, 80).lower()
        filtered = [r for r in rows if query in r["title"].lower() or query in r["summary"].lower() or query in r["category"].lower()]
    return {
        "region": region, "region_name": REGIONS[region]["name"], "count": len(filtered),
        "trends": filtered, "sources": source_status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "score_note": "Trend scores are TrendRadar estimates based on recency, repeated feed mentions, source coverage and available popularity signals. They are not official platform virality metrics.",
    }


@app.get("/api/predictions")
async def trend_predictions(region: str = "IN", limit: int = 10):
    """Estimate near-term trend direction from observed in-memory snapshots.

    This reuses the existing trends cache/feed path. It does not call an AI model,
    add polling, or claim to predict virality or trend lifetime.
    """
    if limit < 1 or limit > 20:
        raise HTTPException(status_code=422, detail="Limit must be between 1 and 20.")

    region = region.upper()
    feed = await trends(region=region, q=None)
    history = _trend_history.get(region, [])
    latest_snapshot = history[-1] if history else {}
    latest_items = latest_snapshot.get("items", {})
    predictions = []

    for item in feed.get("trends", []):
        key = re.sub(r"[^a-z0-9]+", " ", item.get("title", "").casefold()).strip()
        observed = []
        for snapshot in history:
            record = snapshot.get("items", {}).get(key)
            if record and isinstance(record.get("score"), (int, float)):
                observed.append({
                    "timestamp": snapshot.get("timestamp"),
                    "score": int(record["score"]),
                })
        observed = observed[-5:]
        current_score = max(0, min(100, int(item.get("trend_score", 0) or 0)))

        if len(observed) < 2:
            predictions.append({
                **item,
                "prediction_status": "collecting",
                "outlook": "insufficient_history",
                "outlook_label": "Collecting signals",
                "momentum_score": None,
                "projected_score": None,
                "confidence": "low",
                "observations": len(observed),
                "score_changes": [],
                "explanation": "At least two trend snapshots are needed before estimating direction.",
            })
            continue

        changes = [
            observed[index]["score"] - observed[index - 1]["score"]
            for index in range(1, len(observed))
        ]
        average_change = sum(changes) / len(changes)
        latest_change = changes[-1]
        acceleration = changes[-1] - changes[-2] if len(changes) > 1 else 0

        if latest_change <= -3:
            outlook, label = "declining", "Losing momentum"
        elif latest_change >= 3 and acceleration >= 2:
            outlook, label = "accelerating", "Momentum accelerating"
        elif latest_change >= 3:
            outlook, label = "rising", "Momentum rising"
        else:
            outlook, label = "steady", "Relatively steady"

        projected_score = max(0, min(100, round(current_score + average_change)))
        momentum_score = max(0, min(100, round(50 + average_change * 5)))
        confidence = "low" if len(observed) == 2 else "moderate"
        if len(observed) >= 5:
            confidence = "moderate"

        explanations = {
            "declining": "Recent snapshots show the trend score moving down.",
            "accelerating": "Recent score gains are getting stronger across snapshots.",
            "rising": "Recent snapshots show the trend score moving up.",
            "steady": "Recent snapshots show little directional change.",
        }
        predictions.append({
            **item,
            "prediction_status": "estimated",
            "outlook": outlook,
            "outlook_label": label,
            "momentum_score": momentum_score,
            "projected_score": projected_score,
            "confidence": confidence,
            "observations": len(observed),
            "score_changes": changes,
            "explanation": explanations[outlook],
        })

    order = {"accelerating": 0, "rising": 1, "steady": 2,
             "declining": 3, "insufficient_history": 4}
    predictions.sort(key=lambda row: (
        order.get(row["outlook"], 5),
        -(row.get("momentum_score") or 0),
        -int(row.get("trend_score", 0) or 0),
    ))
    return {
        "region": feed.get("region", region),
        "count": min(limit, len(predictions)),
        "predictions": predictions[:limit],
        "history_snapshots": len(history),
        "methodology": "Near-term direction estimate from observed TrendRadar score changes across recent in-memory snapshots.",
        "limitations": [
            "This is a directional estimate, not a prediction of virality, views, or trend lifetime.",
            "History is process-local and resets when the service restarts.",
            "A cold start needs at least two feed refresh snapshots before direction can be estimated.",
            "No additional polling, external feed requests, or AI-model calls are added by this endpoint."
        ],
    }


@app.get("/api/opportunities")
async def viral_opportunities(region: str = "IN", limit: int = 10):
    """Rank existing cached trends by transparent, data-backed content opportunity signals.

    This endpoint reuses /api/trends; it does not add feed requests or background work.
    Competition is intentionally not estimated because no reliable competitor dataset
    is currently connected.
    """
    if limit < 1 or limit > 20:
        raise HTTPException(status_code=422, detail="Limit must be between 1 and 20.")

    feed = await trends(region=region, q=None)
    now = datetime.now(timezone.utc)
    opportunities = []

    stage_momentum = {
        "emerging": 95, "rising": 85, "viral": 75, "new": 70,
        "collecting": 55, "stable": 45, "declining": 15,
    }

    for item in feed.get("trends", []):
        trend_score = max(0, min(100, int(item.get("trend_score", 0) or 0)))
        stage = str(item.get("trend_stage") or "collecting").lower()
        delta = item.get("growth_delta")

        momentum = stage_momentum.get(stage, 50)
        if isinstance(delta, (int, float)):
            momentum = max(0, min(100, int(50 + delta * 5)))

        published = parse_date(item.get("published_at"))
        age_hours = max(0, (now - published).total_seconds() / 3600) if published else 48
        freshness = 100 if age_hours <= 3 else 85 if age_hours <= 12 else 70 if age_hours <= 24 else 45 if age_hours <= 48 else 20

        source_names = [part.strip() for part in str(item.get("source", "")).split("+") if part.strip()]
        source_coverage = min(100, len(source_names) * 50)

        # Weighted estimate from signals the product actually has today.
        opportunity_score = round(
            trend_score * 0.40 + momentum * 0.30 +
            freshness * 0.20 + source_coverage * 0.10
        )
        if opportunity_score >= 75:
            label = "High opportunity"
        elif opportunity_score >= 55:
            label = "Promising"
        else:
            label = "Monitor"

        reasons = []
        if trend_score >= 70:
            reasons.append("Strong current trend score")
        if stage in {"emerging", "rising", "new"}:
            reasons.append("Early or growing trend stage")
        if age_hours <= 24:
            reasons.append("Fresh coverage within the last 24 hours")
        if len(source_names) > 1:
            reasons.append("Covered by multiple source types")
        if not reasons:
            reasons.append("Worth monitoring as more trend signals arrive")

        row = dict(item)
        row.update({
            "opportunity_score": opportunity_score,
            "opportunity_label": label,
            "opportunity_reasons": reasons,
            "signals": {
                "audience_interest": trend_score,
                "momentum": momentum,
                "freshness": freshness,
                "source_coverage": source_coverage,
                "competition": None,
            },
            "competition_status": "not_measured",
            "suggested_angle": "Explain what is happening, why it matters to the audience, and what may happen next.",
        })
        opportunities.append(row)

    opportunities.sort(
        key=lambda row: (row["opportunity_score"], row.get("trend_score", 0)),
        reverse=True,
    )
    return {
        "region": feed.get("region", region.upper()),
        "count": min(limit, len(opportunities)),
        "opportunities": opportunities[:limit],
        "methodology": "Estimate based on current trend score (40%), momentum (30%), freshness (20%) and source coverage (10%). This is not a prediction of virality.",
        "limitations": [
            "Creator competition is not measured because reliable competitor data is not connected.",
            "Scores are estimates from available trend feeds, not guaranteed reach or views.",
        ],
        "sources": feed.get("sources", {}),
        "updated_at": feed.get("updated_at"),
    }


class ScriptRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=5000)
    language: str = Field(default="English")
    format: str = Field(default="Shorts (60 seconds)")
    trend_context: str = Field(default="", max_length=6000)

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        value = clean_text(value, 5000)
        if len(value) < 2:
            raise ValueError("Please enter a meaningful topic.")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in {"Hindi", "Hinglish", "English"}:
            raise ValueError("Language must be Hindi, Hinglish, or English.")
        return value

    @field_validator("format")
    @classmethod
    def validate_format(cls, value: str) -> str:
        if value not in {"Shorts (60 seconds)", "Long video (5-10 minutes)"}:
            raise ValueError("Unsupported video format.")
        return value


def is_temporary_gemini_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    message = str(exc).lower()
    return (
        status in {408, 429, 500, 502, 503, 504}
        or any(token in message for token in (
            "503", "service unavailable", "overloaded", "resource_exhausted",
            "429", "rate limit", "temporarily unavailable", "deadline exceeded",
            "internal server error", "bad gateway", "gateway timeout",
        ))
    )


async def _generate_with_one_key(genai, api_key: str, model: str, prompt: str):
    client = genai.Client(api_key=api_key)
    return await asyncio.wait_for(
        asyncio.to_thread(client.models.generate_content, model=model, contents=prompt),
        timeout=GEMINI_TIMEOUT,
    )


async def generate_with_key_fallback(prompt: str):
    """Try configured Gemini keys/models and briefly retry temporary service errors."""
    from google import genai

    models = [GEMINI_MODEL]
    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
        models.append(GEMINI_FALLBACK_MODEL)

    last_error: Exception | None = None

    for model in models:
        for key_index, api_key in enumerate(GEMINI_API_KEYS, start=1):
            for attempt in range(1, GEMINI_MAX_RETRIES + 1):
                try:
                    logger.info(
                        "Trying Gemini model %s, key slot %s, attempt %s/%s",
                        model, key_index, attempt, GEMINI_MAX_RETRIES,
                    )
                    response = await _generate_with_one_key(
                        genai, api_key, model, prompt
                    )
                    return response, model, key_index

                except asyncio.TimeoutError as exc:
                    last_error = exc
                    logger.warning(
                        "Gemini timed out for model %s, key slot %s, attempt %s",
                        model, key_index, attempt,
                    )
                    # A long timeout is not worth repeating on the same key.
                    break

                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Gemini attempt failed for model %s, key slot %s, attempt %s: %s",
                        model, key_index, attempt, exc,
                    )

                    # Retry only temporary errors. Invalid keys/model names need
                    # configuration changes, not repeated requests.
                    if not is_temporary_gemini_error(exc):
                        break

                    if attempt < GEMINI_MAX_RETRIES:
                        delay = min(2 ** attempt, 5)
                        logger.info(
                            "Temporary Gemini error; retrying in %s seconds",
                            delay,
                        )
                        await asyncio.sleep(delay)

    if last_error:
        raise last_error
    raise RuntimeError("No Gemini API key is configured.")


@app.post("/api/script")
async def generate_script(payload: ScriptRequest):
    if not GEMINI_API_KEYS:
        raise HTTPException(status_code=503, detail="AI script generation is not configured. Add GEMINI_API_KEY to Render.")
    if payload.format.startswith("Shorts"):
        structure = "Create a punchy 45–60 second vertical video script. Start with a powerful first-3-second hook, use fast-paced storytelling and interesting accurate context, then end with a memorable CTA."
    else:
        structure = "Create a detailed 5–10 minute YouTube script with a compelling intro, clear sections, detailed explanation, examples, transitions, conclusion and CTA."
    context_block = (
        "\nTREND CONTEXT (use this to explain why this topic is trending):\n"
        + payload.trend_context.strip()
        + "\nUse this context as the story angle. Distinguish confirmed details from speculation. "
          "Do not invent missing facts; if context is weak or incomplete, say so naturally and avoid presenting guesses as facts.\n"
        if payload.trend_context.strip() else
        "\nNo verified trend context was supplied. Do not pretend to know why this topic is trending; build a general explainer and flag facts needing verification.\n"
    )
    prompt = f"""You are an expert YouTube and Instagram video scriptwriter. Write naturally and engagingly in {payload.language}.
Topic: {payload.topic}
Format: {payload.format}
{context_block}
Requirements: {structure}
Language rule: write ALL spoken narration in natural Hindi using Devanagari script. Do not write Roman Hindi or English inside spoken narration.
Return polished creator-ready content with these exact sections:
1. CATCHY TITLE
2. OPENING HOOK
3. COMPLETE VIDEO SCRIPT — this section must contain ONLY the words the narrator should speak, in Hindi Devanagari. Do not include visual directions, camera notes, editor instructions, sound cues, labels such as HOOK/SCENE/VOICEOVER, or explanations inside this section. Keep every spoken sentence on its own line.
4. SCENE-BY-SCENE VISUAL SUGGESTIONS — keep all production instructions here, outside narration.
5. THUMBNAIL TEXT (3 options)
6. VIDEO DESCRIPTION
7. HASHTAGS (8-12 relevant tags)
8. CALL TO ACTION
Do not invent statistics, quotes, breaking-news claims, or factual specifics. If a claim needs verification, mark it [VERIFY] outside spoken narration. Avoid robotic phrasing and filler. Output plain text with clear headings."""
    try:
        response, used_model, used_key_slot = await generate_with_key_fallback(prompt)
        script = (response.text or "").strip()
        if not script:
            raise RuntimeError("Gemini returned an empty response.")
        logger.info("Gemini generation succeeded using model %s and key slot %s", used_model, used_key_slot)
        return {
            "script": script, "topic": payload.topic, "language": payload.language,
            "format": payload.format, "model": used_model,
        }
    except asyncio.TimeoutError:
        logger.exception("Gemini generation timed out after trying configured fallbacks")
        raise HTTPException(status_code=504, detail="Gemini is taking too long. Please try again in a minute.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gemini generation failed after trying configured keys/models")
        if is_temporary_gemini_error(exc):
            raise HTTPException(
                status_code=503,
                detail="Gemini is temporarily busy on the configured model/key accounts. Please try again shortly.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="AI generation failed. Check Gemini API key/model configuration in Render.",
        ) from exc


@app.get("/api/video-research")
async def video_research(
    q: str = "India",
    platform: str = "youtube",
    region: str = "IN",
):
    q = clean_text(q, 100)
    platform = platform.lower().strip()

    if len(q) < 2:
        raise HTTPException(status_code=422, detail="Please enter a search topic.")

    if platform not in {"youtube", "instagram"}:
        raise HTTPException(status_code=422, detail="Choose YouTube or Instagram.")

    async with httpx.AsyncClient(follow_redirects=True) as client:

        # YouTube public video research
        if platform == "youtube":
            if not YOUTUBE_API_KEY:
                raise HTTPException(
                    status_code=503,
                    detail="YouTube API key is not configured. Add YOUTUBE_API_KEY in Render."
                )

            published_after = (
                datetime.now(timezone.utc).timestamp() - 30 * 86400
            )
            after_date = datetime.fromtimestamp(
                published_after, timezone.utc
            ).isoformat().replace("+00:00", "Z")

            search_response = await client.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet",
                    "type": "video",
                    "q": q,
                    "order": "viewCount",
                    "publishedAfter": after_date,
                    "maxResults": 15,
                    "regionCode": region,
                    "key": YOUTUBE_API_KEY,
                },
                timeout=REQUEST_TIMEOUT,
            )
            search_response.raise_for_status()

            items = search_response.json().get("items", [])
            ids = [
                item.get("id", {}).get("videoId")
                for item in items
                if item.get("id", {}).get("videoId")
            ]

            if not ids:
                return {"platform": platform, "query": q, "results": []}

            stats_response = await client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "part": "snippet,statistics",
                    "id": ",".join(ids),
                    "key": YOUTUBE_API_KEY,
                },
                timeout=REQUEST_TIMEOUT,
            )
            stats_response.raise_for_status()

            results = []

            for video in stats_response.json().get("items", []):
                snippet = video.get("snippet", {})
                stats = video.get("statistics", {})
                video_id = video.get("id")

                views = int(stats.get("viewCount", 0) or 0)
                likes = int(stats.get("likeCount", 0) or 0)
                comments = int(stats.get("commentCount", 0) or 0)

                published = parse_date(snippet.get("publishedAt"))
                age_hours = max(
                    1,
                    (datetime.now(timezone.utc) -
                     (published or datetime.now(timezone.utc))).total_seconds() / 3600
                )

                views_per_hour = int(views / age_hours)
                engagement = round(
                    ((likes + comments) / views * 100), 2
                ) if views else 0

                viral_score = min(
                    100,
                    int(
                        min(45, views_per_hour / 1000)
                        + min(30, engagement * 5)
                        + min(25, views / 1000000 * 10)
                    )
                )

                thumbnails = snippet.get("thumbnails", {})
                thumbnail = (
                    thumbnails.get("high")
                    or thumbnails.get("medium")
                    or thumbnails.get("default")
                    or {}
                )

                results.append({
                    "title": clean_text(snippet.get("title"), 200),
                    "channel": clean_text(snippet.get("channelTitle"), 100),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "thumbnail": thumbnail.get("url", ""),
                    "views": views,
                    "likes": likes,
                    "comments": comments,
                    "views_per_hour": views_per_hour,
                    "engagement": engagement,
                    "viral_score": viral_score,
                    "published_at": snippet.get("publishedAt"),
                    "platform": "YouTube",
                })

            results.sort(
                key=lambda item: item["viral_score"],
                reverse=True
            )

            return {
                "platform": "youtube",
                "query": q,
                "results": results,
                "score_note": "TrendRadar estimate, not an official YouTube metric."
            }

        # Instagram: public indexed Reel discovery.
        # This is not a complete live Instagram trending feed.
        import xml.etree.ElementTree as ET

        rss_response = await client.get(
            "https://news.google.com/rss/search",
            params={
                "q": f"site:instagram.com/reel {q}",
                "hl": "en-IN",
                "gl": "IN",
                "ceid": "IN:en",
            },
            headers={"User-Agent": "Mozilla/5.0 TrendRadarAI/1.0"},
            timeout=REQUEST_TIMEOUT,
        )
        rss_response.raise_for_status()

        root = ET.fromstring(rss_response.text)
        results = []

        for item in root.findall(".//item")[:15]:
            title = clean_text(item.findtext("title"), 200)
            link = clean_text(item.findtext("link"), 1000)
            published = item.findtext("pubDate")

            if not title or not link or not safe_http_url(link):
                continue

            results.append({
                "title": title,
                "url": link,
                "published_at": published,
                "platform": "Instagram",
                "note": "Public indexed discovery. Views and likes are not verified."
            })

        return {
            "platform": "instagram",
            "query": q,
            "results": results,
            "notice": "Instagram results are based on publicly indexed content, not a complete live trending feed."
        }


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled API error at %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "An unexpected server error occurred."})


@app.get("/api/news")
async def hindi_news(limit: int = 40):
    import xml.etree.ElementTree as ET

    limit = max(5, min(limit, 60))
    cache_key = f"hindi-news:{limit}"
    cached_rows = cached(cache_key)

    if cached_rows is not None:
        return {
            "count": len(cached_rows),
            "articles": cached_rows,
            "cached": True
        }

    feeds = [
        ("Aaj Tak", "https://www.aajtak.in/rssfeeds/?id=home"),
        ("BBC Hindi", "https://feeds.bbci.co.uk/hindi/rss.xml"),
        ("NDTV India", "https://feeds.feedburner.com/ndtvkhabar-india"),
        ("Amar Ujala", "https://www.amarujala.com/rss/breaking-news.xml"),
        ("Dainik Bhaskar", "https://www.bhaskar.com/rss-v1--category-1061.xml"),
    ]

    async def fetch_feed(client, name, url):
        try:
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 TrendRadarAI/1.2"},
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            root = ET.fromstring(response.text)
            rows = []

            for item in root.findall(".//item")[:30]:
                title = clean_text(item.findtext("title"), 240)
                link = clean_text(item.findtext("link"), 1200)
                description = clean_text(
                    item.findtext("description"), 500
                )
                published = parse_date(item.findtext("pubDate"))

                if title and safe_http_url(link):
                    rows.append({
                        "title": title,
                        "summary": description,
                        "source": name,
                        "url": link,
                        "published_at": (
                            published.isoformat() if published else None
                        )
                    })

            return name, rows

        except Exception as exc:
            logger.warning("News feed %s failed: %s", name, exc)
            return name, []

    async with httpx.AsyncClient(follow_redirects=True) as client:
        results = await asyncio.gather(
            *[
                fetch_feed(client, name, url)
                for name, url in feeds
            ]
        )

    articles = []
    statuses = {}
    seen = set()

    for name, rows in results:
        statuses[name] = "ok" if rows else "unavailable"

        for row in rows:
            key = re.sub(
                r"[^a-z0-9]+", " ",
                row["title"].lower()
            ).strip()

            if key and key not in seen:
                seen.add(key)
                articles.append(row)

    def sort_date(article):
        date = parse_date(article["published_at"])
        return date.timestamp() if date else 0

    articles.sort(key=sort_date, reverse=True)
    articles = articles[:limit]

    if not articles:
        raise HTTPException(
            status_code=503,
            detail="Hindi news feeds temporarily unavailable."
        )

    put_cache(cache_key, articles)

    return {
        "count": len(articles),
        "articles": articles,
        "sources": statuses,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "notice": "Public RSS headlines. Verify important news with the original publisher."
    }


# -------------------- Voice Studio v1.6.1 (Indic Tiny TTS) --------------------

class VisualRequest(BaseModel):
    script: str = Field(min_length=20, max_length=12000)
    topic: str = Field(default="Video", max_length=500)


class VoiceRequest(BaseModel):
    text: str = Field(min_length=2, max_length=12000)
    language: str = Field(default="Hindi")
    gender: str = Field(default="male")
    speed: int = Field(default=100, ge=50, le=150)
    pitch: int = Field(default=0, ge=-6, le=6)  # semitones
    volume: int = Field(default=100, ge=0, le=200)

    @field_validator("text")
    @classmethod
    def validate_voice_text(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Please provide script text for voice generation.")
        return value

    @field_validator("language")
    @classmethod
    def validate_voice_language(cls, value: str) -> str:
        if value != "Hindi":
            raise ValueError("Indic Tiny TTS currently supports Hindi only. Select Hindi script language.")
        return value

    @field_validator("gender")
    @classmethod
    def validate_voice_gender(cls, value: str) -> str:
        if value not in {"male", "female"}:
            raise ValueError("Choose male or female voice.")
        return value


INDIC_TTS_REPO = "sulabhkatiyar/indic-tiny-tts-hindi"
INDIC_TTS_DIR = Path(os.getenv("INDIC_TTS_MODEL_DIR", "/tmp/trendradar-indic-tiny-tts"))
_indic_tts_instance = None
_indic_tts_lock = asyncio.Lock()


async def _get_indic_tts():
    """Download the small ONNX model once, then keep the CPU model in memory."""
    global _indic_tts_instance
    if _indic_tts_instance is not None:
        return _indic_tts_instance
    async with _indic_tts_lock:
        if _indic_tts_instance is not None:
            return _indic_tts_instance

        def load_model():
            from huggingface_hub import snapshot_download
            import sys

            INDIC_TTS_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Preparing Indic Tiny TTS model files from Hugging Face")
            # Download only runtime files. The previous broad snapshot also fetched
            # sample audio/assets, needlessly increasing first-request time and memory
            # pressure on small Render instances.
            model_dir = snapshot_download(
                repo_id=INDIC_TTS_REPO,
                revision="e15ac436560eb24d115d47c0783f62f10a02d30a",
                local_dir=str(INDIC_TTS_DIR),
                allow_patterns=[
                    "config.json",
                    "indic_tiny_tts/**",
                    "onnx/**",
                ],
                max_workers=4,
            )
            required = [
                Path(model_dir) / "config.json",
                Path(model_dir) / "onnx" / "text_encoder.onnx",
                Path(model_dir) / "onnx" / "flow_decoder.onnx",
                Path(model_dir) / "onnx" / "vocoder.onnx",
            ]
            missing = [str(item) for item in required if not item.is_file()]
            if missing:
                raise RuntimeError("Indic Tiny TTS model download is incomplete: " + ", ".join(missing))
            logger.info("Indic Tiny TTS runtime files downloaded; loading ONNX model")
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)
            from indic_tiny_tts import TinyTTSOnnx
            return TinyTTSOnnx.from_pretrained(model_dir)

        _indic_tts_instance = await asyncio.to_thread(load_model)
        logger.info("Indic Tiny TTS ONNX model loaded successfully")
        return _indic_tts_instance


def _render_indic_audio(text: str, gender: str, speed: int, pitch: int, volume: int) -> bytes:
    """Synthesize Hindi speech and return mono 24 kHz PCM WAV bytes."""
    import io
    import wave
    import numpy as np

    # Model reads Devanagari. Refuse Roman-only input instead of silently
    # producing missing/empty speech for Latin-script Hinglish or English.
    if not any("\u0900" <= char <= "\u097f" for char in text):
        raise ValueError(
            "Indic Tiny TTS needs Hindi text in Devanagari. Example: नमस्ते दोस्तों, आज की खबर..."
        )

    model = _indic_tts_instance
    if model is None:
        raise RuntimeError("Indic Tiny TTS model is not loaded.")

    # Speed is applied by the TTS engine itself. Keep it independent from
    # post-processing so changing pitch/volume cannot alter speaking rate.
    audio = model.synthesize(text, voice=gender, speed=float(speed) / 100.0)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise RuntimeError("Indic Tiny TTS returned empty audio.")
    if not np.all(np.isfinite(audio)):
        raise RuntimeError("Indic Tiny TTS returned invalid audio samples.")

    # Lightweight pitch shift using NumPy only. Resampling changes the
    # audible fundamental frequency without requiring librosa/soxr on Render.
    # Positive values sound higher and shorter; negative values sound lower
    # and longer. The TTS speaking-speed setting is still applied separately.
    if pitch:
        pitch_factor = 2.0 ** (float(pitch) / 12.0)
        target_length = max(1, int(round(audio.size / pitch_factor)))
        source_positions = np.arange(audio.size, dtype=np.float64)
        target_positions = np.linspace(0, audio.size - 1, target_length, dtype=np.float64)
        audio = np.interp(target_positions, source_positions, audio).astype(np.float32)
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            raise RuntimeError("Pitch processing returned invalid audio.")

    # Apply volume as gain. A simple peak normalizer would cancel the
    # volume slider whenever the signal clips, so use a gentle soft-knee
    # limiter that preserves the gain difference while preventing clipping.
    gain = float(volume) / 100.0
    audio = audio * gain
    magnitude = np.abs(audio)
    knee = 0.95
    over = magnitude > knee
    if np.any(over):
        compressed = knee + (1.0 - knee) * (1.0 - np.exp(-(magnitude[over] - knee) / (1.0 - knee)))
        audio[over] = np.sign(audio[over]) * compressed
    audio = np.clip(audio, -0.999, 0.999)

    pcm = (audio * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)
    return buffer.getvalue()



@app.get("/api/pixabay-image")
@app.get("/api/image-proxy")
async def proxy_pixabay_image(url: str):
    """Fetch images only from approved HTTPS hosts, validating every redirect."""
    allowed_hosts = (
        "pixabay.com", "wikimedia.org", "flickr.com", "staticflickr.com",
        "wordpress.com", "wp.com", "creativecommons.org", "openverse.org",
        "unsplash.com",
    )

    def approved_image_url(candidate: str) -> bool:
        try:
            parsed = urlparse(candidate)
            host = (parsed.hostname or "").lower().rstrip(".")
            allowed = any(
                host == domain or host.endswith("." + domain)
                for domain in allowed_hosts
            )
            # Reject credentials, non-standard ports, IP literals and non-HTTPS URLs.
            return (
                parsed.scheme == "https"
                and allowed
                and parsed.username is None
                and parsed.password is None
                and parsed.port in (None, 443)
            )
        except (TypeError, ValueError):
            return False

    if not approved_image_url(url):
        raise HTTPException(status_code=400, detail="Unsupported image host.")

    cached_image = _pixabay_image_cache.get(url)
    if cached_image and time.monotonic() - cached_image[0] < 86400:
        return Response(
            content=cached_image[1],
            media_type=cached_image[2],
            headers={"Cache-Control": "public, max-age=86400",
                     "Access-Control-Allow-Origin": "*"},
        )

    max_bytes = 15 * 1024 * 1024
    try:
        current_url = url
        image_bytes = bytearray()
        content_type = ""

        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(20.0, connect=8.0),
        ) as client:
            for redirect_count in range(6):
                if not approved_image_url(current_url):
                    raise HTTPException(status_code=400, detail="Image redirect host is not allowed.")

                async with client.stream(
                    "GET", current_url,
                    headers={"User-Agent": "Mozilla/5.0 TrendRadarAI"},
                ) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location or redirect_count >= 5:
                            raise HTTPException(status_code=502, detail="Image source redirected too many times.")
                        current_url = urljoin(current_url, location)
                        continue

                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].lower()
                    if not content_type.startswith("image/"):
                        raise HTTPException(status_code=415, detail="Image source did not return an image.")

                    content_length = resp.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > max_bytes:
                                raise HTTPException(status_code=415, detail="Image is too large.")
                        except ValueError:
                            pass

                    async for chunk in resp.aiter_bytes():
                        image_bytes.extend(chunk)
                        if len(image_bytes) > max_bytes:
                            raise HTTPException(status_code=415, detail="Image is too large.")
                    break
            else:
                raise HTTPException(status_code=502, detail="Image source redirected too many times.")

        if not image_bytes:
            raise HTTPException(status_code=502, detail="Image source returned an empty file.")

        # Keep the in-memory cache bounded on small instances.
        now = time.monotonic()
        expired = [key for key, item in _pixabay_image_cache.items()
                   if now - item[0] >= 86400]
        for key in expired:
            _pixabay_image_cache.pop(key, None)
        if len(_pixabay_image_cache) >= 100:
            oldest = min(_pixabay_image_cache, key=lambda key: _pixabay_image_cache[key][0])
            _pixabay_image_cache.pop(oldest, None)

        payload = bytes(image_bytes)
        _pixabay_image_cache[url] = (now, payload, content_type)
        return Response(
            content=payload,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=86400",
                     "Access-Control-Allow-Origin": "*"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Image proxy failed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not download selected image.") from exc


@app.post("/api/visuals")
async def generate_visuals(payload: VisualRequest):
    """Turn a script into visual scenes and find openly licensed Wikimedia Commons images."""
    if not GEMINI_API_KEYS:
        raise HTTPException(status_code=503, detail="AI visual planning is not configured. Add GEMINI_API_KEY to Render.")

    prompt = f"""Create a storyboard for a short social video using the supplied script.
Return ONLY valid JSON: an array of 5 to 10 objects with keys:
"narration": a short exact excerpt (max 24 Hindi words) from the script,
"visual": a concise description of what should appear on screen,
"search_query": 2-5 concrete English keywords suitable for finding the exact real subject shown in the scene.
IMPORTANT: If the narration or visual mentions a named TV show, brand, person, event, or place, include that exact proper name in search_query. Never replace a named show such as Bigg Boss with generic terms like television studio, microphone, or logo. For a scene mentioning "Bigg Boss logo", use "Bigg Boss logo" or "Bigg Boss 20 logo".
Do not invent events or imply a real person did something not stated. Prefer exact named entities and specific visible subjects over generic background objects.
Topic: {payload.topic}
Script:
{payload.script}
"""
    try:
        response, used_model, used_key_slot = await generate_with_key_fallback(prompt)
        raw = (response.text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            raise ValueError("AI did not return a valid storyboard.")
        scenes = json.loads(match.group(0))
        if not isinstance(scenes, list):
            raise ValueError("Storyboard format was invalid.")
    except Exception as exc:
        logger.warning("Visual storyboard generation failed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not create scene suggestions. Please retry.") from exc

    scenes = [x for x in scenes if isinstance(x, dict)][:10]
    if not scenes:
        raise HTTPException(status_code=502, detail="No visual scenes were generated.")

    async def find_commons_image(client: httpx.AsyncClient, query: str, broad: bool = False):
        """Search Commons with retry; broad mode removes restrictive bitmap syntax."""
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query if broad else f"filetype:bitmap {query}",
            "gsrnamespace": 6,
            "gsrlimit": 10,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": 2000,
            "format": "json",
        }
        for attempt in range(3):
            try:
                resp = await client.get(
                    "https://commons.wikimedia.org/w/api.php",
                    params=params,
                    headers={
                        "User-Agent": "TrendRadarAI/1.9 (https://github.com/sainiearn-lgtm/trendradar-ai1; contact: project-maintainer)"
                    },
                    timeout=20,
                )
                if resp.status_code in {429, 500, 502, 503, 504}:
                    if attempt < 2:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                resp.raise_for_status()
                pages = resp.json().get("query", {}).get("pages", {})
                found = []
                for page in pages.values():
                    info = (page.get("imageinfo") or [{}])[0]
                    meta = info.get("extmetadata") or {}
                    image_url = info.get("thumburl") or info.get("url")
                    page_url = info.get("descriptionurl") or ""
                    if not safe_http_url(image_url) or not safe_http_url(page_url):
                        continue

                    def meta_value(key):
                        value = meta.get(key, {})
                        return clean_text(value.get("value") if isinstance(value, dict) else value, 300)

                    found.append({
                        "title": clean_text(page.get("title", "").replace("File:", ""), 180),
                        "description": meta_value("ImageDescription") or meta_value("ObjectName"),
                        "image_url": image_url,
                        "source_url": page_url,
                        "license": meta_value("LicenseShortName") or "See source page",
                        "artist": meta_value("Artist") or "See source page",
                        "credit": meta_value("Credit"),
                    })
                return found
            except Exception as exc:
                logger.info("Wikimedia image search failed for query %r (attempt %s): %s", query, attempt + 1, exc)
                if attempt < 2:
                    await asyncio.sleep(0.7 * (attempt + 1))
        return []

    async def find_pixabay_images(client: httpx.AsyncClient, query: str):
        """Search Pixabay; keep the API key server-side and use attribution."""
        if not PIXABAY_API_KEY or not query:
            return []
        cache_key = query.casefold()
        cached_result = _pixabay_search_cache.get(cache_key)
        if cached_result and time.monotonic() - cached_result[0] < 86400:
            return cached_result[1]
        try:
            resp = await client.get(
                "https://pixabay.com/api/",
                params={"key": PIXABAY_API_KEY, "q": query, "image_type": "photo",
                        "orientation": "vertical", "safesearch": "true", "per_page": 20,
                        "min_width": 800, "order": "popular"},
                timeout=18,
            )
            if resp.status_code == 429:
                logger.warning("Pixabay rate limit reached for a scene search")
                return []
            resp.raise_for_status()
            found = []
            for hit in resp.json().get("hits", []):
                image_url = hit.get("largeImageURL") or hit.get("webformatURL")
                source_url = hit.get("pageURL") or "https://pixabay.com/"
                if not safe_http_url(image_url) or not safe_http_url(source_url):
                    continue
                creator = clean_text(hit.get("user") or "Pixabay contributor", 120)
                tags = clean_text(hit.get("tags") or "", 240)
                found.append({"title": tags or f"Photo by {creator}", "description": tags,
                    "image_url": "/api/pixabay-image?url=" + quote(image_url, safe=""), "source_url": source_url,
                    "license": "Pixabay Content License — verify source page",
                    "artist": creator, "credit": f"Image by {creator} / Pixabay",
                    "provider": "Pixabay", "width": hit.get("imageWidth") or hit.get("webformatWidth"),
                    "height": hit.get("imageHeight") or hit.get("webformatHeight")})
            _pixabay_search_cache[cache_key] = (time.monotonic(), found)
            return found
        except Exception as exc:
            logger.info("Pixabay image search failed for query %r: %s", query, exc)
            return []

    async def find_openverse_images(client: httpx.AsyncClient, query: str):
        """Search openly licensed images; anonymous access is rate-limited."""
        if not query:
            return []
        try:
            resp = await client.get("https://api.openverse.org/v1/images/",
                params={"q": query, "page_size": 20, "license_type": "commercial",
                        "mature": "false", "filter_dead": "true"},
                headers={"User-Agent": "TrendRadarAI/2.0 (image search)"}, timeout=18)
            if resp.status_code == 429:
                logger.warning("Openverse rate limit reached for a scene search")
                return []
            resp.raise_for_status()
            found = []
            for hit in resp.json().get("results", []):
                image_url = hit.get("thumbnail") or hit.get("url")
                source_url = hit.get("foreign_landing_url") or hit.get("detail_url")
                if not safe_http_url(image_url) or not safe_http_url(source_url):
                    continue
                creator = clean_text(hit.get("creator") or "Unknown creator", 120)
                title = clean_text(hit.get("title") or "Openly licensed image", 180)
                license_name = clean_text(hit.get("license") or "See source page", 40)
                license_version = clean_text(hit.get("license_version") or "", 20)
                found.append({"title": title, "description": clean_text(hit.get("tags") or "", 240),
                    "image_url": image_url, "source_url": source_url,
                    "license": (license_name + (" " + license_version if license_version else "")).strip(),
                    "artist": creator,
                    "credit": clean_text(hit.get("attribution") or f"{title} by {creator} ({license_name})", 300),
                    "provider": "Openverse", "width": hit.get("width"), "height": hit.get("height")})
            return found
        except Exception as exc:
            logger.info("Openverse image search failed for query %r: %s", query, exc)
            return []

    def core_topic_name() -> str:
        """Remove informational suffixes (net worth, age, biography) from the subject."""
        value = clean_text(payload.topic, 120)
        suffixes = [
            r"\s+(?:net\s+worth|income|salary|earnings|wealth)$",
            r"\s+(?:biography|bio|age|height|family|career|lifestyle|news|latest|update)$",
            r"\s+(?:net\s+worth|income|salary|earnings|wealth)\s+(?:in\s+202[4-9])$",
        ]
        for pattern in suffixes:
            value = re.sub(pattern, "", value, flags=re.I).strip()
        return value or clean_text(payload.topic, 120)

    def concise_terms(value: str, limit: int = 4) -> str:
        stop = {
            "the", "and", "for", "with", "from", "about", "latest", "news", "update",
            "fast", "zoom", "close", "up", "shot", "shots", "montage", "bouncing",
            "screen", "text", "graphic", "overlay", "engaging", "prompt", "asking",
            "audience", "comment", "subscribe", "animation", "on", "a", "an", "of",
            "her", "his", "their", "showing", "estimated", "bold", "question", "mark",
        }
        words = re.findall(r"[a-z0-9]+", clean_text(value, 240).casefold())
        picked = []
        for word in words:
            if word not in stop and (len(word) >= 3 or any(c.isdigit() for c in word)) and word not in picked:
                picked.append(word)
            if len(picked) >= limit:
                break
        return " ".join(picked)

    def is_person_topic() -> bool:
        # Names are usually two or three words; the AI's scene terms identify portrait scenes.
        words = re.findall(r"[a-z]+", core_topic_name().casefold())
        return len(words) >= 2 and not any(x in core_topic_name().casefold() for x in ("iphone", "samsung", "pixel", "car", "phone"))

    def make_search_queries(scene: dict) -> list[str]:
        """Search the actual scene subject, not the full informational topic on every request."""
        core = core_topic_name()
        ai_query = clean_text(scene.get("search_query", ""), 160)
        visual = clean_text(scene.get("visual", ""), 240)
        combined = (ai_query + " " + visual + " " + clean_text(scene.get("narration", ""), 240)).casefold()
        # Preserve named show identity: generic AI visual terms must not override it.
        script_context = (clean_text(payload.topic, 160) + " " + clean_text(payload.script, 5000)).casefold()
        has_bigg_boss = bool(re.search(r"bigg\\s*boss|बिग\\s*बॉस", script_context))
        bigg_boss_brand_scene = has_bigg_boss and any(term in combined for term in (
            "logo", "headline", "studio", "intro", "show", "house", "contestant", "television"
        ))
        person_scene = any(term in combined for term in (
            "portrait", "glamour", "glam", "actress", "actor", "celebrity", "headshot",
            "red carpet", "photoshoot", "fashion", "movie still", "film star"
        ))
        queries = []
        if bigg_boss_brand_scene:
            queries.extend(["Bigg Boss 20 logo", "Bigg Boss reality show India"])
        if person_scene and is_person_topic():
            queries.extend([core, concise_terms(core + " portrait", 4)])
        else:
            scene_terms = concise_terms(ai_query, 4) or concise_terms(visual, 4)
            if scene_terms:
                queries.append(scene_terms)
            # Keep the subject as a fallback, not as a prefix that makes every search too strict.
            if is_person_topic() and any(x in combined for x in ("movie", "film", "studio", "commercial", "shoot")):
                queries.append(core)
        if not queries:
            queries.append(core)
        unique = []
        seen = set()
        for query in queries:
            query = clean_text(query, 100)
            if not query:
                continue

            # India-focused discovery: search Indian context first, then retain
            # the original global query as a fallback. Do not alter named-person
            # portrait searches, where adding a country can return the wrong person.
            candidates = [query]
            if not (is_person_topic() and person_scene):
                candidates = [f"{query} India", query]

            for candidate in candidates:
                candidate = clean_text(candidate, 120)
                key = candidate.casefold()
                if candidate and key not in seen:
                    seen.add(key)
                    unique.append(candidate)
        return unique[:6]

    def is_scene_relevant(item: dict, scene: dict, query: str) -> bool:
        """Reject loosely related Commons hits; prioritize actual title/entity matches."""
        generic = {
            "photo", "image", "picture", "portrait", "close", "shot", "wide", "view",
            "background", "scene", "visual", "video", "news", "latest", "update",
            "showing", "display", "people", "person", "woman", "man", "beautiful",
            "famous", "indian", "india", "bollywood", "celebrity", "actor", "actress",
            "the", "and", "for", "with", "from", "about", "into", "near", "over",
        }
        terms = [w for w in re.findall(r"[a-z0-9]+", query.casefold()) if len(w) >= 3 and w not in generic]
        if not terms:
            return False
        title = re.sub(r"[^a-z0-9]+", " ", str(item.get("title", "")).casefold())
        desc = re.sub(r"[^a-z0-9]+", " ", str(item.get("description", "")).casefold())
        title_hits = sum(bool(re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", title)) for w in terms)
        desc_hits = sum(bool(re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", desc)) for w in terms)
        # A named person / product with two or more distinctive words must match
        # both words somewhere in the title/description; generic single-word hits
        # are not enough to label a result as a correct image.
        total_hits = len(set(
            w for w in terms
            if re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", title + " " + desc)
        ))
        required = 1 if len(terms) == 1 else 2
        if len(terms) >= 2 and len(terms) <= 3 and title_hits == 0:
            return False
        return total_hits >= required and (title_hits >= 1 or desc_hits >= 2)

    async def search_scene(client: httpx.AsyncClient, scene: dict, semaphore: asyncio.Semaphore):
        """Search Pixabay/Openverse for generic B-roll and Commons for named subjects."""
        async with semaphore:
            all_images = []
            seen_urls = set()
            queries = make_search_queries(scene)
            visual_terms = concise_terms(scene.get("visual", ""), 3)
            if visual_terms and visual_terms.casefold() not in {q.casefold() for q in queries}:
                queries.append(visual_terms)

            combined = (clean_text(scene.get("search_query", ""), 160) + " " + clean_text(scene.get("visual", ""), 240)).casefold()
            person_scene = any(term in combined for term in (
                "portrait", "glamour", "glam", "actress", "actor", "celebrity", "headshot",
                "red carpet", "photoshoot", "fashion", "movie still", "film star"
            ))
            named_subject = is_person_topic() and person_scene

            # India-localized queries are tried before global fallbacks for
            # generic scenes, independent of the server's deployment region.
            # Never use stock search as proof of a named celebrity's identity.
            if PIXABAY_API_KEY and not named_subject:
                for query in queries[:3]:
                    results = await find_pixabay_images(client, query)
                    for item in results:
                        url = item.get("image_url")
                        if not url or url in seen_urls:
                            continue
                        metadata = (item.get("title", "") + " " + item.get("description", "")).casefold()
                        terms = [
                            w for w in re.findall(r"[a-z0-9]+", query.casefold())
                            if len(w) >= 4 and w not in {"india", "indian"}
                        ]
                        # Reject empty or unrelated metadata instead of allowing
                        # untagged results to slip through the relevance filter.
                        if terms and not any(
                            re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", metadata)
                            for w in terms
                        ):
                            continue
                        width, height = item.get("width"), item.get("height")
                        if width and height and (int(width) < 500 or int(height) < 600):
                            continue
                        seen_urls.add(url)
                        all_images.append(item)
                        if len(all_images) >= 5:
                            break
                    logger.info("Pixabay scene query: scene=%s query=%r accepted=%s", scene.get("scene", "?"), query, len(all_images))
                    if len(all_images) >= 5:
                        break

            # Openverse supplies another pool of openly licensed generic visuals.
            if len(all_images) < 5 and not named_subject:
                for query in queries[:3]:
                    results = await find_openverse_images(client, query)
                    accepted_before = len(all_images)
                    for item in results:
                        url = item.get("image_url")
                        if not url or url in seen_urls:
                            continue
                        if is_scene_relevant(item, scene, query):
                            seen_urls.add(url)
                            all_images.append(item)
                        if len(all_images) >= 5:
                            break
                    logger.info("Openverse scene query: scene=%s query=%r raw=%s accepted=%s", scene.get("scene", "?"), query, len(results), len(all_images) - accepted_before)
                    if len(all_images) >= 5:
                        break

            # Commons is used for named people and as a fallback for generic visuals.
            if len(all_images) < 5:
                for broad in (True, False):
                    for query in queries[:5]:
                        logger.info("Wikimedia scene search: scene=%s broad=%s query=%r", scene.get("scene", "?"), broad, query)
                        results = await find_commons_image(client, query, broad=broad)
                        accepted_before = len(all_images)
                        for item in results:
                            url = item.get("image_url")
                            if not url or url in seen_urls:
                                continue
                            if is_scene_relevant(item, scene, query):
                                item["provider"] = "Wikimedia Commons"
                                seen_urls.add(url)
                                all_images.append(item)
                            if len(all_images) >= 5:
                                break
                        logger.info("Wikimedia scene query result: scene=%s query=%r raw=%s accepted=%s", scene.get("scene", "?"), query, len(results), len(all_images) - accepted_before)
                        if len(all_images) >= 5:
                            break
                    if len(all_images) >= 5:
                        break
            logger.info("Image search complete: scene=%s images=%s", scene.get("scene", "?"), len(all_images))
            return all_images[:5]

    async with httpx.AsyncClient(follow_redirects=True) as client:
        semaphore = asyncio.Semaphore(2)
        image_results = await asyncio.gather(
            *(search_scene(client, scene, semaphore) for scene in scenes),
            return_exceptions=True,
        )

    output = []
    for index, (scene, results) in enumerate(zip(scenes, image_results), start=1):
        images = results if isinstance(results, list) else []
        if not images:
            logger.warning(
                "No images found for scene %s; tried focused queries: %s",
                index, make_search_queries(scene)
            )
        output.append({
            "scene": index,
            "narration": clean_text(scene.get("narration"), 500),
            "visual": clean_text(scene.get("visual"), 300),
            "search_query": clean_text(scene.get("search_query"), 160),
            "images": images[:5],
            "image_status": "found" if images else "not_found",
        })
    return {
        "topic": payload.topic,
        "scenes": output,
        "image_source": "Pixabay + Openverse + Wikimedia Commons",
        "notice": "Pixabay, Openverse and Wikimedia Commons images include source and creator details where available. Check each source page and license before publishing. Pixabay results require PIXABAY_API_KEY; Openverse is rate-limited and may return no results.",
        "found_scenes": sum(1 for scene in output if scene["images"]),
        "missing_scenes": [scene["scene"] for scene in output if not scene["images"]],
    }

@app.post("/api/ai-image")
async def generate_ai_image(payload: dict):
    """Generate one vertical image through Pollinations; the secret stays server-side."""
    prompt = clean_text(payload.get("prompt"), 700)

    if len(prompt) < 5:
        raise HTTPException(status_code=400, detail="Image prompt is too short.")

    if not POLLINATIONS_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="AI image generation is not configured. Add POLLINATIONS_API_KEY in Render."
        )

    url = f"https://gen.pollinations.ai/image/{quote(prompt, safe='')}"
    params = {
        "model": "black-forest-labs/flux.1-schnell",
        "width": 576,
        "height": 1024,
    }
    headers = {
        "Authorization": f"Bearer {POLLINATIONS_API_KEY}",
        "Accept": "image/*",
    }

    try:
        timeout = httpx.Timeout(120.0, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, params=params, headers=headers)

        if response.status_code in {401, 403}:
            logger.warning("Pollinations rejected the configured API key (HTTP %s)", response.status_code)
            raise HTTPException(
                status_code=502,
                detail="Pollinations API key was rejected. Check the key and its Image permission in Render."
            )
        if response.status_code == 402:
            raise HTTPException(
                status_code=402,
                detail="Pollinations balance or key budget is insufficient. Check your Pollinations account."
            )
        if response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="Pollinations is rate-limiting image generation. Please wait and retry."
            )
        if response.status_code >= 500:
            logger.warning("Pollinations image service returned HTTP %s", response.status_code)
            raise HTTPException(
                status_code=502,
                detail="Pollinations image service is temporarily busy. Please retry shortly."
            )

        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if not content_type.startswith("image/"):
            logger.warning("Pollinations returned a non-image response (content-type: %s)", content_type or "missing")
            raise HTTPException(
                status_code=502,
                detail="Pollinations did not return an image. Please retry."
            )
        if not response.content or len(response.content) > 15 * 1024 * 1024:
            raise HTTPException(
                status_code=502,
                detail="Generated image was empty or too large. Please retry."
            )

        encoded = base64.b64encode(response.content).decode("ascii")
        image_data_url = f"data:{content_type};base64,{encoded}"
        return {
            "status": "done",
            "image_url": image_data_url,
            "provider": "Pollinations AI",
        }

    except HTTPException:
        raise
    except httpx.TimeoutException as exc:
        logger.warning("Pollinations image generation timed out: %s", exc)
        raise HTTPException(
            status_code=504,
            detail="AI image generation is taking too long. Please retry."
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.warning("Pollinations image request failed with HTTP %s", exc.response.status_code)
        raise HTTPException(
            status_code=502,
            detail="Pollinations image request failed. Please retry."
        ) from exc
    except Exception as exc:
        logger.exception("Pollinations image generation failed")
        raise HTTPException(
            status_code=502,
            detail="AI image service is temporarily unavailable."
        ) from exc


@app.get("/api/voice/voices")
async def list_voice_options():
    return {
        "provider": "Indic Tiny TTS Hindi (ONNX Runtime, CPU)",
        "voices": [
            {"language": "Hindi", "gender": gender, "voice": f"Indic Tiny TTS {gender.title()}"}
            for gender in ("male", "female")
        ],
        "notice": "Hindi Devanagari only. The model downloads on first use; first generation may take longer."
    }


@app.post("/api/voice")
async def generate_voice(payload: VoiceRequest):
    try:
        if not any("\u0900" <= char <= "\u097f" for char in payload.text):
            raise HTTPException(
                status_code=422,
                detail="Indic Tiny TTS needs Hindi text in Devanagari. Select Hindi and use text such as: नमस्ते दोस्तों।",
            )
        model = await _get_indic_tts()
        # Set the shared instance before handing synthesis to the worker thread.
        global _indic_tts_instance
        _indic_tts_instance = model
        audio_data = await asyncio.to_thread(
            _render_indic_audio,
            payload.text,
            payload.gender,
            payload.speed,
            payload.pitch,
            payload.volume,
        )
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="trendradar-voice.wav"',
                "Cache-Control": "no-store",
                "X-Voice-Name": f"Indic-Tiny-TTS-{payload.gender}",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Indic Tiny TTS voice generation failed (gender=%s, speed=%s, pitch=%s, volume=%s)", payload.gender, payload.speed, payload.pitch, payload.volume)
        raise HTTPException(
            status_code=502,
            detail="Indic Tiny TTS generation failed. Check Render logs; the first request may be downloading the model.",
        ) from exc

