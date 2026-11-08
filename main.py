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
from urllib.parse import urlparse, quote

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("trendradar")

APP_VERSION = "1.9.3"

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
            rows.append({
                "title": title, "summary": desc or f"Google Trends search interest is rising for {title}.",
                "source": "Google Trends", "source_url": link,
                "published_at": (pub or datetime.now(timezone.utc)).isoformat(),
                "category": "Search trend", "mentions": 1, "popularity": 8,
            })
    return rows


async def fetch_gdelt(client: httpx.AsyncClient, region: str) -> list[dict]:
    country = REGIONS[region]["gdelt"]
    params = {
        "query": f'"{country}"', "mode": "ArtList", "format": "json",
        "sort": "HybridRel", "maxrecords": "50", "timespan": "24h",
    }
    response = await client.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    rows = []
    for article in data.get("articles", []) or []:
        title = clean_text(article.get("title"), 180)
        url = article.get("url", "")
        if not title or not safe_http_url(url):
            continue
        published = parse_date(article.get("seendate"))
        domain = clean_text(article.get("domain") or urlparse(url).netloc, 100)
        rows.append({
            "title": title,
            "summary": clean_text(article.get("socialimage") or article.get("sourcecountry") or f"Recent news coverage related to {country}.", 450),
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
        })
    output.sort(key=lambda x: (x["trend_score"], x["published_at"] or ""), reverse=True)
    return output[:MAX_TRENDS]


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


class ScriptRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=5000)
    language: str = Field(default="English")
    format: str = Field(default="Shorts (60 seconds)")

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
    prompt = f"""You are an expert YouTube and Instagram video scriptwriter. Write naturally and engagingly in {payload.language}.
Topic: {payload.topic}
Format: {payload.format}
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

    # Pitch processing is loaded only when needed. Kaiser resampling avoids
    # the optional soxr backend and is lighter on small Render instances.
    if pitch:
        import librosa
        audio = librosa.effects.pitch_shift(
            y=audio,
            sr=24000,
            n_steps=float(pitch),
            bins_per_octave=12,
            # Use librosa's SciPy-backed Kaiser resampler. soxr_hq requires
            # the optional python-soxr package, which is not installed on
            # every Render build and caused non-zero pitch requests to fail.
            res_type="kaiser_fast",
        )
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
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
    """Fetch approved image hosts server-side and cache image bytes for 24 hours."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    allowed_hosts = (
        "pixabay.com", "wikimedia.org", "flickr.com", "staticflickr.com",
        "wordpress.com", "wp.com", "creativecommons.org", "openverse.org",
        "unsplash.com",
    )
    allowed = any(host == domain or host.endswith("." + domain) for domain in allowed_hosts)
    if parsed.scheme != "https" or not allowed:
        raise HTTPException(status_code=400, detail="Unsupported image host.")

    cached_image = _pixabay_image_cache.get(url)
    if cached_image and time.monotonic() - cached_image[0] < 86400:
        return Response(content=cached_image[1], media_type=cached_image[2],
                        headers={"Cache-Control": "public, max-age=86400", "Access-Control-Allow-Origin": "*"})

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 TrendRadarAI"})
            resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0]
        if not content_type.startswith("image/") or len(resp.content) > 15 * 1024 * 1024:
            raise HTTPException(status_code=415, detail="Invalid or oversized image.")
        _pixabay_image_cache[url] = (time.monotonic(), resp.content, content_type)
        return Response(content=resp.content, media_type=content_type,
                        headers={"Cache-Control": "public, max-age=86400", "Access-Control-Allow-Origin": "*"})
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
"search_query": 2-5 concrete English keywords suitable for finding a real photograph on Wikimedia Commons.
Do not invent events or imply a real person did something not stated. Prefer specific objects, places, people, or concepts over abstract words.
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
        combined = (ai_query + " " + visual).casefold()
        person_scene = any(term in combined for term in (
            "portrait", "glamour", "glam", "actress", "actor", "celebrity", "headshot",
            "red carpet", "photoshoot", "fashion", "movie still", "film star"
        ))
        queries = []
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
            key = query.casefold()
            if query and key not in seen:
                seen.add(key)
                unique.append(query)
        return unique[:4]

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

            # Pixabay is the primary stock-photo source for generic B-roll.
            # Never use stock search as proof of a named celebrity's identity.
            if PIXABAY_API_KEY and not named_subject:
                for query in queries[:3]:
                    results = await find_pixabay_images(client, query)
                    for item in results:
                        url = item.get("image_url")
                        if not url or url in seen_urls:
                            continue
                        metadata = (item.get("title", "") + " " + item.get("description", "")).casefold()
                        terms = [w for w in re.findall(r"[a-z0-9]+", query.casefold()) if len(w) >= 4]
                        if metadata and terms and not any(w in metadata for w in terms):
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

