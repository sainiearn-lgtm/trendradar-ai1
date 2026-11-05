from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("trendradar")

APP_VERSION = "1.1.0"

# Read both keys. Empty keys are ignored; never log key values.
GEMINI_API_KEYS = [
    value.strip()
    for value in (
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("GEMINI_API_KEY_2", ""),
    )
    if value and value.strip()
]
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
# Optional: set this in Render only if you have confirmed that the model is
# available to your Google AI Studio project. Otherwise leave it blank.
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "900"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "12"))
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "90"))
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
        value = clean_text(value, 220)
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
        ))
    )


async def _generate_with_one_key(genai, api_key: str, model: str, prompt: str):
    client = genai.Client(api_key=api_key)
    return await asyncio.wait_for(
        asyncio.to_thread(client.models.generate_content, model=model, contents=prompt),
        timeout=GEMINI_TIMEOUT,
    )


async def generate_with_key_fallback(prompt: str):
    """Try each configured key; if configured, try a second model as well."""
    from google import genai

    models = [GEMINI_MODEL]
    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
        models.append(GEMINI_FALLBACK_MODEL)

    last_error: Exception | None = None
    # Try each key once per model. A 503 on one key will not prevent trying key 2.
    for model in models:
        for key_index, api_key in enumerate(GEMINI_API_KEYS, start=1):
            try:
                logger.info("Trying Gemini model %s with configured key slot %s", model, key_index)
                response = await _generate_with_one_key(genai, api_key, model, prompt)
                return response, model, key_index
            except asyncio.TimeoutError as exc:
                last_error = exc
                logger.warning("Gemini timed out for model %s, key slot %s", model, key_index)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Gemini attempt failed for model %s, key slot %s: %s",
                    model, key_index, exc,
                )
                # Invalid key/model errors are not helped by repeated attempts.
                # Continue to the next configured key/model so a working one can succeed.
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
Return polished creator-ready content with these exact sections:
1. CATCHY TITLE
2. OPENING HOOK
3. COMPLETE VIDEO SCRIPT (include timestamps or scene labels)
4. SCENE-BY-SCENE VISUAL SUGGESTIONS
5. THUMBNAIL TEXT (3 options)
6. VIDEO DESCRIPTION
7. HASHTAGS (8-12 relevant tags)
8. CALL TO ACTION
Do not invent statistics, quotes, breaking-news claims, or factual specifics. If a claim needs verification, mark it [VERIFY]. Avoid robotic phrasing and filler. Output plain text with clear headings."""
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
