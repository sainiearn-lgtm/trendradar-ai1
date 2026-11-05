from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import time
from collections import Counter
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

APP_VERSION = "1.0.1"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
# Keep GEMINI_MODEL in Render Environment Variables if already configured.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "900"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "12"))
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "90"))
GEMINI_MAX_RETRIES = max(1, min(4, int(os.getenv("GEMINI_MAX_RETRIES", "3"))))
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
    text = html.unescape(re.sub(r"<[^>]*>", " ", str(value or "")))
    return re.sub(r"\s+", " ", text).strip()[:limit]


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
                "source": "Google Trends", "source_url": link, "published_at": (pub or datetime.now(timezone.utc)).isoformat(),
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
            "source": f"GDELT · {domain}",
            "source_url": url,
            "published_at": (published or datetime.now(timezone.utc)).isoformat(),
            "category": clean_text(article.get("tone"), 40) or "News",
            "mentions": 1,
            "popularity": 2,
        })
    return rows


async def fetch_youtube(client: httpx.AsyncClient, region: str) -> list[dict]:
    if not YOUTUBE_API_KEY:
        return []
    params = {"part": "snippet,statistics", "chart": "mostPopular", "regionCode": region, "maxResults": 25, "key": YOUTUBE_API_KEY}
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
        mentions = len(group)
        pub = parse_date(best.get("published_at"))
        output.append({
            "id": str(abs(hash(key))),
            "title": best["title"],
            "summary": best["summary"],
            "source": " + ".join(sorted(sources)),
            "source_url": best["source_url"],
            "region": region,
            "category": best.get("category") or "General",
            "trend_score": score_item(pub, mentions, len(sources), max(x.get("popularity", 0) for x in group)),
            "published_at": best.get("published_at"),
            "mentions": mentions,
        })
    output.sort(key=lambda x: (x["trend_score"], x["published_at"] or ""), reverse=True)
    return output[:MAX_TRENDS]


@app.get("/")
async def root():
    return {"name": "TrendRadar AI API", "status": "online", "version": APP_VERSION, "docs": "/docs"}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "gemini_configured": bool(GEMINI_API_KEY), "youtube_configured": bool(YOUTUBE_API_KEY), "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/trends")
async def trends(region: str = "IN", q: str | None = None):
    region = region.upper()
    if region not in REGIONS:
        raise HTTPException(status_code=422, detail=f"Unsupported region. Choose: {', '.join(REGIONS)}")
    cache_key = f"trends:{region}"
    rows = cached(cache_key)
    source_status = {"Google Trends": "not_checked", "GDELT": "not_checked", "YouTube": "not_configured" if not YOUTUBE_API_KEY else "not_checked"}
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
        source_status = {"Google Trends": "cached", "GDELT": "cached", "YouTube": "cached" if YOUTUBE_API_KEY else "not_configured"}
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
    topic: str = Field(min_length=2, max_length=220)
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
        allowed = {"Hindi", "Hinglish", "English"}
        if value not in allowed:
            raise ValueError("Language must be Hindi, Hinglish, or English.")
        return value

    @field_validator("format")
    @classmethod
    def validate_format(cls, value: str) -> str:
        allowed = {"Shorts (60 seconds)", "Long video (5-10 minutes)"}
        if value not in allowed:
            raise ValueError("Unsupported video format.")
        return value


def is_temporary_gemini_error(exc: Exception) -> bool:
    """Identify errors that are usually safe to retry."""
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


async def call_gemini_with_retry(client, prompt: str) -> Any:
    last_error: Exception | None = None

    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=GEMINI_MODEL,
                    contents=prompt,
                ),
                timeout=GEMINI_TIMEOUT,
            )
        except asyncio.TimeoutError as exc:
            last_error = exc
            logger.warning("Gemini request timed out (attempt %s/%s)", attempt + 1, GEMINI_MAX_RETRIES)
            # A timed-out worker thread may still finish in the background.
            # Avoid launching multiple long-running calls after a timeout.
            break
        except Exception as exc:
            last_error = exc
            if not is_temporary_gemini_error(exc) or attempt + 1 >= GEMINI_MAX_RETRIES:
                raise
            delay = min(2 ** attempt, 8)
            logger.warning(
                "Temporary Gemini error (attempt %s/%s): %s. Retrying in %ss.",
                attempt + 1, GEMINI_MAX_RETRIES, exc, delay,
            )
            await asyncio.sleep(delay)

    if isinstance(last_error, asyncio.TimeoutError):
        raise last_error
    if last_error:
        raise last_error
    raise RuntimeError("Gemini request failed without returning a response.")


@app.post("/api/script")
async def generate_script(payload: ScriptRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="AI script generation is not configured yet. Add GEMINI_API_KEY to the backend environment variables.")
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
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = await call_gemini_with_retry(client, prompt)
        script = (response.text or "").strip()
        if not script:
            raise RuntimeError("Gemini returned an empty response.")
        return {
            "script": script,
            "topic": payload.topic,
            "language": payload.language,
            "format": payload.format,
            "model": GEMINI_MODEL,
        }
    except asyncio.TimeoutError:
        logger.exception("Gemini generation timed out")
        raise HTTPException(
            status_code=504,
            detail="Gemini is taking too long right now. Please try again in a minute.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gemini generation failed")
        if is_temporary_gemini_error(exc):
            raise HTTPException(
                status_code=503,
                detail="Gemini is temporarily busy. Please wait 30 seconds and try again.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="AI generation failed. Check the Gemini API key/model configuration and try again.",
        ) from exc


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled API error at %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "An unexpected server error occurred."})
