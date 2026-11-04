"""TrendRadar AI - FastAPI backend. Serves the API and the frontend from one service."""
import asyncio, html, json, math, os, re, time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# code: (name, Google News hl, Google News ceid, GDELT country code)
REGIONS = {
    "IN": ("India", "en-IN", "IN:en", "IN"),
    "US": ("United States", "en-US", "US:en", "US"),
    "GB": ("United Kingdom", "en-GB", "GB:en", "UK"),
    "CA": ("Canada", "en-CA", "CA:en", "CA"),
    "AU": ("Australia", "en-AU", "AU:en", "AS"),
    "DE": ("Germany", "de", "DE:de", "GM"),
}
CATEGORIES = {
    "Technology": "ai tech apple google microsoft app software chip openai phone",
    "Business": "market stock economy bank company price inflation trade tax",
    "Sports": "cricket football match league cup nba nfl goal win team ipl",
    "Entertainment": "film movie actor singer album show series trailer netflix celebrity",
    "Politics": "election minister government president parliament vote policy",
    "Health": "health hospital virus vaccine disease doctor medical",
    "Science": "nasa space study research scientists climate",
}
UA = {"User-Agent": "Mozilla/5.0 TrendRadarAI/1.0"}
CACHE: dict = {}
TTL = 600  # seconds

app = FastAPI(title="TrendRadar AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def tag(el):
    return el.tag.split("}")[-1]


def clean(text, limit=240):
    text = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def categorize(text):
    words = set(re.findall(r"[a-z]+", text.lower()))
    best, hits = "General", 0
    for cat, kw in CATEGORIES.items():
        n = len(words & set(kw.split()))
        if n > hits:
            best, hits = cat, n
    return best


def parse_date(s):
    try:
        d = parsedate_to_datetime(s) if "," in s else datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        try:  # GDELT format 20240101T120000Z
            d = datetime.strptime(s, "%Y%m%dT%H%M%SZ")
        except Exception:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def parse_traffic(s):
    m = re.match(r"([\d,.]+)\s*([KkMm]?)", (s or "").strip())
    if not m:
        return 0
    n = float(m.group(1).replace(",", ""))
    return int(n * {"k": 1e3, "m": 1e6}.get(m.group(2).lower(), 1))


# ---------------- Sources ----------------
async def google_trends(client, code):
    r = await client.get(f"https://trends.google.com/trending/rss?geo={code}", headers=UA)
    r.raise_for_status()
    out = []
    for item in ET.fromstring(r.content).iter("item"):
        d = {tag(c): (c.text or "") for c in item}
        news = [c for c in item if tag(c) == "news_item"]
        n = {tag(c): (c.text or "") for c in news[0]} if news else {}
        pub = parse_date(d.get("pubDate", ""))
        out.append(dict(
            title=clean(d.get("title"), 120),
            summary=clean(n.get("news_item_title") or "Searches for this topic are rising on Google."),
            source="Google Trends", url=n.get("news_item_url") or f"https://trends.google.com/trending?geo={code}",
            published=pub.isoformat() if pub else None, traffic=parse_traffic(d.get("approx_traffic")), signal=0))
    return out


async def google_news(client, code, q):
    _, hl, ceid, _ = REGIONS[code]
    base = "https://news.google.com/rss"
    url = f"{base}/search?q={quote(q)}&hl={hl}&gl={code}&ceid={ceid}" if q else f"{base}?hl={hl}&gl={code}&ceid={ceid}"
    r = await client.get(url, headers=UA)
    r.raise_for_status()
    out = []
    for item in list(ET.fromstring(r.content).iter("item"))[:30]:
        d = {tag(c): (c.text or "") for c in item}
        title, pub_name = d.get("title", ""), d.get("source", "")
        if pub_name and title.endswith(" - " + pub_name):
            title = title[: -len(pub_name) - 3]
        pub = parse_date(d.get("pubDate", ""))
        out.append(dict(
            title=clean(title, 140), summary=clean(d.get("description")) if clean(d.get("description")) != clean(title) else f"Top story reported by {pub_name or 'Google News'}.",
            source=f"Google News" + (f" · {pub_name}" if pub_name else ""), url=d.get("link", ""),
            published=pub.isoformat() if pub else None, traffic=0, signal=0))
    return out


async def gdelt(client, code, q):
    country = REGIONS[code][3]
    query = f"{q} sourcecountry:{country}" if q else f"(news OR update OR latest) sourcecountry:{country}"
    r = await client.get("https://api.gdeltproject.org/api/v2/doc/doc", headers=UA, params=dict(
        query=query, mode="artlist", format="json", maxrecords=30, sort="datedesc", timespan="1d"))
    r.raise_for_status()
    out = []
    for a in r.json().get("articles", []):
        pub = parse_date(a.get("seendate", ""))
        out.append(dict(
            title=clean(a.get("title"), 140), summary=f"Reported by {a.get('domain', 'a news site')} ({a.get('language', 'n/a')}).",
            source="GDELT News", url=a.get("url", ""), published=pub.isoformat() if pub else None, traffic=0, signal=0))
    return out


async def youtube(client, code):
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("YOUTUBE_API_KEY not set (optional source)")
    r = await client.get("https://www.googleapis.com/youtube/v3/videos", params=dict(
        part="snippet,statistics", chart="mostPopular", regionCode=code, maxResults=25, key=key))
    r.raise_for_status()
    out = []
    for v in r.json().get("items", []):
        s = v["snippet"]
        out.append(dict(
            title=clean(s.get("title"), 140), summary=clean(s.get("description")) or f"Popular video by {s.get('channelTitle')}.",
            source="YouTube Trending", url=f"https://www.youtube.com/watch?v={v['id']}", published=s.get("publishedAt"),
            traffic=int(v.get("statistics", {}).get("viewCount", 0)), signal=0))
    return out


# ---------------- Scoring ----------------
STOP = set("the a an and or of to in on for with at by from is are was be this that it as after new says say how why what who will vs".split())


def tokens(t):
    return {w for w in re.findall(r"\w+", t.lower()) if len(w) > 2 and w not in STOP}


def score_all(items):
    """Score 0-100 = source signal (0-45) + recency (0-30) + repeated mentions across sources (0-25)."""
    now = datetime.now(timezone.utc)
    toks = [tokens(i["title"]) for i in items]
    for idx, it in enumerate(items):
        # Source signal
        if it["source"] == "Google Trends":
            sig = 20 + min(25, math.log10(max(it["traffic"], 1)) * 4.5)
        elif it["source"] == "YouTube Trending":
            sig = 15 + min(30, math.log10(max(it["traffic"], 1)) * 4)
        else:
            sig = 12
        # Recency: full points when brand new, fading to 0 over 48h
        rec = 0
        if it["published"]:
            age_h = max((now - datetime.fromisoformat(it["published"])).total_seconds() / 3600, 0)
            rec = 30 * max(0, 1 - age_h / 48)
        # Repetition: how many items from OTHER sources talk about the same thing
        mentions, srcs = 0, set()
        for j, other in enumerate(items):
            if j == idx or other["source"].split(" ·")[0] == it["source"].split(" ·")[0] and other["source"] == it["source"]:
                continue
            a, b = toks[idx], toks[j]
            if a and b and len(a & b) / min(len(a), len(b)) >= 0.5 and len(a & b) >= 2:
                mentions += 1
                srcs.add(other["source"].split(" ·")[0])
        rep = min(25, mentions * 5 + len(srcs) * 4)
        it["mentions"] = mentions
        it["score"] = int(round(min(100, sig + rec + rep)))
        it.pop("traffic", None); it.pop("signal", None)
    return items


def dedupe(items):
    seen, out = [], []
    for it in sorted(items, key=lambda x: x["score"], reverse=True):
        t = tokens(it["title"])
        if any(t and s and len(t & s) / max(len(t | s), 1) > 0.7 for s in seen):
            continue
        seen.append(t); out.append(it)
    return out


# ---------------- Routes ----------------
@app.get("/api/health")
def health():
    return {"ok": True, "gemini": bool(os.getenv("GEMINI_API_KEY")), "youtube": bool(os.getenv("YOUTUBE_API_KEY"))}


@app.get("/api/trends")
async def trends(region: str = "IN", q: str = Query("", max_length=80)):
    region = region.upper()
    if region not in REGIONS:
        raise HTTPException(400, "Unsupported region")
    q = q.strip()
    ck = (region, q.lower())
    if ck in CACHE and time.time() - CACHE[ck][0] < TTL:
        return CACHE[ck][1]

    jobs = {"Google Trends": google_trends, "Google News": google_news, "GDELT": gdelt, "YouTube": youtube}
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
        results = await asyncio.gather(
            google_trends(c, region), google_news(c, region, q), gdelt(c, region, q), youtube(c, region),
            return_exceptions=True)
    names = list(jobs)
    items, status = [], []
    for name, res in zip(names, results):
        if isinstance(res, Exception):
            status.append({"source": name, "ok": False, "error": str(res)[:120] or res.__class__.__name__})
        else:
            if q and name in ("Google Trends", "YouTube"):  # these feeds can't search, so filter locally
                terms = tokens(q)
                res = [i for i in res if terms & tokens(i["title"] + " " + i["summary"])]
            status.append({"source": name, "ok": True, "count": len(res)})
            items += res
    for it in items:
        it["region"] = REGIONS[region][0]
        it["category"] = categorize(it["title"] + " " + it["summary"])
    items = dedupe(score_all(items))[:40]
    for rank, it in enumerate(items, 1):
        it["rank"] = rank
    payload = {
        "region": REGIONS[region][0], "query": q, "updated": datetime.now(timezone.utc).isoformat(),
        "sources": status, "trends": items,
        "note": "Trend scores are estimates built from source signals, recency and repeated mentions. They are not official social platform virality metrics.",
    }
    if items:
        CACHE[ck] = (time.time(), payload)
    return payload


class ScriptRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=300)
    context: str = Field("", max_length=600)
    language: str = "English"
    format: str = "shorts"


LANG = {"Hindi": "Hindi written in Devanagari script", "Hinglish": "Hinglish (Hindi written in Roman/English letters, casual Indian creator style)", "English": "clear, energetic English"}


@app.post("/api/script")
async def script(req: ScriptRequest):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(503, "GEMINI_API_KEY is not set on the server.")
    if req.language not in LANG or req.format not in ("shorts", "long"):
        raise HTTPException(400, "Invalid language or format")
    length = ("a YouTube Short of about 60 seconds (roughly 130-160 spoken words, 5-7 fast scenes)" if req.format == "shorts"
              else "a long YouTube video of 5-10 minutes (roughly 900-1400 spoken words, 8-12 scenes with intro, body sections and outro)")
    prompt = f"""You are an expert YouTube and Instagram scriptwriter.
Write {length} about: "{req.topic}".
Background from news feeds (use only as context, do not invent facts or statistics): "{req.context}".
Language: {LANG[req.language]}.
Return JSON with exactly these keys:
title (catchy, under 70 chars), hook (powerful first 3 seconds), script (full spoken script, plain text with line breaks),
scenes (array of objects: time, visual, voiceover), thumbnail_text (max 5 words), description (SEO video description, 2-3 short paragraphs),
hashtags (array of 8-12 strings starting with #), cta (call to action line).
Keep claims cautious and factual. Never present uncertain details as confirmed."""
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.8}}
    last = "Unknown error"
    async with httpx.AsyncClient(timeout=90) as c:
        for attempt in range(2):
            try:
                r = await c.post(url, json=body, headers={"x-goog-api-key": key})
                if r.status_code == 200:
                    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    data = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip()).strip())
                    return data
                last = f"Gemini returned {r.status_code}: {r.text[:200]}"
                if r.status_code not in (429, 500, 503):
                    break
            except Exception as e:  # network or JSON parse problem
                last = f"{e.__class__.__name__}: {e}"
            await asyncio.sleep(2)
    raise HTTPException(502, f"Script generation failed. {last}")


FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
