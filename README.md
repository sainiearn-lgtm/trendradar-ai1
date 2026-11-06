# TrendRadar AI

## Voice Studio v1.6.2 — Indic Tiny TTS Hindi

- Uses Indic Tiny TTS Hindi with male/female voices and WAV output.
- Script + Voice runs sequentially; voice receives only the narration section, not production notes or English headings.
- Narration is requested in Devanagari Hindi. Speed, pitch, and volume controls remain available.


TrendRadar AI is a responsive creator-intelligence dashboard. It combines public trend feeds and an optional YouTube Data API signal, ranks source-backed topics with a transparent estimate, and generates video scripts through Google Gemini.

## What is included

- FastAPI backend with `/`, `/api/health`, `/api/trends`, and `/api/script`.
- Public Google Trends RSS and GDELT news feed integrations.
- Optional YouTube Data API `mostPopular` integration (requires an API key).
- Region support: India, US, UK, Canada, Australia, Germany.
- Trend scoring using recency, repeated feed mentions, source coverage and optional popularity indicators.
- Gemini-powered Hindi, Hinglish and English scripts for Shorts and long videos.
- Voice Studio: generate a script only, or generate the script plus a neural MP3 voiceover with Hindi/Hinglish/English male/female voice choices and speaking speed, pitch and volume controls (with reset-to-default).
- In-memory cache, request timeout, basic per-IP rate limit, input validation, logging and CORS.
- Responsive static frontend.

## Important notes

1. Google Trends RSS and GDELT are public endpoints, but they can change, throttle, or occasionally be unavailable. The API reports source status and returns a useful error if every feed fails.
2. GDELT results are recent articles matching the selected region, not a certified list of viral topics. Google Trends RSS is the search-trend signal. Optional YouTube results are the platform's `mostPopular` chart, not Instagram trends.
3. TrendRadar scores are estimates, not official virality metrics. They are based on available source signals and should be treated as a research aid.
4. Instagram does not provide a general public trending-topics API. This project does not pretend to fetch private Instagram analytics or fabricate Instagram trends.
5. Gemini is optional for trend discovery, but required for AI script generation. Keep its key on the backend only.
6. Voice generation uses the `edge-tts` package with Microsoft Edge online neural voices. It requires outbound internet access; the speech service can change, throttle, or become unavailable. It is not a guaranteed human recording, and commercial usage should be checked against the provider's current terms.
7. The free Render instance may sleep when idle. The first request after inactivity can take longer. In-memory cache resets when the service restarts.

## Run locally (beginner-friendly)

You need Python 3.10+.

### 1. Open the project folder

Open a terminal in `TrendRadar-AI/backend`.

### 2. Create a virtual environment

Windows:
```bash
py -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Configure secrets

Copy `.env.example` to `.env`, then open `.env` and add your Gemini API key:
```env
GEMINI_API_KEY=your_real_key
CORS_ORIGINS=*
```
Get a key from Google AI Studio: https://aistudio.google.com/app/apikey

Optional YouTube key: create a project in Google Cloud Console, enable YouTube Data API v3, and set `YOUTUBE_API_KEY`.

### 5. Start the backend

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Check:
- http://127.0.0.1:8000/
- http://127.0.0.1:8000/api/health
- http://127.0.0.1:8000/docs

### 6. Open the frontend

For a quick local preview, open `frontend/index.html` in your browser. The frontend API base must point to your backend. In `index.html`, replace:
```js
"https://YOUR-BACKEND.onrender.com"
```
with:
```js
"http://127.0.0.1:8000"
```

## Deploy backend to Render

1. Create a GitHub account/repository and upload the complete `TrendRadar-AI` folder.
2. In Render, choose **New + → Blueprint** and connect the repository. Render reads `render.yaml`.
3. Confirm the service root is `backend` (already configured in the Blueprint).
4. In the Render service's **Environment** tab, set:
   - `GEMINI_API_KEY` = your Google AI Studio key
   - `CORS_ORIGINS` = `*` for first deployment, or your exact frontend origin after deploying it (example: `https://trendradar-ai.onrender.com`)
   - `YOUTUBE_API_KEY` = optional
5. Deploy. Copy the service URL, for example `https://trendradar-ai-api.onrender.com`.
6. Test `https://YOUR-BACKEND.onrender.com/api/health` and `/api/trends?region=IN`.

Do not put Gemini or YouTube keys in frontend code, GitHub, or a public README.

## Deploy frontend

### Option A: Render Static Site

1. In Render choose **New + → Static Site** and connect the same GitHub repository.
2. Set **Root Directory** to `frontend`.
3. No build command is needed. Set **Publish Directory** to `.`.
4. Deploy the site.
5. Open `frontend/index.html` in GitHub and replace the `API_BASE` value with your backend Render URL, commit the change, and let Render redeploy.
6. Set backend `CORS_ORIGINS` to your exact static-site URL, save, and redeploy the backend.

### Option B: Vercel

1. Import the GitHub repository into Vercel.
2. Set the project **Root Directory** to `frontend`.
3. Framework preset: **Other**. Build command: leave blank. Output directory: `.`.
4. Deploy, then update `API_BASE` in `frontend/index.html` to your backend URL and push the change.
5. Set backend `CORS_ORIGINS` to the exact Vercel domain, then redeploy backend.

## API reference

### `GET /`
Returns API name, status and version.

### `GET /api/health`
Returns health state and whether Gemini / YouTube keys are configured. It never returns secret values.

### `GET /api/trends?region=IN&q=ai`
Supported region codes: `IN`, `US`, `GB`, `CA`, `AU`, `DE`. `q` is optional and filters returned topics. Response includes `trends`, source statuses, `updated_at`, and score explanation.

Each trend includes:
`title`, `summary`, `source`, `source_url`, `region`, `category`, `trend_score`, `published_at`, `mentions`.

### `POST /api/script`
Request:
```json
{
  "topic": "Artificial Intelligence",
  "language": "Hinglish",
  "format": "Shorts (60 seconds)"
}
```
Allowed languages: `Hindi`, `Hinglish`, `English`.
Allowed formats: `Shorts (60 seconds)`, `Long video (5-10 minutes)`.
Response contains `script`, `topic`, `language`, `format`, and model name.

## Troubleshooting

- **Backend is sleeping / first load is slow:** Render free services can spin down. Wait and retry once.
- **CORS error:** Set `CORS_ORIGINS` to the exact frontend origin, including `https://`, with no trailing slash. Redeploy backend.
- **AI says not configured:** Add `GEMINI_API_KEY` to Render Environment and redeploy.
- **Trend source temporarily unavailable:** Retry later. The API uses remaining sources when one feed fails; if all feeds fail, it returns HTTP 503 rather than inventing topics.
- **No YouTube items:** Set a valid `YOUTUBE_API_KEY`; this source is optional.
- **Wrong API endpoint in frontend:** Replace the `API_BASE` placeholder in `frontend/index.html` with the deployed backend URL.

## Security and production considerations

This starter is designed for a small creator dashboard. The in-memory cache and IP rate limiter are per process and reset on restart; for a larger deployment, use Redis-backed rate limiting/cache, add authentication, monitor API quotas, and pin CORS to the frontend domain. Source titles and summaries are escaped in the browser before rendering. Source URLs are checked for HTTP(S) before they are returned.


## Frontend configuration

Before deploying the frontend, open `frontend/index.html` and replace:

`https://YOUR-BACKEND.onrender.com`

in the `API_BASE` constant with your actual deployed FastAPI backend URL (without a trailing slash). For example:

`const API_BASE = "https://trendradar-api.onrender.com";`

The frontend is designed to work with this project's FastAPI response fields (`trend_score`, `source_url`, `detail`) and normalizes the long-video format value expected by the backend. Deploy the `frontend` folder as a static site on Render or as the project root on Vercel.


## Voice Studio v1.6.1 — Indic Tiny TTS Hindi

- Hindi male and female neural voices using the ONNX Runtime model `sulabhkatiyar/indic-tiny-tts-hindi`.
- CPU inference; no paid TTS API or GPU required.
- Speed (50–150%), pitch (-6 to +6 semitones), and volume (0–200%) controls.
- Audio preview and WAV download.
- The model supports Hindi in Devanagari script only. Roman-script Hinglish and English are not supported by this model.
- The model files are downloaded from Hugging Face on first voice generation and cached in `/tmp`; the first request can take longer.

### Required attribution and license

This project uses **Indic Tiny TTS: Hindi** by Sulabh Katiyar. The acoustic model weights are licensed under **CC-BY-4.0** and may be used commercially when attribution is provided. The model uses the AI4Bharat **Rasa** Hindi speech dataset (CC-BY-4.0); please credit the dataset and cite the model as requested by its model card. The Vocos vocoder weights are MIT-licensed and the model code is MIT-licensed. Review the upstream model card and `LICENSE.md` before redistribution: https://huggingface.co/sulabhkatiyar/indic-tiny-tts-hindi

Suggested attribution: “Indic Tiny TTS: Hindi by Sulabh Katiyar; trained using the AI4Bharat Rasa dataset. Model: https://huggingface.co/sulabhkatiyar/indic-tiny-tts-hindi.”


### v1.6.1 reliability fix

The first-use model download is restricted to the required Python package, configuration, and ONNX runtime files. Sample audio, demo assets, and safetensors are not downloaded. The model revision is pinned for repeatable deployments.
