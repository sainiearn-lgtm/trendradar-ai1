# TrendRadar AI

Live trend discovery plus an AI video script studio. One Python service serves both the API and the website.

Sources: Google Trends RSS, Google News RSS, GDELT, and YouTube Trending (optional, needs a key).
Scores (0-100) are estimates from source signals + recency + repeated mentions, not official virality metrics.

## 1. Get your keys
- **Gemini (required):** https://aistudio.google.com/apikey -> Create API key.
- **YouTube Data API v3 (optional):** Google Cloud Console -> enable "YouTube Data API v3" -> Credentials -> API key.

## 2. Run on your computer (optional)
1. Install Python 3.11+ from python.org.
2. In a terminal inside this folder:
   ```
   cd backend
   pip install -r requirements.txt
   ```
3. Set your key, then start:
   - Mac/Linux: `export GEMINI_API_KEY=your_key` then `uvicorn main:app --reload`
   - Windows (PowerShell): `$env:GEMINI_API_KEY="your_key"; uvicorn main:app --reload`
4. Open http://127.0.0.1:8000

## 3. Put it on GitHub
1. Create a new empty repository on github.com.
2. In this folder:
   ```
   git init
   git add .
   git commit -m "TrendRadar AI"
   git branch -M main
   git remote add origin https://github.com/YOUR-NAME/trendradar-ai.git
   git push -u origin main
   ```
   (`.env` is ignored, so your keys never get uploaded.)

## 4. Deploy on Render (free)
1. Sign in at render.com with GitHub -> **New +** -> **Blueprint** -> pick your repo. Render reads `render.yaml`.
2. When asked, paste `GEMINI_API_KEY` (and `YOUTUBE_API_KEY` if you have one).
3. Click **Apply**. After the build, open the `.onrender.com` link. Done.

The free plan sleeps after inactivity, so the first load can take ~50 seconds.

## Optional: frontend on Vercel
Not needed, but if you want it: import the repo in Vercel, set Root Directory to `frontend`, then edit `frontend/config.js` and set `window.API_BASE` to your Render URL.

## Troubleshooting
- "GEMINI_API_KEY is not set": add it under Render -> Environment, then redeploy.
- A feed shows an error in the status pill: that source is down or rate-limited (GDELT limits fast repeats). Others keep working.
- Results are cached for 10 minutes to protect free rate limits.
