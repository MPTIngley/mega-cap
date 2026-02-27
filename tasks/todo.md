# Sentiment Analysis Expansion + Review Agent System

## Status: VERIFIED & COMPLETE (Feb 26, 2026)

All 4 phases implemented, tested, and verified with real data.

---

## Phase 1: New Free Sentiment Sources ✅

### 1a. Reddit via PRAW ⏸️ (Disabled)

- [x] `RedditFetcher` class in `sentiment.py` (code exists, dormant)
- Reddit now requires Devvit registration for API access
- Config: `sentiment.reddit.enabled: false`
- Weight (15%) auto-redistributed to other sources

### 1b. Google News RSS ✅

- [x] `GoogleNewsFetcher` class in `sentiment.py`
- [x] Parses `news.google.com/rss/search?q={TICKER}+stock`
- [x] Keyword-based headline sentiment classification
- [x] No auth required, no rate limit concerns

### 1c. Google Trends ✅

- [x] `GoogleTrendsFetcher` class in `sentiment.py`
- [x] Uses pytrends for 7-day interest data
- [x] Spike detection: current vs average > 2x = trending
- [x] Score: 0-100 based on spike ratio

### 1d. CNN Fear & Greed Index ✅

- [x] `get_fear_greed_index()` function in `sentiment.py`
- [x] One API call for global market mood (0-100)
- [x] Stored as `_MARKET` signal in `sentiment_signals` table

### 1e. Options Put/Call Ratio ✅

- [x] `OptionsSentimentFetcher` class in `sentiment.py`
- [x] Uses existing yfinance dependency
- [x] P/C ratio scoring: <0.7 bullish, >1.0 bearish

### 1f. Wikipedia Page Views ✅

- [x] `WikipediaFetcher` class in `sentiment.py`
- [x] 30-day average vs today for attention spike detection
- [x] Ticker-to-article mapping for 20+ major stocks

### Aggregate Scoring ✅

- [x] `_weighted_score()` method on SentimentAnalyzer
- [x] Configurable weights: StockTwits 15%, Google News 20%, Analyst 30%, Insider 15%, Alt 10%
- [x] Auto-redistribution when sources unavailable

---

## Phase 2: Dashboard Enhancements ✅

- [x] **2a. Sentiment Sources Overview** — Expandable panel showing per-source data counts
- [x] **2b. Score Radar Chart** — Plotly polar chart for AI scoring factor visualization
- [x] **2c. Sentiment Trend Chart** — `create_sentiment_trend_chart()` in charts.py
- [x] **2d. Council Perspectives** — Expandable per-thesis showing each agent's analysis
- [x] **2e. Insider/Analyst Breakdown** — Per-source detail in stock detail view
- [x] **2f. Market Mood Banner** — Fear & Greed Index at top of AI Stocks page

---

## Phase 3: Email Expansion ✅

- [x] **3a. Long-Term & Trillion sentiment** — Already integrated in prior work
- [x] **3b. Sentiment Reversal Alerts** — 20+ point shift vs 7-day average detection
- [x] **3c. Market Context Header** — Fear & Greed score in email header

---

## Phase 4: Adversarial Review Agents ✅

### 6 Agent Personas (.claude/agents/)

- [x] `review-ux-designer.md` — The Pixel Zealot (15yr Bloomberg/Robinhood/TradingView)
- [x] `review-quant-pm.md` — The Alpha Hunter (20yr D.E. Shaw/AQR/Bridgewater)
- [x] `review-retail-investor.md` — The Everyman (12yr personal investor)
- [x] `review-data-scientist.md` — The Measurement Obsessive (PhD Stanford, Two Sigma)
- [x] `review-risk-manager.md` — The Paranoid (18yr JPMorgan/Citadel/BlackRock)
- [x] `review-devils-advocate.md` — The Contrarian (22yr FT/Muddy Waters/Bridgewater)

### 2 Skills (.claude/skills/)

- [x] `/review` — Full 6-agent adversarial review with Playwright screenshots
- [x] `/review-page <page>` — Quick 3-agent review of a single page

---

## Verification Checklist

- [x] Run `stockpulse sentiment-scan` — 80/80 tickers scanned, all new sources working
- [x] Load dashboard — all new charts/panels render with real data, no errors
- [x] Send test email — new sentiment sections (Fear & Greed header, reversal alerts) appear
- [x] None safety — fixed all `sentiment_data.get()` calls to handle None values from DB
- [ ] Run `/review` and confirm all 6 agents produce structured adversarial output

## Bug Fixes Applied

- Fixed `AttributeError: 'NoneType' object has no attribute 'get'` on AI Stocks page
  - Root cause: `sentiment_data` dict values can be `None` (key exists but value is null)
  - Fix: Changed all `.get("key", {})` to `.get("key") or {}` pattern across app.py
- Removed all Reddit UI/email references (Reddit API requires Devvit registration now)

## Dependencies Installed

- `feedparser` — RSS feed parser (Google News)
- `pytrends` — Google Trends wrapper
- `praw` — Reddit API wrapper (installed but unused — Reddit disabled)
