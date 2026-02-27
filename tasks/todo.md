# Sentiment Analysis Expansion + Review Agent System

## Status: COMPLETE (Feb 26, 2026)

All 4 phases implemented and compiling. Ready for integration testing.

---

## Phase 1: New Free Sentiment Sources ✅

### 1a. Reddit via PRAW ✅

- [x] `RedditFetcher` class in `sentiment.py`
- [x] Searches wallstreetbets+stocks+investing+options+stockmarket
- [x] Keyword-based sentiment classification with upvote ratio boost
- [x] `get_trending_tickers()` for Reddit buzz section
- [x] Ticker blacklist to filter common words (DD, CEO, YOLO, etc.)
- [x] Config: `config.yaml > sentiment.reddit`

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
- [x] Configurable weights: Social 30%, News 20%, Analyst 25%, Insider 15%, Alternative 10%
- [x] Auto-redistribution when sources unavailable

---

## Phase 2: Dashboard Enhancements ✅

- [x] **2a. Sentiment Sources Overview** — Expandable panel showing per-source data counts
- [x] **2b. Score Radar Chart** — Plotly polar chart for AI scoring factor visualization
- [x] **2c. Sentiment Trend Chart** — `create_sentiment_trend_chart()` in charts.py
- [x] **2d. Council Perspectives** — Expandable per-thesis showing each agent's analysis
- [x] **2e. Insider/Analyst Breakdown** — Per-source detail in stock detail view
- [x] **2f. Market Mood Banner** — Fear & Greed Index at top of AI Stocks page
- [x] **2g. Reddit Buzz Section** — Top Reddit-discussed tickers table

---

## Phase 3: Email Expansion ✅

- [x] **3a. Long-Term & Trillion sentiment** — Already integrated in prior work
- [x] **3b. Reddit Buzz in AI Pulse** — Top 5 Reddit-discussed stocks section
- [x] **3c. Sentiment Reversal Alerts** — 20+ point shift vs 7-day average detection
- [x] **3d. Market Context Header** — Fear & Greed score in email header

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

- [ ] Run `stockpulse sentiment-scan` and verify new sources collect data
- [ ] Check `sentiment_daily` table has rows for source='reddit' and source='google_news'
- [ ] Load dashboard, verify new charts/panels render with real data
- [ ] Send test email and verify new sentiment sections appear
- [ ] Run `/review` and confirm all 6 agents produce structured adversarial output

## Dependencies Installed

- `praw` — Reddit API wrapper
- `feedparser` — RSS feed parser (Google News)
- `pytrends` — Google Trends wrapper
