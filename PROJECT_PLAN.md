# PROJECT_PLAN.md — StockPulse

## CLAUDE CODE: READ THIS FIRST EVERY SESSION

Before writing any code, do the following:
1. Read this file completely
2. Check which phase we're in and what's next
3. Confirm with Martin what we're working on today
4. After completing work, update the checkboxes below and note the date

---

## Current Phase: **6 — Live Paper Trading (ACTIVE)**
## Last Session: 2026-02-08
## Status: RUNNING IN PRODUCTION

### What's Working:
- ✅ Scheduler running (`stockpulse run`)
- ✅ 6 active trading strategies with optimized params
- ✅ Confidence-based position sizing (5-15% based on signal quality)
- ✅ Strategy allocation weights (sector_rotation 2.0x, etc.)
- ✅ Per-strategy concentration limits (max 40%)
- ✅ Consolidated scan emails (one per scan, only if changed)
- ✅ Daily portfolio digest at 17:00 ET
- ✅ Long-term scanner digest at 17:30 ET
- ✅ Trillion+ Club scanner at 17:31 ET (mega-cap entry points)
- ✅ AI Pulse scanner at 17:32 ET (~70 AI stocks + thesis research)
- ✅ Auto-open/close paper positions
- ✅ P&L tracking with transaction costs
- ✅ Per-strategy signal breakdown in scan output
- ✅ Near-miss detection (stocks close to triggering)
- ✅ Human-readable strategy descriptions
- ✅ Long-term email: trend tracking, Strong Buy categories
- ✅ Compact scheduler countdown bar
- ✅ Live Signals: BUY/SELL split with portfolio/cooldown markers
- ✅ Signal Action Analysis: comprehensive blocking reasons for each signal
  - Already in portfolio (📌 HELD)
  - Cooldown period (⏱️ COOLDOWN)
  - Max positions limit
  - Sector concentration limit
  - Strategy concentration limit
  - Loss limit reached
- ✅ Top Signals by Strategy breakdown with status and reason
- ✅ Performance page: Portfolio value and cash time series
- ✅ Cooldown tracking for winning trades (not just losses)
- ✅ Historical P/E backfill via FMP API or calculated method
- ✅ Dashboard data status shows actual price count
- ✅ **Trillion+ Club auto-backfill** (like long-term scanner)
- ✅ **Trillion email trend column** matches long-term format
- ✅ **AI Pulse email** now shows full AI stocks + theses
- ✅ **Consistent email format** across longterm, trillion, AI (table layouts, reasoning rows, compact breakdowns)
- ✅ **Sentiment integration in AI Pulse** — StockTwits + Haiku analysis in scheduler
- ✅ **Git permissions documented** — Martin pushes to main, Claude works on feature branch

### Email Schedule:
| Time (ET) | Email | Content |
|-----------|-------|---------|
| 17:00 | Daily Portfolio Digest | Portfolio value, P&L, positions, activity |
| 17:30 | Long-Term Opportunities | Value stocks near 52-week lows |
| 17:31 | Trillion+ Club | Mega-cap entry points with consolidated score breakdown |
| 17:32 | AI Pulse | AI universe stocks, thesis research, market pulse |
| On signal change | Scan Alert | Consolidated BUY/SELL signals |

### Position Sizing Formula:
```
final_size = base (5%) × strategy_weight × confidence_multiplier
final_size = min(final_size, 15%)  # Hard cap

Confidence multipliers:
- <75%: 1.0x
- 75-84%: 2.0x
- 85%+: 3.0x
```

### Commands:
```bash
stockpulse run              # Start scheduler (continuous)
stockpulse scan             # One-off scan
stockpulse digest           # Send portfolio digest now
stockpulse longterm-scan    # Run long-term scanner
stockpulse trillion-scan    # Run Trillion+ Club scanner
stockpulse ai-scan          # Run AI Pulse scan
stockpulse sentiment-scan   # Daily sentiment scan for AI universe
stockpulse sentiment-check --ticker NVDA  # Quick sentiment check
stockpulse longterm-backfill # Backfill 6 weeks trend data
stockpulse ai-backfill      # Initialize trillion club and theses
stockpulse pe-backfill      # Backfill historical P/E ratios
stockpulse test-email       # Test email config
stockpulse dashboard        # Launch Streamlit
stockpulse reset            # Clear trading data (keeps prices)
```

---

## Phase 0 — Foundation (COMPLETE)

- [x] Initialize project structure (src/, tests/, data/, config/, docs/)
- [x] Set up pyproject.toml with dependencies
- [x] Create config.yaml with sensible defaults
- [x] Set up SQLite (WAL mode) and define schema (all core tables)
- [x] Build universe manager: fetch top 100 US stocks by market cap, store in DB, support manual overrides
- [x] Build intraday data ingestion: yfinance 15-min OHLCV for full universe
- [x] Build daily data ingestion: EOD prices + basic fundamentals
- [x] Add rate limiting, caching, and staleness detection for data pulls
- [x] Set up APScheduler: 15-min intraday job + daily post-close job
- [ ] Write basic integration test: scheduler runs, data lands in DB
- [x] **MILESTONE:** 15-min price data flowing into SQLite for 100 stocks

## Phase 1 — Strategies + Backtesting (COMPLETE)

- [x] Build strategy base class / interface (entry signal, exit signal, confidence score, params)
- [x] Build backtesting framework (vectorized, walk-forward capable)
- [x] Implement Strategy #1: Mean Reversion (RSI)
- [x] Implement Strategy #2: Bollinger Band Squeeze Breakout
- [x] Implement Strategy #3: MACD Crossover + Volume Confirmation
- [x] Implement Strategy #4: Z-Score Mean Reversion
- [x] Implement Strategy #5: Momentum Breakout
- [x] Build signal generator: runs strategies against live data, writes signals to DB
- [x] Build ensemble layer: multi-strategy agreement boosts confidence
- [x] Paper position manager: auto-open positions on signals, track to exit
- [x] **MILESTONE:** 5 strategies implemented, signals generating, paper positions tracking

## Phase 2 — Email Alerts (COMPLETE)

- [x] Build email sender (SMTP, HTML templates)
- [x] Wire up alert triggers: new signal → email
- [x] Implement configurable thresholds (min confidence, quiet hours, per-strategy toggle)
- [x] Build daily digest email for long-term scanner
- [x] Add error alerting (data staleness, system failures)
- [x] Log all sent alerts to DB
- [x] **MILESTONE:** Email alerts system ready (needs Gmail App Password to test)

## Phase 3 — Streamlit Dashboard (COMPLETE)

- [x] Set up Streamlit app skeleton with tab navigation
- [x] Define chart theme (white bg, large fonts, clean axes — NO default matplotlib)
- [x] Tab: Live Signals — table + filters by strategy, confidence, ticker
- [x] Tab: Paper Portfolio — open positions, unrealized P&L, entry dates
- [x] Tab: Performance — equity curve, per-strategy P&L, win rate, Sharpe, drawdown
- [x] Tab: Backtests — historical strategy performance, equity curves, stats tables
- [x] Tab: Settings — config viewer/editor (read-only)
- [x] Tab: Long-Term Watchlist
- [x] Auto-refresh on new data
- [x] **MILESTONE:** Dashboard running with all tabs

## Phase 4 — Long-Term Investment Scanner (COMPLETE)

- [x] Expand fundamentals ingestion: P/E, P/B, PEG, dividend yield, earnings growth, payout ratio
- [x] Build scoring model: valuation + technical accumulation + dividend + earnings quality
- [x] Add 52-week high/low context and historical valuation percentiles
- [x] Generate ranked watchlist, write to DB
- [x] Build weekly digest email with top opportunities + reasoning
- [x] Dashboard tab: Long-Term Watchlist with scores, fundamentals, and charts
- [x] **MILESTONE:** Long-term scanner complete

### Long-Term Backtester Enhancements (2026-02-04)
- [x] Transaction costs (0.1% round-trip)
- [x] Optimizable stop-loss (extended to 20-45%)
- [x] VIX regime detection (skip buying in high vol)
- [x] Sector diversification (40% max per sector)
- [x] Walk-forward optimization (2yr train / 1yr test)
- [x] Optimizable min_score (extended to 40-70)
- [x] Edge-of-range detection for optimal parameters
- [x] **Earnings calendar blackout** - skip buying before earnings
- [x] **Momentum factor** - weight recent 3mo winners
- [x] **Dynamic position sizing** - 0.75x-1.5x based on conviction
- [x] **Trailing stop-loss option** - vs fixed stop
- [x] **Risk profiles** - optimal params by stop-loss level (20-45%)
- [x] **Holdings tracker** - track actual purchases in dashboard
- [x] CLI commands: `stockpulse add-holding`, `stockpulse close-holding`

## Phase 5 — Trade Tracker (Real Trades) (COMPLETE)

- [x] Add manual trade entry (ticker, direction, price, date, size, strategy)
- [x] Track real position P&L using live prices
- [x] Side-by-side comparison: paper vs. real performance
- [x] CSV export of trade history
- [x] CSV import of trade history
- [x] **MILESTONE:** Can log real trades and see comparative P&L vs. paper

## Phase 6 — Hardening (IN PROGRESS)

- [x] Consolidated email alerts (one per scan)
- [x] Daily portfolio digest with full status
- [x] Confidence-based position sizing
- [x] Per-strategy concentration limits
- [ ] Comprehensive error handling and retry logic on all data pulls
- [ ] Data source fallback (yfinance → Alpha Vantage → manual)
- [ ] Walk-forward validation for all strategies (no in-sample-only backtests)
- [ ] System health monitoring (last successful data pull, signal generation status)
- [ ] Graceful handling of market holidays, half days, after-hours
- [ ] Strategy auto-disable on sustained drawdown (configurable threshold)
- [ ] Launchd/systemd service for auto-restart on crash
- [ ] Health check endpoint for monitoring
- [ ] Run unattended for 2+ weeks without intervention
- [ ] **MILESTONE:** Production-grade reliability

### Next Steps (Priority Order):
1. **Monitor for 1-2 weeks** — Watch performance, check emails arrive, verify positions open/close correctly
2. **Add health monitoring** — Dashboard indicator for last successful scan, data freshness
3. **Handle market holidays** — Graceful skip on NYSE closed days
4. **Auto-restart service** — launchd plist for macOS to keep scheduler running
5. **Walk-forward validation** — Re-run optimizer with proper train/test split

## Phase 5.5 — Smart Trading (COMPLETE)

- [x] De-duplication: Cooldown after losses (7 days)
- [x] De-duplication: Churn prevention (3 days after any exit)
- [x] De-duplication: Block ticker after 3 consecutive losses
- [x] Concentration limits: Max 30% portfolio in single sector
- [x] Equity curve time series with drawdown tracking
- [x] Mark-to-market for open positions
- [x] Blocked tickers display in dashboard
- [x] Historical data preloading (2 years on init)
- [x] WCAG 2.1 AA compliant color contrast
- [x] COMMANDS.md reference sheet
- [x] **MILESTONE:** Smart trading logic prevents repeat bad trades

---

## API Keys & Tokens Required

### Email Alerts (REQUIRED)
- **Gmail App Password**: For sending email alerts via SMTP
  - Go to Google Account > Security > 2-Step Verification
  - Scroll to "App passwords" > Generate for "Mail"
  - Set as `STOCKPULSE_EMAIL_PASSWORD` environment variable
  - Also set `STOCKPULSE_EMAIL_SENDER` and `STOCKPULSE_EMAIL_RECIPIENT`

### Historical P/E Data (OPTIONAL)
- **FMP API Key**: For historical P/E ratios from Financial Modeling Prep
  - Get free key at: https://site.financialmodelingprep.com/developer/docs
  - 250 requests/day free tier
  - Set as `FMP_API_KEY` in `.env`
  - If not set, P/E is calculated from price/EPS (works fine)

### Data Sources (NO API KEY NEEDED)
- **yfinance**: Free, no API key required
- **SQLite**: Local database with WAL mode for concurrent access

### Future Options (NOT CURRENTLY USED)
- Alpha Vantage: Free tier available if yfinance becomes unreliable
- Polygon.io: For more reliable/real-time data (paid)

---

## Backlog (Future / Unscheduled)

- [x] Add more strategies: Gap Fade, 52-Week Low Bounce, Sector Rotation
- [x] Hyperparameter optimization with drawdown constraint
- [ ] Earnings Drift strategy (post-earnings momentum)
- [ ] Options-based strategies
- [ ] Broker API integration for automated execution
- [ ] Mobile-friendly dashboard or push notifications
- [ ] ML-based signal combination / meta-learner
- [ ] Portfolio-level risk management (correlation, max sector exposure)
- [ ] Paid data source integration (Polygon.io, Alpaca)
- [ ] Docker containerization for deployment
- [ ] Cloud deployment (EC2 / GCP / Railway)
- [ ] **Social Sentiment Integration** - See Phase 7 proposal below

### Long-Term Backtester Future Enhancements
- [ ] Tax-loss harvesting simulation
- [ ] Dividend reinvestment (DRIP)
- [ ] **Options overlay (covered calls/cash-secured puts)** - generate income on holdings
- [ ] Multi-factor momentum (6mo + 12mo lookback)
- [ ] Earnings surprise momentum (EPS beat streak)
- [ ] **Put/Call Wheel Strategy** for dividend income and buying dips:
  - Sell cash-secured puts on stocks we want to own at lower prices
  - If assigned, collect premium + buy at discount
  - Sell covered calls on holdings for additional income
  - If called away, collect premium + profit
  - Wheel back into puts, repeat

---

## Phase 7 — Social Sentiment Integration (IN PROGRESS)

**Goal:** Use internet chatter (StockTwits, Reddit, news) to inform AI strategies with sentiment analysis via Claude Haiku.

**Status:** ACTIVE - AI Pulse integration only (Trillion+/Long-Term deferred)

### Review & Bug Fixes (2026-02-08)

**Bugs Fixed:**
- [x] Cache key truncation bug (wasn't creating proper 5-min buckets)
- [x] Added retry logic with exponential backoff to StockTwits & Finnhub fetchers
- [x] Added error handling wrapper around Haiku AI analysis
- [x] Added `json.JSONDecodeError` handling for malformed API responses

**See:** `docs/SENTIMENT_REVIEW.md` for full analysis and enhancement roadmap

### Implementation Summary (2026-02-07)

#### Approach: Start Free, Scale Later

Instead of paid APIs, we're starting with **FREE data sources** that require no API keys:

| Source | Access | Cost | Status |
|--------|--------|------|--------|
| **StockTwits** | Public API | **FREE** | ✅ **IMPLEMENTED** |
| **Finnhub** | Free tier (60 req/min) | **FREE** (API key) | ✅ Ready (optional) |
| **Claude Haiku** | Per-token | ~$5/mo | ✅ AI analysis |

#### What's Implemented

**New Module:** `src/stockpulse/data/sentiment.py`

- `StockTwitsFetcher` - Fetches last 30 messages with sentiment (bullish/bearish/neutral)
  - Now with retry logic (3 retries, exponential backoff)
- `FinnhubFetcher` - News sentiment (requires free API key)
  - Now with retry logic (3 retries, exponential backoff)
- `HaikuSentimentAnalyzer` - Claude Haiku for intelligent summarization
- `SentimentAnalyzer` - Unified interface combining all sources
- `SentimentStorage` - Database caching for daily scans
- `run_daily_sentiment_scan()` - Batch scan for AI universe

**Database Tables:**
- `sentiment_daily` - Per-ticker daily sentiment scores
- `sentiment_category_daily` - Aggregated category sentiment

**CLI Commands:**
```bash
stockpulse sentiment-scan   # Run daily sentiment scan for AI universe
stockpulse sentiment-check --ticker NVDA  # Quick check single ticker
```

#### Integration Status

| Scanner | Status | Notes |
|---------|--------|-------|
| **AI Pulse** | ✅ Integrated | Sentiment section in email |
| **Trillion+ Club** | ⏳ Deferred | Future integration |
| **Long-Term** | ⏳ Deferred | Future integration |

#### Daily Workflow

1. 17:00 ET: Scheduler runs sentiment-scan (caches for AI Pulse)
2. Results cached in `sentiment_daily` table
3. 17:30 ET: AI Pulse email includes "Social Sentiment" section
4. Haiku analyzes top bullish/bearish tickers

#### Cost Estimates (Current)

| Component | Monthly Cost |
|-----------|--------------|
| StockTwits API | **$0** (free) |
| Claude Haiku (~6 analyses/day) | ~$2 |
| Finnhub (optional) | **$0** (free tier) |
| **Total** | **~$2/month** |

### API Capacity Analysis

**Current usage vs limits:**
| API | Rate Limit | Current Use | Daily Capacity | Utilization |
|-----|------------|-------------|----------------|-------------|
| StockTwits | ~200/hour | 80/day | 4,800/day | **1.7%** |
| Finnhub | 60/min | 80/day | 86,400/day | **0.1%** |

**Conclusion:** We're using <2% of available API capacity. Room for hourly collection.

### API Keys Required

```bash
# Add to .env (OPTIONAL - StockTwits works without any keys)
FINNHUB_API_KEY=your_key_here  # Optional: https://finnhub.io/register (free)
ANTHROPIC_API_KEY=already_set  # Required for Haiku analysis
```

### Phase 7b: Enhanced Sentiment (IMPLEMENTED 2026-02-08)

**What's New:**
- ✅ Hourly sentiment scan for top 20 AI stocks (10:30-15:30 ET, every hour)
- ✅ Finnhub analyst ratings integration (buy/hold/sell consensus)
- ✅ Finnhub insider transactions integration (Form 4 data)
- ✅ Sentiment integrated into Trillion+ Club emails
- ✅ Sentiment integrated into Long-Term scanner emails
- ✅ Enhanced score calculation (social 40% + analyst 35% + insider 25%)

**New Schedule:**
```
10:30-15:30 ET hourly: Hourly sentiment for top 20 AI stocks (no Haiku)
17:00 ET: Full daily sentiment scan (80 stocks + analyst + insider)
17:15 ET: Long-Term scan (now with sentiment)
17:20 ET: Trillion+ Club scan (now with sentiment)
17:30 ET: AI Pulse email with all sentiment data
```

**Top 20 AI Stocks (hourly monitoring):**
NVDA, MSFT, AAPL, GOOGL, AMZN, META, TSM, AVGO, ORCL, CRM,
AMD, PLTR, SNOW, NOW, ADBE, IBM, INTC, MU, QCOM, ARM

**Database Tables Added:**
- `sentiment_hourly` - Hourly snapshots for top 20
- `sentiment_signals` - Analyst ratings + insider transactions

### Future Enhancements (Phase 7c+)

**Deferred for later:**
- [ ] Trillion+ Club sentiment integration
- [ ] Long-Term scanner sentiment integration
- [ ] Sentiment alerts (spikes, reversals)
- [ ] Insider buying alerts for thesis stocks
- [ ] Analyst upgrade/downgrade alerts
- [ ] Sentiment divergence detection (price vs sentiment)
- [ ] Backtest sentiment as alpha factor

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| StockTwits rate limits | 500ms delay, retry logic, cache daily |
| API failures | Exponential backoff retries (3 attempts) |
| Sentiment noise | Use as confirming factor only, not primary |
| Module not working | **Isolated code** - can be removed cleanly |
| Cost overruns | Start free, only add paid if value proven |

### Success Metrics

- Sentiment data for 80%+ of AI universe daily
- <$10/month total API costs
- API failure rate <5% (with retries)
- User finds sentiment section valuable in AI Pulse email
- No impact on existing scanner reliability

---

## Decisions Log

| Date | Decision | Context |
|------|----------|---------|
| 2026-02-01 | DuckDB over SQLite | Better for analytical queries on price data |
| 2026-02-01 | yfinance as primary data source | Free, no API key, good enough to start |
| 2026-02-01 | Paper trading first | No real money until system proves itself |
| 2026-02-01 | APScheduler over cron | In-process, easier to manage with the Python app |
| 2026-02-02 | 5 strategies implemented | RSI, Bollinger, MACD, Z-Score, Momentum |
| 2026-02-02 | Transaction costs modeled | Slippage 5bps, spread 2bps, no commission |
| 2026-02-02 | Smart trading logic added | Cooldowns, loss limits, sector concentration limits |
| 2026-02-02 | Equity curves added | Time series tracking with drawdown visualization |
| 2026-02-02 | WCAG 2.1 AA color contrast | Dashboard restyled for accessibility |
| 2026-02-02 | **Switched to SQLite** | DuckDB had macOS locking issues; SQLite WAL mode works |
| 2026-02-02 | Dark mode dashboard | Slate color palette, better visual design |
| 2026-02-02 | Auto .env loading | Commands auto-load .env, no manual sourcing needed |
| 2026-02-02 | Scheduler countdown | Shows time until next scan in console |
| 2026-02-02 | Market snapshot in scan | Shows stocks near signal thresholds |
| 2026-02-02 | Hyperparameter optimizer | `stockpulse optimize` finds best params with 25% max DD |
| 2026-02-02 | 3 new strategies added | Gap Fade, 52-Week Low Bounce, Sector Rotation |
| 2026-02-02 | 8 total strategies | Full suite for different market conditions |
| 2026-02-02 | **Optimization complete** | 18-month backtest (2024-08-11 to 2026-02-02), 30 tickers, 50 param combos each |
| 2026-02-02 | Disabled 2 strategies | Gap Fade (-1.3%), Bollinger Squeeze (+0.6%) - underperforming |
| 2026-02-02 | Added allocation weights | sector_rotation 2.0x, momentum/week52 1.5x, others 1.0-1.2x |
| 2026-02-02 | Best performer | Sector Rotation: +40.2% return, 2.87 Sharpe, 9.3% max DD |
| 2026-02-02 | Best risk-adjusted | 52-Week Low Bounce: +10.4% return, 3.67 Sharpe, 0.8% max DD |
| 2026-02-03 | Confidence-based sizing | 5% base × strategy_weight × confidence_mult, capped at 15% |
| 2026-02-03 | Per-strategy limits | Max 40% of capital in any single strategy |
| 2026-02-03 | Consolidated emails | One email per scan (not per signal), only if changed |
| 2026-02-03 | Daily digest enhanced | Portfolio value, unrealized P&L, today's activity |
| 2026-02-03 | Max positions to 40 | Increased from 25 to allow more diversification |
| 2026-02-03 | **LIVE TRADING STARTED** | Scheduler running, paper portfolio active |
| 2026-02-04 | Long-term backtester v2 | Trailing stops, momentum, dynamic sizing, earnings blackout |
| 2026-02-04 | Extended param ranges | stop_loss 20-45%, min_score 40-70 |
| 2026-02-04 | Risk profiles | Optimal params by stop-loss level for risk tuning |
| 2026-02-04 | Holdings tracker | Track actual purchases, CLI commands, dashboard page |
| 2026-02-05 | Per-strategy insights | Near-miss detection, blocking reasons, signal breakdown in scan |
| 2026-02-05 | Human-readable strategies | Wife-friendly descriptions with acronym definitions |
| 2026-02-05 | Long-term trend tracking | Consecutive days on list, score trends, Strong Buy categories |
| 2026-02-05 | Backfill command | `longterm-backfill` populates 6 weeks historical data |
| 2026-02-05 | Compact scheduler output | Single-line status with countdown bar, ET times |
| 2026-02-07 | Trillion+ Club scanner | Mega-cap entry point tracking with consolidated score breakdown |
| 2026-02-07 | AI Pulse scanner | ~70 AI stocks, 7 categories, Claude thesis research |
| 2026-02-07 | Email consolidation | Score breakdowns in single table for easy comparison |
| 2026-02-07 | Trillion+ auto-backfill | Auto-detect gaps and backfill 3 weeks of historical data (like long-term) |
| 2026-02-07 | AI Pulse email fixed | Now shows AI stocks table + theses (was blank) |
| 2026-02-07 | Email format consistency | Trillion + AI emails reformatted to match longterm style (table layouts, reasoning rows, compact breakdowns) |
| 2026-02-07 | Sentiment integration proposal | Phase 7 plan: StockGeist + ApeWisdom + Claude, $15-70/mo |
| 2026-02-07 | **Sentiment MVP implemented** | StockTwits (free) + Haiku, ~$5/mo, AI Pulse only |
| 2026-02-07 | Sentiment isolation | Code in `data/sentiment.py` can be removed if not working |
| 2026-02-07 | Trillion+/Long-Term deferred | Sentiment integration for other scanners is future work |
| 2026-02-08 | **Sentiment review complete** | Fixed cache key bug, added retry logic, error handling |
| 2026-02-08 | API capacity analysis | Using <2% of available capacity; room for hourly collection |
| 2026-02-08 | Alternative data sources identified | SEC EDGAR, Finnhub analyst ratings, insider transactions - all FREE |
| 2026-02-08 | Phase 7b roadmap created | Hourly collection option + more data sources proposed |
| 2026-02-08 | **Phase 7b implemented** | Hourly sentiment + analyst ratings + insider txns + Trillion/LT integration |
| 2026-02-08 | Cost-conscious data strategy | Maximize FREE APIs, limit Haiku to 6 calls/day (top 3 bullish + top 3 bearish) |

---

## Optimization Results (2026-02-02)

### Backtest Period
- **Start:** 2024-08-11
- **End:** 2026-02-02 (18 months)
- **Universe:** 30 tickers (top by market cap)
- **Initial Capital:** $100,000
- **Constraint:** Max 25% drawdown

### Strategy Performance Summary

| Strategy | Final Value | Return | Std Dev | Sharpe | Max DD | Trades |
|----------|-------------|--------|---------|--------|--------|--------|
| sector_rotation | $140,157 | +40.2% | ±8.2% | 2.87 | 9.3% | Active |
| momentum_breakout | $119,346 | +19.3% | ±5.5% | 2.23 | 3.3% | Active |
| zscore_mean_reversion | $114,881 | +14.9% | ±5.3% | 1.80 | 1.8% | Active |
| macd_volume | $113,195 | +13.2% | ±3.9% | 2.21 | 2.4% | Active |
| rsi_mean_reversion | $111,422 | +11.4% | ±4.3% | 1.73 | 3.9% | Active |
| week52_low_bounce | $110,392 | +10.4% | ±1.8% | 3.67 | 0.8% | Active |
| bollinger_squeeze | $100,622 | +0.6% | ±0.3% | 1.66 | 0.0% | **DISABLED** |
| gap_fade | $98,682 | -1.3% | ±3.6% | -0.23 | 6.0% | **DISABLED** |

### Optimized Parameters (Active Strategies)

**sector_rotation** (2.0x allocation)
- lookback_days: 10
- top_sectors: 2
- min_sector_return: 1.5
- relative_strength_threshold: 1.2
- stop_loss_pct: 3.0
- take_profit_pct: 8.0

**momentum_breakout** (1.5x allocation)
- lookback_days: 20
- breakout_threshold: 0.01
- volume_confirmation: 1.3
- stop_loss_pct: 5.0
- take_profit_pct: 15.0

**week52_low_bounce** (1.5x allocation)
- low_threshold_pct: 12.0
- bounce_threshold_pct: 2.0
- volume_surge: 1.5
- stop_loss_pct: 5.0
- take_profit_pct: 20.0

**zscore_mean_reversion** (1.2x allocation)
- lookback_period: 20
- zscore_entry: -2.25
- zscore_exit: 0.5
- stop_loss_pct: 6.0
- take_profit_pct: 12.0

**macd_volume** (1.2x allocation)
- macd_fast: 16
- macd_slow: 26
- macd_signal: 7
- volume_threshold: 1.5
- stop_loss_pct: 3.0
- take_profit_pct: 15.0

**rsi_mean_reversion** (1.0x allocation)
- rsi_period: 10
- rsi_oversold: 25
- rsi_overbought: 80
- stop_loss_pct: 5.0
- take_profit_pct: 6.0

---

## Notes for Claude Code

- Martin is a data scientist. He thinks in terms of experimentation and statistical rigor. Every strategy needs backtested evidence before going live.
- Charts must be publication-quality: white background, large fonts, clear labels. No default styling. Ever.
- Martin prefers to talk through design before coding. Ask before building if requirements are ambiguous.
- Use type hints. Write docstrings. Keep functions small and testable.
- Config-driven everything. No magic numbers in strategy code.
- **ALWAYS give pull/run/test commands** at end of changes. No comments in command blocks (Mac terminal hates them).

### Cost-Conscious Data Strategy (IMPORTANT)

**GOAL:** Maximize free data sources, minimize paid API calls (especially Claude/Anthropic).

| Data Type | Source | Cost | Usage |
|-----------|--------|------|-------|
| Social Sentiment | StockTwits | FREE | Unlimited |
| News Sentiment | Finnhub | FREE | 60/min |
| Analyst Ratings | Finnhub | FREE | 60/min |
| Insider Transactions | Finnhub | FREE | 60/min |
| SEC Filings | SEC EDGAR | FREE | 10/sec |
| AI Analysis | Claude Haiku | **PAID** | **LIMIT** |

**Haiku Usage Rules:**
- Only analyze top 3 bullish + top 3 bearish stocks per day (6 total)
- Never run Haiku on hourly scans - only daily aggregation
- Cache Haiku results for 24 hours - don't re-analyze same ticker
- Target: <$5/month Anthropic spend

---

## Git Hygiene (Claude Code must follow)

### Branch Strategy
- **Feature branch**: `claude/init-repo-setup-maaOL` — All Claude Code work happens here
- **Main branch**: `main` — Production code, updated via merge from feature branch
- Martin pulls from feature branch on his local machine, merges to main, pushes to origin

### Workflow for Claude Code
1. **Always work on feature branch**: `git checkout claude/init-repo-setup-maaOL`
2. **Commit frequently** with clear messages describing what changed
3. **Push to feature branch**: `git push -u origin claude/init-repo-setup-maaOL`
4. **Never force push** or rewrite history
5. **Update PROJECT_PLAN.md** at end of each session with work done

### Permissions
- **Martin has full push access to `main`** — can push directly after merging feature branch
- Claude Code cannot push to `main` (403 protected) — works on feature branch only

### Workflow for Martin (local machine)
```bash
# Pull latest from feature branch
git fetch origin claude/init-repo-setup-maaOL
git checkout main
git merge origin/claude/init-repo-setup-maaOL

# Push to main (Martin has permission)
git push origin main

# Run the app
stockpulse run
```

### Commit Message Format
```
Short summary of change (50 chars max)

- Bullet point details if needed
- What was added/changed/fixed

https://claude.ai/code/session_ID
```

### Session Checklist (Claude Code)
- [ ] Read PROJECT_PLAN.md at session start
- [ ] Work on feature branch only
- [ ] Commit and push changes
- [ ] Update PROJECT_PLAN.md with session work
- [ ] Give Martin pull/merge commands at session end
