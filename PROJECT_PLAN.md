# PROJECT_PLAN.md — StockPulse

## CLAUDE CODE: READ THIS FIRST EVERY SESSION

Before writing any code, do the following:
1. Read this file completely
2. Check which phase we're in and what's next
3. Confirm with Martin what we're working on today
4. After completing work, update the checkboxes below and note the date

---

## Current Phase: **5 — Trade Tracker (COMPLETE)**
## Last Session: 2026-02-02
## Next Steps:
1. ✅ Gmail App Password configured in .env
2. **NOW:** Test dashboard launch
3. Run initial data ingestion (happens automatically on first launch)
4. Verify email alerts work (send test)
5. Run backtest to validate strategies on historical data
6. Begin Phase 6 (Hardening) if everything works

---

## Phase 0 — Foundation (COMPLETE)

- [x] Initialize project structure (src/, tests/, data/, config/, docs/)
- [x] Set up pyproject.toml with dependencies
- [x] Create config.yaml with sensible defaults
- [x] Set up DuckDB and define schema (all core tables)
- [x] Build universe manager: fetch top 100 US stocks by market cap, store in DB, support manual overrides
- [x] Build intraday data ingestion: yfinance 15-min OHLCV for full universe
- [x] Build daily data ingestion: EOD prices + basic fundamentals
- [x] Add rate limiting, caching, and staleness detection for data pulls
- [x] Set up APScheduler: 15-min intraday job + daily post-close job
- [ ] Write basic integration test: scheduler runs, data lands in DB
- [x] **MILESTONE:** 15-min price data flowing into DuckDB for 100 stocks

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

## Phase 5 — Trade Tracker (Real Trades) (COMPLETE)

- [x] Add manual trade entry (ticker, direction, price, date, size, strategy)
- [x] Track real position P&L using live prices
- [x] Side-by-side comparison: paper vs. real performance
- [x] CSV export of trade history
- [x] CSV import of trade history
- [x] **MILESTONE:** Can log real trades and see comparative P&L vs. paper

## Phase 6 — Hardening (PENDING)

- [ ] Comprehensive error handling and retry logic on all data pulls
- [ ] Data source fallback (yfinance → Alpha Vantage → manual)
- [ ] Walk-forward validation for all strategies (no in-sample-only backtests)
- [ ] System health monitoring (last successful data pull, signal generation status)
- [ ] Graceful handling of market holidays, half days, after-hours
- [ ] Strategy auto-disable on sustained drawdown (configurable threshold)
- [ ] Run unattended for 2+ weeks without intervention
- [ ] **MILESTONE:** Production-grade reliability

---

## API Keys & Tokens Required

### Email Alerts (REQUIRED)
- **Gmail App Password**: For sending email alerts via SMTP
  - Go to Google Account > Security > 2-Step Verification
  - Scroll to "App passwords" > Generate for "Mail"
  - Set as `STOCKPULSE_EMAIL_PASSWORD` environment variable
  - Also set `STOCKPULSE_EMAIL_SENDER` and `STOCKPULSE_EMAIL_RECIPIENT`

### Data Sources (NO API KEY NEEDED)
- **yfinance**: Free, no API key required
- **DuckDB**: Local database, no external service

### Future Options (NOT CURRENTLY USED)
- Alpha Vantage: Free tier available if yfinance becomes unreliable
- Polygon.io: For more reliable/real-time data (paid)

---

## Backlog (Future / Unscheduled)

- [ ] Add more strategies: Earnings Drift, Gap-and-Go, MA Crossover (10/50 EMA)
- [ ] Sector rotation signals
- [ ] Options-based strategies
- [ ] Broker API integration for automated execution
- [ ] Mobile-friendly dashboard or push notifications
- [ ] ML-based signal combination / meta-learner
- [ ] Portfolio-level risk management (correlation, max sector exposure)
- [ ] Paid data source integration (Polygon.io, Alpaca)
- [ ] Docker containerization for deployment
- [ ] Cloud deployment (EC2 / GCP / Railway)

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

---

## Notes for Claude Code

- Martin is a data scientist. He thinks in terms of experimentation and statistical rigor. Every strategy needs backtested evidence before going live.
- Charts must be publication-quality: white background, large fonts, clear labels. No default styling. Ever.
- Martin prefers to talk through design before coding. Ask before building if requirements are ambiguous.
- Use type hints. Write docstrings. Keep functions small and testable.
- Config-driven everything. No magic numbers in strategy code.
