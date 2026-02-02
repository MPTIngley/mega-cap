# PROJECT_PLAN.md — StockPulse

## ⚠️ CLAUDE CODE: READ THIS FIRST EVERY SESSION

Before writing any code, do the following:
1. Read this file completely
2. Check which phase we're in and what's next
3. Confirm with Martin what we're working on today
4. After completing work, update the checkboxes below and note the date

---

## Current Phase: **0 — Foundation**
## Last Session: _not started_
## Blockers: _none yet_

---

## Phase 0 — Foundation

- [ ] Initialize project structure (src/, tests/, data/, config/, docs/)
- [ ] Set up pyproject.toml with dependencies
- [ ] Create config.yaml with sensible defaults
- [ ] Set up DuckDB and define schema (all core tables)
- [ ] Build universe manager: fetch top 100 US stocks by market cap, store in DB, support manual overrides
- [ ] Build intraday data ingestion: yfinance 15-min OHLCV for full universe
- [ ] Build daily data ingestion: EOD prices + basic fundamentals
- [ ] Add rate limiting, caching, and staleness detection for data pulls
- [ ] Set up APScheduler: 15-min intraday job + daily post-close job
- [ ] Write basic integration test: scheduler runs, data lands in DB
- [ ] **MILESTONE:** 15-min price data flowing into DuckDB for 100 stocks

## Phase 1 — Strategies + Backtesting

- [ ] Build strategy base class / interface (entry signal, exit signal, confidence score, params)
- [ ] Build backtesting framework (vectorized, walk-forward capable)
- [ ] Implement Strategy #1: Mean Reversion (RSI)
- [ ] Backtest Strategy #1, generate performance report
- [ ] Implement Strategy #2: Bollinger Band Squeeze Breakout
- [ ] Backtest Strategy #2, generate performance report
- [ ] Implement Strategy #3: MACD Crossover + Volume Confirmation
- [ ] Backtest Strategy #3, generate performance report
- [ ] Build signal generator: runs strategies against live data, writes signals to DB
- [ ] Build ensemble layer: multi-strategy agreement boosts confidence
- [ ] Paper position manager: auto-open positions on signals, track to exit
- [ ] **MILESTONE:** 3 strategies backtested, signals generating on live data, paper positions tracking

## Phase 2 — Email Alerts

- [ ] Build email sender (SMTP, HTML templates)
- [ ] Wire up alert triggers: new signal → email
- [ ] Implement configurable thresholds (min confidence, quiet hours, per-strategy toggle)
- [ ] Build daily digest email for long-term scanner (placeholder content for now)
- [ ] Add error alerting (data staleness, system failures)
- [ ] Log all sent alerts to DB
- [ ] **MILESTONE:** Receiving real email alerts when signals fire

## Phase 3 — Streamlit Dashboard

- [ ] Set up Streamlit app skeleton with tab navigation
- [ ] Define chart theme (white bg, large fonts, clean axes — NO default matplotlib)
- [ ] Tab: Live Signals — table + filters by strategy, confidence, ticker
- [ ] Tab: Paper Portfolio — open positions, unrealized P&L, entry dates
- [ ] Tab: Performance — equity curve, per-strategy P&L, win rate, Sharpe, drawdown
- [ ] Tab: Backtests — historical strategy performance, equity curves, stats tables
- [ ] Tab: Settings — config viewer/editor (read-only initially, editable later)
- [ ] Auto-refresh on new data
- [ ] **MILESTONE:** Dashboard running with all tabs populated from live data

## Phase 4 — Long-Term Investment Scanner

- [ ] Expand fundamentals ingestion: P/E, P/B, PEG, dividend yield, earnings growth, payout ratio
- [ ] Build scoring model: valuation + technical accumulation + dividend + earnings quality
- [ ] Add 52-week high/low context and historical valuation percentiles
- [ ] Generate ranked watchlist, write to DB
- [ ] Build weekly digest email with top opportunities + reasoning
- [ ] Dashboard tab: Long-Term Watchlist with scores, fundamentals, and charts
- [ ] **MILESTONE:** Weekly long-term opportunity digest arriving by email

## Phase 5 — Trade Tracker (Real Trades)

- [ ] Add manual trade entry to dashboard (ticker, direction, price, date, size, strategy)
- [ ] Track real position P&L using live prices
- [ ] Side-by-side comparison: paper vs. real performance
- [ ] CSV export of trade history
- [ ] **MILESTONE:** Can log real trades and see comparative P&L vs. paper

## Phase 6 — Hardening

- [ ] Comprehensive error handling and retry logic on all data pulls
- [ ] Data source fallback (yfinance → Alpha Vantage → manual)
- [ ] Walk-forward validation for all strategies (no in-sample-only backtests)
- [ ] System health monitoring (last successful data pull, signal generation status)
- [ ] Graceful handling of market holidays, half days, after-hours
- [ ] Strategy auto-disable on sustained drawdown (configurable threshold)
- [ ] Run unattended for 2+ weeks without intervention
- [ ] **MILESTONE:** Production-grade reliability

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

---

## Notes for Claude Code

- Martin is a data scientist. He thinks in terms of experimentation and statistical rigor. Every strategy needs backtested evidence before going live.
- Charts must be publication-quality: white background, large fonts, clear labels. No default styling. Ever.
- Martin prefers to talk through design before coding. Ask before building if requirements are ambiguous.
- Use type hints. Write docstrings. Keep functions small and testable.
- Config-driven everything. No magic numbers in strategy code.
