# PROJECT_PLAN.md — StockPulse

## CLAUDE CODE: READ THIS FIRST EVERY SESSION

Before writing any code, do the following:
1. Read this file completely
2. Check which phase we're in and what's next
3. Confirm with Martin what we're working on today
4. After completing work, update the checkboxes below and note the date

---

## Current Phase: **6 — Live Paper Trading (ACTIVE)**
## Last Session: 2026-02-03
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
- ✅ Auto-open/close paper positions
- ✅ P&L tracking with transaction costs

### Email Schedule:
| Time (ET) | Email | Content |
|-----------|-------|---------|
| 17:00 | Daily Portfolio Digest | Portfolio value, P&L, positions, activity |
| 17:30 | Long-Term Opportunities | Value stocks near 52-week lows |
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
stockpulse run          # Start scheduler (continuous)
stockpulse scan         # One-off scan
stockpulse digest       # Send portfolio digest now
stockpulse test-email   # Test email config
stockpulse dashboard    # Launch Streamlit
stockpulse reset        # Clear trading data (keeps prices)
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
