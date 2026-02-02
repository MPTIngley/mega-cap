# PRD: StockPulse — Automated Stock Scanning & Trading Signal System

**Owner:** Martin
**Status:** Draft v1.0
**Last Updated:** 2026-02-01

---

## CLAUDE CODE INSTRUCTION

**Every session, start by reading `PROJECT_PLAN.md` in the project root.** Check off completed items, flag blockers, and confirm what we're working on before writing any code. No exceptions.

---

## 1. Vision

A self-hosted system that scans the 100 largest US stocks by market cap every 15 minutes, runs a suite of short-term trading strategies (2–30 day holding periods), surfaces long-term investment opportunities, sends email alerts, and presents everything through a Streamlit dashboard. Starts as a paper-trading system with a path to tracking real trades.

## 2. Goals

- **Make money.** Generate actionable, high-conviction short-term trade signals that outperform buy-and-hold on a risk-adjusted basis.
- **Spot long-term value.** Flag stocks entering attractive long-term accumulation zones.
- **Stay informed.** Email alerts so Martin never misses a signal.
- **Track performance.** Dashboard showing strategy P&L, win rates, and open positions — paper and eventually real.
- **Learn and iterate.** Backtest-first development so every strategy ships with historical evidence.

## 3. Non-Goals (for now)

- Automated order execution / broker integration
- Options or derivatives
- International stocks
- High-frequency trading (sub-minute)
- Real-money tracking (Phase 2)

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   SCHEDULER (cron/APScheduler)       │
│              Runs every 15 min during market hours   │
│              + daily post-close batch                │
└──────────────┬──────────────────────┬────────────────┘
               │                      │
       ┌───────▼───────┐    ┌────────▼─────────┐
       │  DATA INGESTION│    │  DAILY BATCH      │
       │  (price, vol)  │    │  (fundamentals,   │
       │  Yahoo Finance │    │   technicals,     │
       │  or yfinance   │    │   long-term scan) │
       └───────┬───────┘    └────────┬─────────┘
               │                      │
       ┌───────▼──────────────────────▼────────┐
       │            LOCAL DATABASE              │
       │         (SQLite or DuckDB)             │
       └───────────────┬───────────────────────┘
                       │
          ┌────────────▼────────────┐
          │     STRATEGY ENGINE     │
          │  Short-term strategies  │
          │  Long-term scanner      │
          │  Signal generation      │
          └────────────┬────────────┘
                       │
            ┌──────────▼──────────┐
            │   ALERT / ACTION    │
            │  Email (SMTP/SES)   │
            │  Signal log to DB   │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  STREAMLIT DASHBOARD│
            │  Signals, P&L,      │
            │  portfolio tracker  │
            └─────────────────────┘
```

## 5. Components

### 5.1 Universe Definition

- Top 100 US stocks by market cap
- Refresh the universe list weekly (companies move in/out)
- Store as a config/table so it's easy to override manually

### 5.2 Data Ingestion

- **Source:** `yfinance` Python library (free, no API key) as primary. Fallback: Alpha Vantage or Polygon.io free tier.
- **Intraday:** OHLCV at 15-min intervals during market hours (9:30 AM – 4:00 PM ET, Mon–Fri)
- **Daily:** End-of-day OHLCV, fundamentals (P/E, P/B, dividend yield, earnings dates) pulled once after close
- **Storage:** Local database (DuckDB preferred for analytical queries, SQLite as fallback)
- **Rate limiting:** Respect API limits. Batch requests. Cache aggressively.

### 5.3 Short-Term Trading Strategies (2–30 day hold)

Each strategy must be implemented with:
1. A **backtesting module** showing historical performance before going live
2. Clear **entry signal**, **exit signal** (target + stop-loss), and **position sizing** logic
3. A **confidence score** (0–100) for each signal

**Initial strategy candidates** (prioritize by evidence strength):

| # | Strategy | Core Idea | Typical Hold |
|---|----------|-----------|--------------|
| 1 | Mean Reversion (RSI) | Buy oversold (RSI < 30), sell overbought (RSI > 70) on daily bars | 3–10 days |
| 2 | Bollinger Band Squeeze Breakout | Enter on volatility expansion after squeeze | 2–15 days |
| 3 | MACD Crossover + Volume Confirmation | Trend-following with volume filter | 5–20 days |
| 4 | Earnings Drift | Post-earnings momentum in direction of surprise | 5–30 days |
| 5 | Gap-and-Go | Buy large gap-ups with volume on first 15-min bar hold | 2–5 days |
| 6 | Moving Average Crossover (10/50 EMA) | Classic trend signal filtered by broader trend | 10–30 days |

**Ensemble layer:** When multiple strategies agree on a signal, boost confidence. Dashboard shows which strategies are firing and whether they align.

### 5.4 Long-Term Investment Scanner

Separate module. Runs once daily after market close. Flags opportunities based on:

- **Valuation:** P/E, P/B, PEG ratio relative to sector and own history
- **Technical accumulation:** Price near 52-week low + rising OBV / accumulation-distribution
- **Dividend yield:** Unusually high yield vs. own 5-year average (potential value trap filter needed)
- **Earnings quality:** Consistent earnings growth + reasonable payout ratio
- **Macro context:** Sector rotation signals (optional, Phase 2)

Output: Weekly digest email + dashboard tab with ranked opportunities and reasoning.

### 5.5 Email Alerts

- **Transport:** SMTP (Gmail app password) or AWS SES
- **Trigger conditions:**
  - New short-term BUY signal with confidence ≥ 70
  - New short-term SELL/EXIT signal on an open position
  - Long-term opportunity flagged (daily digest)
  - System errors or data staleness warnings
- **Format:** Clean HTML email with ticker, strategy, confidence, entry/target/stop, and a mini chart (optional)
- **Configurable:** Alert thresholds, quiet hours, per-strategy toggles

### 5.6 Streamlit Dashboard

**Tabs:**

1. **Live Signals** — Current active signals across all strategies. Filterable by strategy, confidence, ticker.
2. **Portfolio (Paper)** — Open paper positions, unrealized P&L, entry date, strategy source.
3. **Performance** — Strategy-level and aggregate P&L, win rate, Sharpe, max drawdown. Time-series equity curve.
4. **Long-Term Watchlist** — Scored opportunities with fundamental data and charts.
5. **Backtests** — Historical performance of each strategy. Equity curves, drawdowns, stats.
6. **Settings** — Email config, strategy toggles, confidence thresholds, universe overrides.

**Charts:** Publication-quality. White backgrounds, large readable fonts, clean axes. No default matplotlib garbage. Use Plotly for interactivity where useful, otherwise Altair or matplotlib with a clean custom theme.

### 5.7 Trade Tracker (Phase 2)

- Manual trade entry: ticker, direction, entry price, entry date, size, strategy source
- Auto-track P&L using live price data
- Compare real vs. paper performance
- Export to CSV

---

## 6. Data Model (core tables)

```
universe          — ticker, company_name, sector, market_cap, last_refreshed
prices_intraday   — ticker, timestamp, open, high, low, close, volume
prices_daily      — ticker, date, open, high, low, close, adj_close, volume
fundamentals      — ticker, date, pe, pb, peg, div_yield, earnings_date, ...
signals           — id, timestamp, ticker, strategy, direction, confidence,
                     entry_price, target_price, stop_price, status (open/closed/expired)
positions_paper   — id, signal_id, ticker, entry_price, entry_date, exit_price,
                     exit_date, pnl, status
positions_real    — (Phase 2, same schema + size, fees)
strategy_config   — strategy_name, enabled, params (JSON)
alerts_log        — id, timestamp, signal_id, email_sent, error
```

## 7. Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python 3.11+ | Ecosystem, yfinance, pandas, scikit-learn |
| Data | DuckDB | Fast analytical queries, zero config, single file |
| Scheduling | APScheduler | In-process, lightweight, cron-like |
| Strategies | pandas + ta-lib (or pandas-ta) | Standard technical analysis |
| Backtesting | vectorbt or custom | Fast vectorized backtests |
| Dashboard | Streamlit | Rapid iteration, good enough for personal use |
| Charts | Plotly + custom Streamlit theme | Interactive, clean |
| Email | smtplib + email.mime | Simple, no dependencies |
| Config | YAML or TOML file | Human-readable, easy to edit |

## 8. Configuration

All in a single `config.yaml`:

```yaml
email:
  smtp_server: smtp.gmail.com
  smtp_port: 587
  sender: ""
  recipient: ""
  # password via environment variable STOCKPULSE_EMAIL_PASSWORD

scanning:
  interval_minutes: 15
  market_open: "09:30"
  market_close: "16:00"
  timezone: "US/Eastern"

strategies:
  mean_reversion_rsi:
    enabled: true
    rsi_oversold: 30
    rsi_overbought: 70
    min_confidence: 60
  bollinger_squeeze:
    enabled: true
    squeeze_lookback: 20
    ...

alerts:
  min_confidence_for_email: 70
  daily_digest_time: "17:00"
  quiet_hours: ["22:00", "07:00"]

universe:
  source: "auto"  # or "manual"
  manual_tickers: []  # override list
```

## 9. Risk & Limitations

- **Not financial advice.** This is a personal research tool. All trading decisions are Martin's.
- **Data quality:** `yfinance` is unofficial and can break. Build in staleness detection and fallback sources.
- **Overfitting:** Backtest results ≠ future performance. Use walk-forward validation, not just in-sample.
- **Rate limits:** yfinance hammering can get IP-blocked. Use caching and respect intervals.
- **No real execution.** Paper trading only until Phase 2. Even then, manual entry only — no broker API.

## 10. Success Metrics

- Paper portfolio Sharpe ratio > 1.0 over 3-month evaluation period
- Signal precision (% of signals that hit target before stop) > 55%
- System uptime > 99% during market hours
- Alert delivery latency < 2 minutes from signal generation
- Martin actually uses it and finds it useful

---

## 11. Open Questions

- [ ] Which email account for alerts?
- [ ] Hosting: local machine, home server, or cloud (EC2/GCP)?
- [ ] Budget for paid data APIs if yfinance proves unreliable?
- [ ] Any specific stocks Martin always wants in the universe regardless of market cap?
- [ ] Risk tolerance parameters: max position size, max portfolio exposure, max drawdown before strategy auto-disable?

---

## 12. Phasing

| Phase | Scope | Definition of Done |
|-------|-------|--------------------|
| **0 — Foundation** | Project structure, DB, data ingestion, universe management | 15-min price data flowing into DB for 100 stocks |
| **1 — Strategies** | Implement + backtest first 3 strategies, signal generation | Backtest reports for each strategy, signals writing to DB |
| **2 — Alerts** | Email pipeline, alert triggers, formatting | Receiving email alerts on new signals |
| **3 — Dashboard** | Streamlit app: signals, paper portfolio, performance, backtests | Running dashboard with live data |
| **4 — Long-Term Scanner** | Fundamental data ingestion, scoring model, weekly digest | Weekly long-term opportunity emails |
| **5 — Trade Tracker** | Manual trade entry, real vs. paper comparison | Can log real trades and see comparative P&L |
| **6 — Hardening** | Error handling, monitoring, data fallbacks, walk-forward validation | System runs unattended for 2+ weeks without intervention |
