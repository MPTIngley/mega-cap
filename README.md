# StockPulse

Automated Stock Scanning & Trading Signal System

A self-hosted system that scans the top 100 US stocks by market cap, runs trading strategies, generates signals with email alerts, and presents everything through a Streamlit dashboard.

## Features

- **5 Trading Strategies**: RSI Mean Reversion, Bollinger Squeeze, MACD+Volume, Z-Score Mean Reversion, Momentum Breakout
- **Backtesting Framework**: Test strategies with historical data including transaction costs
- **Email Alerts**: Get notified of new signals and position exits
- **Streamlit Dashboard**: Monitor signals, positions, and performance
- **Long-Term Scanner**: Identifies value investment opportunities
- **Real Trade Tracker**: Track and compare real vs paper performance

## Quick Start

### 1. Install Dependencies

```bash
cd mega-cap
pip install -e ".[dev]"
```

### 2. Configure Email (for alerts)

Copy the example and edit:

```bash
cp .env.example .env
open -e .env   # or use any text editor
```

Set these values in `.env`:

```
STOCKPULSE_EMAIL_SENDER=your-email@gmail.com
STOCKPULSE_EMAIL_RECIPIENT=your-email@gmail.com
STOCKPULSE_EMAIL_PASSWORD=your-gmail-app-password
```

**Note:** Commands auto-load `.env` - no manual sourcing required.

### 3. Initialize Database and Fetch Data

```bash
python -m stockpulse.main init
```

### 4. Run the Dashboard

```bash
python -m stockpulse.main dashboard
```

Or run the full scheduler:

```bash
python -m stockpulse.main run
```

## Commands

| Command | Description |
|---------|-------------|
| `stockpulse init` | Initialize database and fetch initial data |
| `stockpulse run` | Run the scheduler for continuous scanning |
| `stockpulse dashboard` | Launch Streamlit dashboard |
| `stockpulse backtest` | Run backtests for all strategies |
| `stockpulse ingest` | Manually run data ingestion |
| `stockpulse scan` | Run a single scan |

## Configuration

Edit `config/config.yaml` to customize:

- Strategy parameters
- Alert thresholds
- Risk management settings
- Scanning intervals

## API Keys & Tokens Required

### Required for Email Alerts

1. **Gmail App Password** (for sending email alerts)
   - Go to Google Account > Security > 2-Step Verification
   - Scroll to "App passwords"
   - Generate a new app password for "Mail"
   - Use this password for `STOCKPULSE_EMAIL_PASSWORD`

### No API Key Required

- **yfinance**: Stock data is fetched using yfinance (free, no API key)
- **SQLite**: Local database with WAL mode (concurrent read/write support)

## Project Structure

```
mega-cap/
├── config/
│   └── config.yaml          # Configuration file
├── data/
│   └── stockpulse.sqlite    # Database (created on init)
├── src/stockpulse/
│   ├── alerts/              # Email alert system
│   ├── dashboard/           # Streamlit dashboard
│   ├── data/                # Database and data ingestion
│   ├── scanner/             # Long-term investment scanner
│   ├── strategies/          # Trading strategies and backtesting
│   ├── tracker/             # Real trade tracker
│   └── main.py              # CLI entry point
├── tests/                   # Unit tests
├── PRD.md                   # Product Requirements Document
├── PROJECT_PLAN.md          # Implementation plan
└── pyproject.toml           # Project dependencies
```

## Strategies

### 1. RSI Mean Reversion
Buy oversold (RSI < 30), sell overbought (RSI > 70). Best in range-bound markets.

### 2. Bollinger Squeeze Breakout
Enter on volatility expansion after a squeeze period. Volume confirmation required.

### 3. MACD + Volume
Classic trend-following with MACD crossovers, filtered by above-average volume.

### 4. Z-Score Mean Reversion
Statistical mean reversion when price deviates significantly from recent mean.

### 5. Momentum Breakout
Breakouts above recent highs with volume confirmation for momentum plays.

## Transaction Costs

The system accounts for realistic transaction costs:
- **Commission**: $0 (most brokers)
- **Slippage**: 0.05% (5 bps)
- **Spread**: 0.02% (2 bps)
- **Total round-trip**: ~0.14%

## Risk Management

Configurable limits:
- Max 5% portfolio per position
- Max 20 concurrent positions
- Max 80% portfolio exposure
- Auto-disable on 15% drawdown

## Disclaimer

This is a personal research tool, not financial advice. All trading decisions are your own. Paper trade first before risking real money.

## License

MIT
