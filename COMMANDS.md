# StockPulse Command Reference

## How StockPulse Works

**You need TWO processes running for full operation:**

| Process | Terminal | Purpose |
|---------|----------|---------|
| `stockpulse run` | Terminal 1 | Scheduler - fetches data, generates signals, manages positions, sends email alerts |
| `stockpulse dashboard` | Terminal 2 | Web UI - view signals, portfolio, performance (http://localhost:8501) |

The scheduler writes to the database. The dashboard reads from it. Both can run simultaneously.

**Without the scheduler running:**
- Dashboard shows stale data
- No new signals generated
- No email alerts sent
- Positions not updated

---

## Quick Setup (One-Time)

Add this to your `~/.zshrc` for easy access from any terminal:

```bash
echo 'stockpulse() {
  cd ~/Documents/AIGames/mega-cap &&
  source venv/bin/activate &&
  python3 -m stockpulse.main "$@"
}' >> ~/.zshrc
source ~/.zshrc
```

Then use `stockpulse <command>` from anywhere.

**Note:** Commands auto-load `.env` - no manual env sourcing needed.

---

## Process Management

### Starting Processes

```bash
stockpulse run         # Start scheduler (Terminal 1)
stockpulse dashboard   # Start dashboard (Terminal 2)
```

### Stopping Processes

| Method | Command | When to use |
|--------|---------|-------------|
| **Graceful stop** | `Ctrl+C` | Normal shutdown (recommended) |
| **Find processes** | `ps aux \| grep stockpulse` | See what's running |
| **Kill by PID** | `kill <PID>` | Stop a specific process |
| **Force kill** | `kill -9 <PID>` | Process won't stop normally |
| **Kill all Python** | `pkill -f stockpulse` | Stop all stockpulse processes |

### Quick Commands

```bash
# See what's running
ps aux | grep stockpulse

# Kill everything stockpulse-related
pkill -f stockpulse

# Kill stuck Streamlit dashboard specifically
pkill -f streamlit

# Kill stuck scheduler specifically
pkill -f "stockpulse.main run"
```

---

## Main Commands

| Command | What it does |
|---------|--------------|
| `stockpulse dashboard` | Launch the web dashboard (http://localhost:8501) |
| `stockpulse run` | Start the scheduler (runs all scans on schedule) |
| `stockpulse scan` | Run a single signal scan now |
| `stockpulse backtest` | Run backtests on all strategies |
| `stockpulse optimize` | Run hyperparameter optimization for all 8 strategies |
| `stockpulse reset` | Reset trading data (keeps historical prices) |
| `stockpulse reset --clear-all` | Reset ALL data including historical prices |
| `stockpulse init` | Initialize DB and fetch 2 years of historical data |
| `stockpulse ingest` | Refresh universe and fetch latest price data |

## Long-Term & Research Commands

| Command | What it does |
|---------|--------------|
| `stockpulse longterm-scan` | Run 8-component value scoring on all stocks + email |
| `stockpulse trillion-scan` | Run Trillion+ Club mega-cap scanner + email |
| `stockpulse ai-scan` | Run AI Pulse scan (~70 AI stocks) + Claude thesis research + email |
| `stockpulse longterm-backfill` | Build 6 weeks of historical scan data for trends |
| `stockpulse ai-backfill` | Initialize trillion club history and default theses |

### AI Pulse Categories

The `ai-scan` covers 7 AI investment categories (~70 stocks):

| Category | Focus | Example Tickers |
|----------|-------|-----------------|
| AI Infrastructure | GPUs, chips, data centers, power | NVDA, AMD, TSM, ASML, SMCI, CEG |
| Hyperscaler | Cloud giants | MSFT, AMZN, GOOGL, ORCL, BABA |
| AI Software | Platforms, enterprise AI, cybersecurity | PLTR, CRM, NOW, CRWD, PANW |
| Robotics/Physical AI | Autonomous systems, industrial | TSLA, ISRG, HON, DE, ABB |
| AI Edge/Consumer | On-device AI, consumer tech | AAPL, QCOM, SONY, LOGI |
| AI Healthcare | Drug discovery, diagnostics | RXRX, VEEV, ILMN, TMO |
| Neocloud | AI-native cloud providers | CRWV |

Each stock gets an **AI Score (0-100)** based on:
- 30/90-day price performance (pullbacks = opportunities)
- AI category positioning (Infrastructure gets highest weight)
- Technical setup (RSI, 50-day MA)
- Valuation (PEG ratio, P/E)
- Market cap tier

---

## Strategies (8 total)

| Strategy | Type | Signal |
|----------|------|--------|
| `rsi_mean_reversion` | Mean Reversion | RSI oversold/overbought |
| `bollinger_squeeze` | Breakout | Bollinger Band squeeze |
| `macd_volume` | Momentum | MACD crossover + volume |
| `zscore_mean_reversion` | Mean Reversion | Price Z-score extremes |
| `momentum_breakout` | Momentum | Price/volume breakout |
| `gap_fade` | Mean Reversion | Overnight gap fill |
| `week52_low_bounce` | Value | Bounce from 52-week low |
| `sector_rotation` | Momentum | Buy leaders in hot sectors |

---

## Typical Workflow

**First time setup:**
```bash
stockpulse init
```

**Optimize strategy parameters:**
```bash
stockpulse optimize
```

**Daily operation (two terminals):**
```bash
stockpulse run
stockpulse dashboard
```

**Fresh start (reset trading data, keep historical prices):**
```bash
stockpulse reset              # Clears trades/signals, KEEPS 2 years of price data
stockpulse optimize           # Re-optimize with fresh slate
git add config/config.yaml
git commit -m "Optimized strategy params"
git push origin claude/init-repo-setup-maaOL
```

**Full reset (delete everything including historical data):**
```bash
stockpulse reset --clear-all  # Deletes ALL data including prices
stockpulse init               # Re-fetch 2 years of historical data
stockpulse optimize
```

**When will you see trades?**
- Signals generate during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
- Scheduler scans every 15 minutes
- Alerts sent via email when high-confidence signals found

---

## Git Workflow

### Pull Latest Changes

Always pull before starting work:
```bash
cd ~/Documents/AIGames/mega-cap
git pull origin claude/init-repo-setup-maaOL
```

### Check Current Status
```bash
git status
git log --oneline -5   # See recent commits
```

### Push to Feature Branch
```bash
git add .
git commit -m "Your commit message"
git push origin claude/init-repo-setup-maaOL
```

### Merge to Main (when ready for production)

**Option 1: Fast-forward merge (clean history)**
```bash
git checkout main
git pull origin main
git merge claude/init-repo-setup-maaOL
git push origin main
```

**Option 2: Create a Pull Request (recommended for review)**
```bash
# Push your branch first
git push origin claude/init-repo-setup-maaOL

# Then create PR on GitHub or via CLI:
gh pr create --base main --head claude/init-repo-setup-maaOL --title "Your PR title"
```

### Start Fresh Branch
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

---

## Manual Setup (Per Terminal)

If you don't set up the alias, run these in each new terminal:

```bash
cd ~/Documents/AIGames/mega-cap
source venv/bin/activate
python3 -m stockpulse.main <command>
```

The `.env` file is automatically loaded by the application.

---

## Database Management

**Clear trades but keep market data (prices, universe):**
```bash
stockpulse reset
# Deletes: signals, positions, alerts, backtest_results
# Keeps: prices_daily, prices_intraday, fundamentals, universe (2 years of data!)
```

**Clear EVERYTHING including market data (requires re-init):**
```bash
stockpulse reset --clear-all
stockpulse init  # Re-fetches 2 years of historical data
```

**Alternative: Delete database files directly:**
```bash
rm -f data/stockpulse.sqlite data/stockpulse.sqlite-wal data/stockpulse.sqlite-shm
stockpulse init  # Re-fetches everything
```

**Get data summary:**
```bash
cd ~/Documents/AIGames/mega-cap && source venv/bin/activate
python3 -c "from stockpulse.data.database import get_data_summary; import json; print(json.dumps(get_data_summary(), indent=2, default=str))"
```

**Check for stuck Python processes:**
```bash
ps aux | grep python | grep -v grep
# Kill any stuck process:
kill <PID>
```

---

## Environment Variables (.env file)

Located at `~/Documents/AIGames/mega-cap/.env`:

```
STOCKPULSE_EMAIL_SENDER=your.email@gmail.com
STOCKPULSE_EMAIL_RECIPIENT=your.email@gmail.com
STOCKPULSE_EMAIL_PASSWORD=your-16-char-app-password
STOCKPULSE_INITIAL_CAPITAL=100000
STOCKPULSE_POSITION_SIZE_PCT=5.0
STOCKPULSE_MAX_POSITIONS=20
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named 'stockpulse'` | Activate venv: `source venv/bin/activate` |
| `Email not configured` | Check `.env` file has correct values |
| Dashboard shows 0 stocks | Run `stockpulse init` to fetch data |
| No signals appearing | Make sure `stockpulse run` is running during market hours |
| Stale dashboard data | Refresh browser; ensure scheduler is running |
| Want to reset trades | Use `reset_trading_data()` - see Database Management above |
| Database locked error | Use SQLite (already configured); only one writer allowed |
| `unrecognized arguments` error | Don't use inline `#` comments in shell commands (see note below) |

### Shell Command Note

**Do NOT use inline comments when running commands.** This will fail:

```bash
# WRONG - the # comment breaks the command
stockpulse longterm-scan   # Long-term opportunities
```

Use clean commands instead:

```bash
# CORRECT - comment on separate line, or no comment
stockpulse longterm-scan
stockpulse trillion-scan
stockpulse ai-scan
```

For multi-command runs:
```bash
stockpulse longterm-scan && stockpulse trillion-scan && stockpulse ai-scan
```

---

## Schedule (When Things Run)

| Job | Time (ET) | Frequency | Email |
|-----|-----------|-----------|-------|
| Intraday scan | 9:30 AM - 4:00 PM | Every 15 minutes | Signal alerts |
| Daily summary | 4:30 PM | Once per day (Mon-Fri) | Portfolio digest |
| Daily digest | 5:00 PM | Once per day (Mon-Fri) | Trading summary |
| Long-term scan | 5:30 PM | Once per day (Mon-Fri) | Value opportunities |
| Trillion+ Club | 5:31 PM | Once per day (Mon-Fri) | Mega-cap entry points |
| AI Thesis | 5:32 PM | Once per day (Mon-Fri) | Research insights |

Outside market hours, the scheduler runs but skips intraday scans.

## Dashboard Tabs

| Tab | What it shows |
|-----|---------------|
| Live Signals | Real-time trading signals from all 8 strategies |
| Paper Portfolio | Simulated trades and P&L tracking |
| Long-Term Holdings | Your actual investment positions |
| Performance | Strategy performance metrics and analytics |
| Backtests | Historical strategy backtesting results |
| Long-Term Watchlist | 8-component value scoring opportunities |
| Trillion Club | Mega-cap stocks ($1T+) with entry scores |
| AI Theses | Investment thesis research from Claude |
| Universe | All stocks being tracked |
| Settings | Configuration and system status |

---

## Systemd Deployment (Linux Server)

Run StockPulse as persistent background services that auto-restart on failure and start on boot.

### Initial Setup

```bash
sudo cp deploy/stockpulse.service /etc/systemd/system/
sudo cp deploy/stockpulse-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable stockpulse stockpulse-dashboard
```

### Service Management

| Action | Command |
|--------|---------|
| **Start both** | `sudo systemctl start stockpulse stockpulse-dashboard` |
| **Stop both** | `sudo systemctl stop stockpulse stockpulse-dashboard` |
| **Restart both** | `sudo systemctl restart stockpulse stockpulse-dashboard` |
| **Check status** | `sudo systemctl status stockpulse stockpulse-dashboard` |
| **View logs** | `sudo journalctl -u stockpulse -f` |
| **View dashboard logs** | `sudo journalctl -u stockpulse-dashboard -f` |

### Quick Commands

```bash
# Start everything
sudo systemctl start stockpulse stockpulse-dashboard

# Stop everything
sudo systemctl stop stockpulse stockpulse-dashboard

# Check if running
sudo systemctl is-active stockpulse stockpulse-dashboard

# View live scheduler logs
sudo journalctl -u stockpulse -f

# View last 100 lines of logs
sudo journalctl -u stockpulse -n 100

# View logs from application log files
tail -f logs/stockpulse.log
tail -f logs/dashboard.log
```

### Resilience Features

The systemd services include:

- **Auto-restart**: Restarts within 10 seconds if process dies
- **Restart limits**: Max 10 restarts per 10 minutes (prevents crash loops)
- **Memory limits**: Scheduler 2GB, Dashboard 1GB max
- **CPU limits**: Scheduler 80%, Dashboard 50% max
- **Graceful shutdown**: SIGTERM with timeout before SIGKILL
- **Boot persistence**: Starts automatically on system boot

### Disable Services

```bash
# Stop and disable (won't start on boot)
sudo systemctl stop stockpulse stockpulse-dashboard
sudo systemctl disable stockpulse stockpulse-dashboard
```
