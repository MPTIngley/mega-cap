# StockPulse Command Reference

## Quick Setup (One-Time)

Add this to your `~/.zshrc` for easy access from any terminal:

```bash
echo 'stockpulse() {
  cd ~/Documents/AIGames/mega-cap &&
  source venv/bin/activate &&
  set -a && source .env && set +a &&
  python3 -m stockpulse.main "$@"
}' >> ~/.zshrc
source ~/.zshrc
```

Then use `stockpulse <command>` from anywhere.

---

## Main Commands

| Command | What it does |
|---------|--------------|
| `stockpulse dashboard` | Launch the web dashboard (http://localhost:8501) |
| `stockpulse run` | Start the scheduler (runs scans every 15 min during market hours) |
| `stockpulse scan` | Run a single signal scan now |
| `stockpulse backtest` | Run backtests on all strategies |
| `stockpulse init` | Initialize DB and fetch 2 years of historical data |
| `stockpulse ingest` | Refresh universe and fetch latest price data |

---

## Manual Setup (Per Terminal)

If you don't set up the alias, run these in each new terminal:

```bash
cd ~/Documents/AIGames/mega-cap
source venv/bin/activate
set -a && source .env && set +a
python3 -m stockpulse.main <command>
```

---

## Database Management

**Wipe everything and start fresh:**
```bash
rm data/stockpulse.duckdb
stockpulse init
```

**Clear just positions/signals (keep price history):**
```bash
cd ~/Documents/AIGames/mega-cap
source venv/bin/activate
python3 -c "
from stockpulse.data.database import get_db
db = get_db()
db.execute('DELETE FROM positions_paper')
db.execute('DELETE FROM signals')
print('Positions and signals cleared')
"
```

**Check for locked database:**
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

## Git Commands

**Pull latest changes:**
```bash
cd ~/Documents/AIGames/mega-cap
git pull origin claude/init-repo-setup-maaOL
```

**Check status:**
```bash
git status
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named 'stockpulse'` | Activate venv: `source venv/bin/activate` |
| `database is locked` | Kill stuck process: `ps aux \| grep python` then `kill <PID>` |
| `Email not configured` | Load env vars: `set -a && source .env && set +a` |
| Dashboard shows 0 stocks | Run `stockpulse init` to fetch data |
