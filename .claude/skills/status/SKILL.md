---
name: status
description: This skill should be used when the user says "/status" or asks for system status. It checks services, logs, and git state.
version: 0.1.0
---

# Status Skill

Check the current state of StockPulse services, logs, and git.

## Steps

Run these checks and present a concise summary:

1. **Service status**: `launchctl list | grep stockpulse`
2. **Recent logs**: Read the last 20 lines of `logs/stockpulse.log`
3. **Dashboard logs**: Read the last 10 lines of `logs/dashboard.log`
4. **Git status**: `git status --short` and `git log --oneline -5`
5. **Process check**: `ps aux | grep stockpulse | grep -v grep`

## Output Format

Present results in a compact table/summary. Flag any issues:
- Services not running
- Errors in recent logs
- Uncommitted changes
- Unusual process state
