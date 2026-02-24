---
name: scans
description: This skill should be used when the user says "/scans" or asks to run all StockPulse scans. It runs the full scan pipeline.
version: 0.1.0
---

# Scans Skill

Run the complete StockPulse scan pipeline in sequence.

## Commands

Execute these commands in order, stopping if any fails:

1. `stockpulse ingest` - Fetch latest market data
2. `stockpulse longterm-scan` - Run long-term analysis
3. `stockpulse trillion-scan` - Run trillion club scan
4. `stockpulse ai-scan` - Run AI-powered analysis

## Notes

- Run commands sequentially since each depends on fresh data from the previous step
- Report timing for each scan
- If any scan fails, report the error and stop
- Show a brief summary of results after all scans complete
