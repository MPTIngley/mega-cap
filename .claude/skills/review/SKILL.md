# /review — Full Adversarial Review of StockPulse

## Description

Runs a full adversarial review of the StockPulse project using 6 SME personas who evaluate the dashboard, data pipeline, scoring system, and overall product from different perspectives.

## Instructions

When the user invokes `/review`, execute the following steps:

### Step 1: Screenshot Dashboard Pages

Use Playwright MCP to capture 5 dashboard pages:

1. Navigate to `http://localhost:8501` (AI Stocks page - default)
2. Take screenshot of AI Stocks page
3. Navigate to Trillion Club page via sidebar
4. Take screenshot of Trillion Club page
5. Navigate to AI Theses page via sidebar
6. Take screenshot of AI Theses page
7. Navigate to Portfolio page via sidebar (if exists)
8. Take screenshot of Portfolio page
9. Navigate to Settings page via sidebar
10. Take screenshot of Settings page

If the dashboard is not running, read the key files instead:

- `src/stockpulse/dashboard/app.py`
- `src/stockpulse/dashboard/charts.py`
- `src/stockpulse/data/sentiment.py`
- `src/stockpulse/alerts/alert_manager.py`
- `config/config.yaml`

### Step 2: Spawn 6 Review Agents in Parallel

Launch 6 Task agents simultaneously using Haiku model, each with their persona loaded:

For each agent, provide:

- The agent's persona file content (from `.claude/agents/review-*.md`)
- Screenshots of all 5 pages (or file contents if no dashboard)
- Brief context: "You are reviewing StockPulse, a personal stock analysis tool with sentiment scoring, AI-powered thesis research, and automated email digests."

Agent prompts should be:

```
You are [PERSONA NAME]. Review the StockPulse dashboard and system.

[PERSONA FILE CONTENT]

Here are screenshots of the 5 main dashboard pages. Also consider:
- The system pulls sentiment from StockTwits, Reddit, Google News, Finnhub, analyst ratings, insider transactions, Google Trends, options data, and Wikipedia page views
- Scores are weighted aggregates (configurable)
- Emails go out daily with AI Pulse digest, Long-Term opportunities, and Trillion Club updates
- A council of 4 AI agents (analyst, skeptic, quant, philosopher) researches investment theses

Provide your assessment following your output format. Max 400 words. Be specific and actionable.
```

### Step 3: Synthesize into Brief

After all 6 agents return, synthesize their feedback into a structured brief:

```
# StockPulse Review Brief

## Consensus Items (agreed by 4+ agents)
- [items most agents flagged]

## Key Tensions
- [where agents disagreed and why]

## Quick Wins (high impact, low effort)
1. [recommendation] — Flagged by: [agent names]
2. ...

## Risk Flags
- [items flagged as risks by Risk Manager or Devil's Advocate]

## Prioritized Improvements
| Priority | Improvement | Effort | Impact | Flagged By |
|----------|-------------|--------|--------|------------|
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |

## Individual Assessments
[Include each agent's full response in an expander/section]
```

### Step 4: Save Results

Write the review brief to `tasks/review-{date}.md`.

## User-Invocable

This skill can be invoked by the user with `/review`.
