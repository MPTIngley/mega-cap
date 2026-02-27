# /review-page — Quick Review of a Single Dashboard Page

## Description

Quick variant of `/review` that evaluates a single dashboard page using the 3 most relevant agents. Faster and cheaper than a full review.

## Arguments

- `<page>`: The page to review. Options: `ai-stocks`, `trillion`, `theses`, `portfolio`, `settings`

## Instructions

When the user invokes `/review-page <page>`, execute the following steps:

### Step 1: Screenshot the Page

Use Playwright MCP to capture the specified page:

- `ai-stocks`: Navigate to `http://localhost:8501` (default page)
- `trillion`: Navigate to Trillion Club via sidebar
- `theses`: Navigate to AI Theses via sidebar
- `portfolio`: Navigate to Portfolio via sidebar
- `settings`: Navigate to Settings via sidebar

If the dashboard is not running, read the relevant code sections instead.

### Step 2: Select 3 Most Relevant Agents

Based on the page type, pick the 3 most relevant agents:

| Page        | Agent 1         | Agent 2          | Agent 3        |
| ----------- | --------------- | ---------------- | -------------- |
| `ai-stocks` | UX Designer     | Retail Investor  | Quant PM       |
| `trillion`  | Retail Investor | Risk Manager     | UX Designer    |
| `theses`    | Quant PM        | Devil's Advocate | Data Scientist |
| `portfolio` | Risk Manager    | Retail Investor  | Quant PM       |
| `settings`  | UX Designer     | Risk Manager     | Data Scientist |

### Step 3: Spawn 3 Agents in Parallel

Launch 3 Task agents simultaneously using Haiku model.

Each agent receives:

- Their persona file content
- Screenshot of the page
- Context: "Review this specific page of StockPulse. Focus on your area of expertise."

### Step 4: Synthesize Quick Brief

```
# Quick Review: [Page Name]

## Top 3 Issues
1. [issue] — [agent who flagged it]
2. ...
3. ...

## Quick Wins
1. [recommendation]
2. ...

## Individual Takes
### [Agent 1 Name]
[summary]

### [Agent 2 Name]
[summary]

### [Agent 3 Name]
[summary]
```

## User-Invocable

This skill can be invoked by the user with `/review-page <page>`.
