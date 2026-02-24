# Claude Code Project Instructions

## Workflow & Operating Rules

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

---

## Command Preferences

Use the `stockpulse` alias for all commands (not `python3 -m stockpulse.main`):

```
stockpulse longterm-scan
stockpulse trillion-scan
stockpulse ai-scan
stockpulse run
stockpulse dashboard
stockpulse init
stockpulse reset
```

## Running StockPulse (macOS)

### One-Time Setup

Install the launchd services (auto-restart, survives reboots):

```
cp deploy/com.stockpulse.scheduler.plist ~/Library/LaunchAgents/
cp deploy/com.stockpulse.dashboard.plist ~/Library/LaunchAgents/
```

### Start Services

```
launchctl load ~/Library/LaunchAgents/com.stockpulse.scheduler.plist
launchctl load ~/Library/LaunchAgents/com.stockpulse.dashboard.plist
```

### Stop Services

```
launchctl unload ~/Library/LaunchAgents/com.stockpulse.scheduler.plist
launchctl unload ~/Library/LaunchAgents/com.stockpulse.dashboard.plist
```

### Restart Services

```
launchctl unload ~/Library/LaunchAgents/com.stockpulse.scheduler.plist
launchctl load ~/Library/LaunchAgents/com.stockpulse.scheduler.plist
launchctl unload ~/Library/LaunchAgents/com.stockpulse.dashboard.plist
launchctl load ~/Library/LaunchAgents/com.stockpulse.dashboard.plist
```

### Check Status

```
launchctl list | grep stockpulse
```

### View Logs

```
tail -f ~/Documents/AIGames/mega-cap/logs/stockpulse.log
tail -f ~/Documents/AIGames/mega-cap/logs/dashboard.log
```

### Kill Processes

```
pkill -f "stockpulse run"
pkill -f "stockpulse dashboard"
```

### Quick Reference

| Action | Command |
|--------|---------|
| Start scheduler | `launchctl load ~/Library/LaunchAgents/com.stockpulse.scheduler.plist` |
| Stop scheduler | `launchctl unload ~/Library/LaunchAgents/com.stockpulse.scheduler.plist` |
| Start dashboard | `launchctl load ~/Library/LaunchAgents/com.stockpulse.dashboard.plist` |
| Stop dashboard | `launchctl unload ~/Library/LaunchAgents/com.stockpulse.dashboard.plist` |

## Code Block Rules

- NEVER use inline `#` comments in bash code blocks - they break shell commands
- Keep code blocks clean with no trailing comments
- If explanation needed, put comments on separate lines BEFORE the command

## Git Hygiene

### Claude Code Workflow
- Always work on branch: `claude/init-repo-setup-maaOL`
- Pull before starting work: `git pull origin claude/init-repo-setup-maaOL`
- Commit with clear, descriptive messages
- Push with: `git push -u origin claude/init-repo-setup-maaOL`
- Claude cannot push to `main` (403 protected)

### Martin's Workflow (Update Main)

**Quick version - merge Claude's work to main:**
```
git fetch origin claude/init-repo-setup-maaOL && git checkout main && git merge origin/claude/init-repo-setup-maaOL && git push origin main
```

**Step-by-step version:**
```
git fetch origin claude/init-repo-setup-maaOL
git checkout main
git merge origin/claude/init-repo-setup-maaOL
git push origin main
```

**Optional - sync Claude's branch with main (only if needed):**
```
git checkout claude/init-repo-setup-maaOL
git pull origin main
git push origin claude/init-repo-setup-maaOL
```

## Project Structure

- Main source: `src/stockpulse/`
- Config: `config/config.yaml`
- Data: `data/stockpulse.sqlite`
- Docs: `docs/`, `COMMANDS.md`, `PROJECT_PLAN.md`

## Key Files Reference

- `COMMANDS.md` - All available stockpulse commands
- `PROJECT_PLAN.md` - Feature roadmap and implementation status
- `docs/SENTIMENT_REVIEW.md` - Sentiment system documentation

## Post-Change Review Protocol

After any **major code change** (new features, significant refactors, bug fixes touching multiple files):

1. **Double Pass Review** - Do TWO separate passes through all changed code:
   - **Pass 1**: Check for bugs, logic errors, edge cases, unhandled exceptions
   - **Pass 2**: Check for typos, incorrect variable names, missing imports, broken references

2. **Ask About Rubber Duck Session** - After completing the double pass, ASK the user:
   > "Would you like me to do a rubber duck session to walk through the code changes in detail? This covers code structure, logic flow, how functions interact, and ensures everything makes sense."

If they say yes, execute the detailed prompt below.

---

## Rubber Duck Session Prompt

When the user agrees to a rubber duck session, follow this structured walkthrough:

```
RUBBER DUCK CODE REVIEW SESSION
===============================

I will now walk through the recent changes as if explaining to a staff engineer.
This covers EVERYTHING in detail.

### 1. HIGH-LEVEL OVERVIEW
- What was the goal of these changes?
- What problem does this solve?
- How does it fit into the existing system?

### 2. FILE-BY-FILE WALKTHROUGH
For each modified/new file:
- What is this file's responsibility?
- What functions/classes were added or changed?
- Why were these changes made here vs elsewhere?

### 3. FUNCTION-BY-FUNCTION DEEP DIVE
For each significant function:
- What does this function do? (plain English)
- What are the inputs and outputs?
- What are the edge cases?
- How is this function called? By what?
- What does it call downstream?

### 4. DATA FLOW ANALYSIS
- How does data flow through the system?
- What gets read from where?
- What gets written to where?
- How do the pieces connect?

### 5. SCANNING LOGIC (if applicable)
- What scans run and when?
- What triggers them?
- What do they check for?
- What thresholds/criteria are used?

### 6. EMAIL/NOTIFICATION LOGIC (if applicable)
- When are emails sent?
- What data goes into them?
- What templates are used?
- What conditions must be met?

### 7. ERROR HANDLING
- What can go wrong?
- How are errors caught and handled?
- Are there any silent failures?

### 8. TESTING CONSIDERATIONS
- How would you test this?
- What manual tests should be run?
- Are there automated tests?

### 9. POTENTIAL ISSUES IDENTIFIED
- Any concerns spotted during this review?
- Any TODOs or future improvements noted?

### 10. SUMMARY
- Recap of what was built/changed
- Confidence level in the implementation
- Recommended next steps
```
