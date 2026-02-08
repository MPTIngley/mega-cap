# Claude Code Project Instructions

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
When ready to update main with Claude's changes:

```
git fetch origin claude/init-repo-setup-maaOL
git checkout main
git merge origin/claude/init-repo-setup-maaOL
git push origin main
```

Then sync the feature branch:
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
