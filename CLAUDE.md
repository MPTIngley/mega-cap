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

- Always work on branch: `claude/init-repo-setup-maaOL`
- Pull before starting work: `git pull origin claude/init-repo-setup-maaOL`
- Commit with clear, descriptive messages
- Push with: `git push -u origin claude/init-repo-setup-maaOL`

## Project Structure

- Main source: `src/stockpulse/`
- Config: `config/config.yaml`
- Data: `data/stockpulse.sqlite`
- Docs: `docs/`, `COMMANDS.md`, `PROJECT_PLAN.md`

## Key Files Reference

- `COMMANDS.md` - All available stockpulse commands
- `PROJECT_PLAN.md` - Feature roadmap and implementation status
- `docs/SENTIMENT_REVIEW.md` - Sentiment system documentation
