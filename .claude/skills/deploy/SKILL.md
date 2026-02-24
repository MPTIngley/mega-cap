---
name: deploy
description: This skill should be used when the user says "/deploy" or asks to deploy, push, or merge changes to main. It handles the full git + service restart cycle.
version: 0.1.0
---

# Deploy Skill

Execute the full deployment pipeline for StockPulse. This covers git operations and service restarts.

## Steps

1. **Stage and commit** all changes with a descriptive commit message
2. **Push** to the feature branch: `git push -u origin claude/init-repo-setup-maaOL`
3. **Checkout main**: `git checkout main`
4. **Pull latest main**: `git pull origin main`
5. **Merge feature branch**: `git merge claude/init-repo-setup-maaOL`
6. **Push main**: `git push origin main`
7. **Switch back** to feature branch: `git checkout claude/init-repo-setup-maaOL`
8. **Restart services**:
   - `launchctl unload ~/Library/LaunchAgents/com.stockpulse.scheduler.plist`
   - `launchctl load ~/Library/LaunchAgents/com.stockpulse.scheduler.plist`
   - `launchctl unload ~/Library/LaunchAgents/com.stockpulse.dashboard.plist`
   - `launchctl load ~/Library/LaunchAgents/com.stockpulse.dashboard.plist`
9. **Verify** services are running: `launchctl list | grep stockpulse`

## Important Notes

- Always switch back to the feature branch after pushing main
- If merge conflicts occur, resolve them before continuing
- Confirm services restarted successfully before reporting done
- Use the Co-Authored-By trailer on commits
