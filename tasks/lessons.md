# StockPulse - Lessons Learned

<!-- After ANY correction from the user, log the pattern here to prevent repeating it. -->

## Patterns

- **Dictionary key mismatches**: When a function stores data with one key name and another function reads it, always verify the keys match. The AI score breakdown bug (pct_30d vs perf_30d) was a display-only bug that persisted across multiple sessions because the email "looked right" at a glance.
- **Score saturation in pullback markets**: Step-based scoring with a high base (50/100) saturates quickly when all components are correlated (pullback → low RSI → below MA). Use continuous scoring and a lower base for better differentiation.
- **Thread safety with stdout/stderr redirection**: When using APScheduler with concurrent jobs, global stdout/stderr redirection requires a threading.Lock or jobs will leak output into each other.
- **Model ID changes**: Claude model IDs change over time. The old `claude-haiku-4-20250514` was replaced by `claude-haiku-4-5-20251001`. Always check for 404 errors in logs after API updates.
- **Git workflow**: Martin expects Claude to handle ALL git operations end-to-end (commit, push feature branch, merge to main, push main). Don't ask him to do git.
