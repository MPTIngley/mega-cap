---
name: council
description: This skill should be used when the user says "/council" followed by a research topic or question. It spawns 4 parallel research agents with distinct perspectives and synthesizes their findings.
version: 0.1.0
---

# Council Multi-Perspective Research

When invoked with `/council <topic>`, perform multi-perspective research by spawning 4 parallel Task subagents.

## Instructions

1. **Parse the topic** from the user's input after `/council`.

2. **Spawn 4 Task subagents in parallel** (all in a single message), each with `subagent_type: "general-purpose"`. Each agent gets:
   - The research topic
   - Their specific persona instructions (from the agent files below)
   - Instructions to end their response with `RECOMMENDATION: bullish/bearish/neutral` and `CONFIDENCE: 0-100`

   The 4 personas:

   **The Analyst** (council-analyst): Focus on news, catalysts, market narrative. What's the market saying? What events are coming?

   **The Skeptic** (council-skeptic): Focus on risk, doubt, what could go wrong. What are we not seeing? Capital preservation first.

   **The Quant** (council-quant): Focus on data, RSI, P/E, technicals, flows. Numbers only, no narrative.

   **The Philosopher** (council-philosopher): Reframe the question. What if the thesis itself is wrong? Historical analogies, second-order effects.

3. **Collect all 4 results**, then **synthesize** into a council brief:

   ```
   ## Council Brief: <topic>

   ### Verdict: <UNANIMOUS/MAJORITY/SPLIT> <RECOMMENDATION> (<N/4>)
   Confidence: <weighted average>% | Agreement: <score>

   ### Perspectives

   **The Analyst**: <1-2 sentence summary> — <recommendation>
   **The Skeptic**: <1-2 sentence summary> — <recommendation>
   **The Quant**: <1-2 sentence summary> — <recommendation>
   **The Philosopher**: <1-2 sentence summary> — <recommendation>

   ### Key Disagreements
   <If any dissent, highlight the core tension>

   ### Synthesis
   <2-3 sentences combining the strongest points from each perspective>
   ```

4. Present the synthesized brief to the user.

## Notes

- Use web search within each agent for current data when relevant
- Each agent should research independently — don't share findings between them
- The synthesis is YOUR job after collecting all 4 results
- For investment topics, agents should consider actual market data
- For non-investment topics, agents adapt their lens (risk, data, narrative, reframing)
