"""Council Multi-Perspective Research System.

Calls Claude 4x per thesis with distinct analyst personas, then aggregates
into a consensus verdict with disagreement signals. Reduces confirmation bias
and surfaces angles a single prompt would miss.

Personas adapted from Mnemosyne's multi-perspective approach:
  - The Analyst: News, catalysts, market narrative
  - The Skeptic: Risk, doubt, what could go wrong
  - The Quant: Numbers, RSI, P/E, flows
  - The Philosopher: Reframing, alternative theses
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger

logger = get_logger(__name__)

PERSPECTIVES = {
    "analyst": {
        "name": "The Analyst",
        "focus": "News, catalysts, market narrative",
        "system_prompt": (
            "You are The Analyst — a senior equity research analyst focused on "
            "catalysts, news flow, and market narrative. Your job is to assess "
            "what the market is pricing in, what catalysts are ahead, and whether "
            "the current narrative supports the thesis. You cite recent events, "
            "earnings, product launches, and sector trends. You are forward-looking "
            "and focused on what moves the stock next."
        ),
    },
    "skeptic": {
        "name": "The Skeptic",
        "focus": "Risk, doubt, what could go wrong",
        "system_prompt": (
            "You are The Skeptic — a risk-focused analyst whose job is to find "
            "holes in the thesis. You ask: what could go wrong? What is the market "
            "NOT seeing? You look for crowded trades, regulatory risk, competitive "
            "threats, valuation excess, and narrative fatigue. You are not bearish "
            "by default, but you demand extraordinary evidence for bullish claims. "
            "Your bias is toward capital preservation."
        ),
    },
    "quant": {
        "name": "The Quant",
        "focus": "Data, RSI, P/E, flows, technicals",
        "system_prompt": (
            "You are The Quant — a quantitative analyst who speaks in numbers. "
            "You focus on RSI, P/E ratios, price-to-sales, moving averages, "
            "volume patterns, and institutional flows. You assess whether the "
            "price action supports the thesis, whether valuations are stretched, "
            "and where key technical levels sit. Narrative means nothing to you; "
            "only the data matters."
        ),
    },
    "philosopher": {
        "name": "The Philosopher",
        "focus": "Reframing, alternative theses",
        "system_prompt": (
            "You are The Philosopher — a contrarian thinker who questions the "
            "thesis itself. You ask: what if the entire framing is wrong? You "
            "propose alternative interpretations of the same data. You consider "
            "second-order effects, historical analogies, and structural shifts "
            "that others miss. You are comfortable saying 'the question itself "
            "is wrong' when warranted."
        ),
    },
}

# Valid recommendations for parsing
VALID_RECOMMENDATIONS = {"bullish", "bearish", "neutral"}


def _call_perspective(
    api_key: str,
    model: str,
    perspective_key: str,
    thesis_name: str,
    context: str,
    tickers: list[str],
    ticker_performance: dict[str, dict] | None,
    max_tokens: int,
) -> dict[str, Any]:
    """Call Claude with a single perspective. Designed for ThreadPoolExecutor."""
    perspective = PERSPECTIVES[perspective_key]

    # Build performance data section
    perf_section = ""
    if ticker_performance:
        perf_lines = []
        for ticker, data in ticker_performance.items():
            perf_lines.append(
                f"  {ticker}: ${data['price']:.2f} | "
                f"30d: {data['pct_30d']:+.1f}% | "
                f"90d: {data['pct_90d']:+.1f}% | "
                f"RSI: {data['rsi']:.0f} ({data['signal']})"
            )
        perf_section = "\n\nACTUAL PRICE PERFORMANCE:\n" + "\n".join(perf_lines)

    user_prompt = f"""Analyze this investment thesis from your perspective.

THESIS: {thesis_name}
CONTEXT: {context}
RELATED TICKERS: {", ".join(tickers)}
{perf_section}

TODAY'S DATE: {date.today().isoformat()}

Provide:
1. Your perspective on this thesis (2-3 sentences)
2. Key evidence supporting your view
3. RECOMMENDATION: bullish / bearish / neutral
4. CONFIDENCE: 0-100

Keep response under 250 words. End with exactly:
RECOMMENDATION: <bullish|bearish|neutral>
CONFIDENCE: <number>"""

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=perspective["system_prompt"],
            messages=[{"role": "user", "content": user_prompt}],
        )

        analysis = message.content[0].text
        recommendation, confidence = _parse_verdict(analysis)

        return {
            "perspective": perspective_key,
            "name": perspective["name"],
            "focus": perspective["focus"],
            "analysis": analysis,
            "recommendation": recommendation,
            "confidence": confidence,
        }

    except Exception as e:
        logger.error(f"Council {perspective_key} error: {e}")
        return {
            "perspective": perspective_key,
            "name": perspective["name"],
            "focus": perspective["focus"],
            "analysis": f"Error: {e}",
            "recommendation": "neutral",
            "confidence": 0,
        }


def _parse_verdict(analysis: str) -> tuple[str, float]:
    """Parse recommendation and confidence from Claude response."""
    lines = analysis.strip().split("\n")
    recommendation = "neutral"
    confidence = 50.0

    for line in reversed(lines):
        line_lower = line.lower().strip()
        if line_lower.startswith("recommendation:"):
            value = line_lower.split(":", 1)[1].strip().rstrip(".")
            # Handle "**bullish**" markdown bold
            value = value.strip("*").strip()
            if value in VALID_RECOMMENDATIONS:
                recommendation = value
        elif line_lower.startswith("confidence:"):
            try:
                value = line_lower.split(":", 1)[1].strip().rstrip("%").rstrip(".")
                # Handle "**75**" markdown bold
                value = value.strip("*").strip()
                confidence = float(value)
                confidence = max(0, min(100, confidence))
            except (ValueError, IndexError):
                pass

    return recommendation, confidence


def _build_consensus(perspectives: list[dict]) -> dict[str, Any]:
    """Aggregate perspectives into consensus verdict."""
    recs = [p["recommendation"] for p in perspectives if p["confidence"] > 0]
    confs = [p["confidence"] for p in perspectives if p["confidence"] > 0]

    if not recs:
        return {
            "recommendation": "neutral",
            "confidence": 50,
            "agreement_score": 0,
            "vote_breakdown": {},
        }

    # Count votes
    vote_counts = {}
    for r in recs:
        vote_counts[r] = vote_counts.get(r, 0) + 1

    # Majority vote
    consensus_rec = max(vote_counts, key=vote_counts.get)
    majority_count = vote_counts[consensus_rec]

    # Agreement score: fraction that agree with majority
    agreement_score = majority_count / len(recs) if recs else 0

    # Weighted average confidence (weight by agreement with majority)
    weighted_conf = []
    for p in perspectives:
        if p["confidence"] > 0:
            weight = 1.5 if p["recommendation"] == consensus_rec else 0.5
            weighted_conf.append(p["confidence"] * weight)
    avg_confidence = (
        sum(weighted_conf)
        / sum(
            1.5 if p["recommendation"] == consensus_rec else 0.5
            for p in perspectives
            if p["confidence"] > 0
        )
        if weighted_conf
        else 50
    )

    return {
        "recommendation": consensus_rec,
        "confidence": round(avg_confidence, 1),
        "agreement_score": round(agreement_score, 2),
        "vote_breakdown": vote_counts,
    }


def _find_dissent(perspectives: list[dict], consensus_rec: str) -> str:
    """Summarize dissenting views."""
    dissenters = [
        p for p in perspectives if p["recommendation"] != consensus_rec and p["confidence"] > 0
    ]
    if not dissenters:
        return ""

    parts = []
    for d in dissenters:
        # Extract first sentence of analysis as summary
        first_sentence = d.get("analysis", "").split(".")[0].strip()
        if len(first_sentence) > 120:
            first_sentence = first_sentence[:117] + "..."
        name = d.get("name", d.get("perspective", "Unknown"))
        parts.append(f"{name} ({d['recommendation']}): {first_sentence}")

    return "; ".join(parts)


class CouncilResearch:
    """Multi-perspective research using 4 Claude personas per thesis."""

    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.is_configured = bool(self.api_key)
        self.config = get_config()
        council_cfg = self.config.get("ai_pulse", {}).get("research", {}).get("council", {})
        self.enabled = council_cfg.get("enabled", False)
        self.model = council_cfg.get("model", "claude-haiku-4-5-20251001")
        self.active_perspectives = council_cfg.get("perspectives", list(PERSPECTIVES.keys()))

    def research_with_council(
        self,
        thesis_name: str,
        context: str,
        tickers: list[str],
        ticker_performance: dict[str, dict] | None = None,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Research a thesis using the full council (4 parallel perspectives).

        Returns:
            {
                "thesis": str,
                "perspectives": [
                    {"perspective": str, "name": str, "analysis": str,
                     "recommendation": str, "confidence": float}, ...
                ],
                "consensus": {
                    "recommendation": str, "confidence": float,
                    "agreement_score": float, "vote_breakdown": dict
                },
                "dissent": str,
                "verdict": str,
                "analysis": str,  # consensus summary for backwards compat
                "recommendation": str,
                "confidence": float,
                "researched_at": str,
            }
        """
        if not self.is_configured:
            return self._empty_result(thesis_name, "ANTHROPIC_API_KEY not set")

        perspectives = []
        active_keys = [k for k in self.active_perspectives if k in PERSPECTIVES]

        logger.info(f"Council researching '{thesis_name}' with {len(active_keys)} perspectives")

        with ThreadPoolExecutor(max_workers=len(active_keys)) as executor:
            futures = {
                executor.submit(
                    _call_perspective,
                    self.api_key,
                    self.model,
                    key,
                    thesis_name,
                    context,
                    tickers,
                    ticker_performance,
                    max_tokens,
                ): key
                for key in active_keys
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    perspectives.append(result)
                except Exception as e:
                    key = futures[future]
                    logger.error(f"Council perspective {key} failed: {e}")

        # Sort perspectives in canonical order
        order = {k: i for i, k in enumerate(PERSPECTIVES.keys())}
        perspectives.sort(key=lambda p: order.get(p["perspective"], 99))

        consensus = _build_consensus(perspectives)
        dissent = _find_dissent(perspectives, consensus["recommendation"])

        # Build human-readable verdict
        total = len([p for p in perspectives if p["confidence"] > 0])
        majority = consensus["vote_breakdown"].get(consensus["recommendation"], 0)
        rec_upper = consensus["recommendation"].upper()

        if consensus["agreement_score"] == 1.0:
            verdict = f"UNANIMOUS {rec_upper} ({total}/{total})"
        else:
            verdict = f"{rec_upper} ({majority}/{total})"
            if dissent:
                verdict += f" — Dissent: {dissent}"

        # Build analysis summary for backwards compatibility
        analysis_parts = [f"COUNCIL VERDICT: {verdict}\n"]
        for p in perspectives:
            # First 2 sentences
            sentences = p["analysis"].split(".")
            summary = ".".join(sentences[:2]).strip()
            if summary and not summary.endswith("."):
                summary += "."
            analysis_parts.append(f"[{p['name']}] {summary}")

        analysis_text = "\n\n".join(analysis_parts)

        logger.info(
            f"Council verdict for '{thesis_name}': {verdict} "
            f"(confidence: {consensus['confidence']:.0f}%)"
        )

        return {
            "thesis": thesis_name,
            "perspectives": perspectives,
            "consensus": consensus,
            "dissent": dissent,
            "verdict": verdict,
            "analysis": analysis_text,
            "recommendation": consensus["recommendation"],
            "confidence": consensus["confidence"],
            "researched_at": datetime.now().isoformat(),
        }

    def _empty_result(self, thesis_name: str, reason: str) -> dict[str, Any]:
        return {
            "thesis": thesis_name,
            "perspectives": [],
            "consensus": {
                "recommendation": "neutral",
                "confidence": 0,
                "agreement_score": 0,
                "vote_breakdown": {},
            },
            "dissent": "",
            "verdict": reason,
            "analysis": reason,
            "recommendation": "neutral",
            "confidence": 0,
            "researched_at": datetime.now().isoformat(),
        }
