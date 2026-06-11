"""
RegimeAnalystNode.py - Agent 2: Capital Rotation and Macro Regime Detector.

This agent acts as a strict quantitative gatekeeper. It ingests Agent 1's
payload (ScoutArticle with EmotionalAnalysis and cluster context) alongside
a baseline CurrentMarketState and evaluates whether the incoming facts
constitute a genuine "Regime Change."

A regime change is NOT just a macroeconomic pivot. It heavily includes
Capital Rotation: money moving between broad sectors (e.g. Tech to Healthcare)
or intra-sector shifts (e.g. Semiconductors into Software).

Scoring formula (computed in Python, not by LLM):
    S = alpha*M + beta*R + gamma*E
where:
    M = Macroeconomic Impact (LLM scores 1-10)
    R = Capital Rotation Intensity (LLM scores 1-10)
    E = Emotional Arbitrage Gap (LLM scores 1-10)
    alpha=0.35, beta=0.40, gamma=0.25  (configurable in src/config.py)

All tunable constants are in src/config.py.
"""

from typing import List, Optional

from google import genai

from src.state import ScoutArticle, RegimeAnalysis, GraphState, CurrentMarketState
from src.utils.json_repair import parse_json_with_repair
from src.config import (
    GEMINI_API_KEY,
    REGIME_GEMINI_MODEL,  # Per-agent model config
    REGIME_LLM_TEMPERATURE,
    REGIME_LLM_MAX_TOKENS,
    REGIME_WEIGHT_MACRO,
    REGIME_WEIGHT_ROTATION,
    REGIME_WEIGHT_EMOTIONAL,
    REGIME_SIGNIFICANCE_THRESHOLD,
    REGIME_INDIVIDUAL_TRIGGER_THRESHOLD,
    REGIME_DEFAULT_MARKET_STATE,
)


class RegimeAnalystNode:
    """
    Evaluates enriched articles against the current market baseline to
    detect macro regime shifts and capital rotation signals.
    """

    def __init__(self):
        self._client: Optional[genai.Client] = None
        if GEMINI_API_KEY:
            self._client = genai.Client(api_key=GEMINI_API_KEY)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_regime_analyst_node(self, state: GraphState) -> GraphState:
        """
        Main entry point. Accepts the full GraphState, runs the regime
        analysis, and writes results back into state.

        Returns the mutated GraphState with:
          - regime_analysis populated
          - proceed_to_portfolio_manager set
        """
        articles = state.get("articles", [])
        market_state = state.get("market_state", REGIME_DEFAULT_MARKET_STATE)

        if not articles:
            print("\n  ⚠️  No articles to analyze. Skipping regime analysis.")
            state["regime_analysis"] = RegimeAnalysis(
                Macro_Analysis="No articles provided for analysis.",
                Rotation_Analysis="N/A",
                Emotional_Arbitrage_Analysis="N/A",
                Significance_Score=0,
                proceed_to_portfolio_manager=False,
            )
            state["proceed_to_portfolio_manager"] = False
            return state

        print(f"\n  🏛️  Regime Analyst: evaluating {len(articles)} article(s) against market baseline...")

        analysis = self._run_analysis(articles, market_state)
        state["regime_analysis"] = analysis
        state["proceed_to_portfolio_manager"] = analysis.proceed_to_portfolio_manager

        return state

    # ------------------------------------------------------------------
    # Core LLM-based analysis
    # ------------------------------------------------------------------
    def _run_analysis(
        self,
        articles: List[ScoutArticle],
        market_state: CurrentMarketState,
    ) -> RegimeAnalysis:
        if not self._client:
            return self._heuristic_analysis(articles, market_state)

        # Build the payload: summary of all articles with their emotional/factual split
        article_summaries = self._build_article_payload(articles)

        prompt = self._build_system_prompt(article_summaries, market_state)

        try:
            response = self._client.models.generate_content(
                model=REGIME_GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": REGIME_LLM_TEMPERATURE,
                    "max_output_tokens": REGIME_LLM_MAX_TOKENS,
                },
            )
            text = response.text.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            if text.startswith("json"):
                text = text[4:].strip()

            llm_result = parse_json_with_repair(text)

            # Extract LLM scores (1-10 per factor)
            macro_score = max(1, min(10, int(llm_result.get("macro_score", 1))))
            rotation_score = max(1, min(10, int(llm_result.get("rotation_score", 1))))
            emotional_score = max(1, min(10, int(llm_result.get("emotional_arbitrage_score", 1))))

            # Compute weighted Significance Score (0-100 scale)
            # Each factor scored 1-10, so (score/10)*weight*100 gives factor's contribution
            significance = int(
                round(
                    (macro_score / 10.0) * REGIME_WEIGHT_MACRO * 100
                    + (rotation_score / 10.0) * REGIME_WEIGHT_ROTATION * 100
                    + (emotional_score / 10.0) * REGIME_WEIGHT_EMOTIONAL * 100
                )
            )
            significance = max(0, min(100, significance))

            # Composite gate
            composite_trigger = significance > REGIME_SIGNIFICANCE_THRESHOLD

            # Individual trigger: open the gate if ANY single factor score
            # reaches the threshold, regardless of the weighted composite.
            individual_trigger = (
                macro_score >= REGIME_INDIVIDUAL_TRIGGER_THRESHOLD
                or rotation_score >= REGIME_INDIVIDUAL_TRIGGER_THRESHOLD
                or emotional_score >= REGIME_INDIVIDUAL_TRIGGER_THRESHOLD
            )

            proceed = composite_trigger or individual_trigger

            analysis = RegimeAnalysis(
                Macro_Analysis=llm_result.get("Macro_Analysis", "No macro analysis provided."),
                Rotation_Analysis=llm_result.get("Rotation_Analysis", "No rotation analysis provided."),
                Emotional_Arbitrage_Analysis=llm_result.get(
                    "Emotional_Arbitrage_Analysis", "No emotional arbitrage analysis provided."
                ),
                macro_score=macro_score,
                rotation_score=rotation_score,
                emotional_arbitrage_score=emotional_score,
                Significance_Score=significance,
                proceed_to_portfolio_manager=proceed,
            )

            print_analysis(analysis, composite_trigger, individual_trigger)
            return analysis

        except Exception as e:
            print(f"    ⚠️  LLM regime analysis failed: {e}")
            return self._heuristic_analysis(articles, market_state)

    def _build_article_payload(self, articles: List[ScoutArticle]) -> str:
        """Build a concise text summary of all articles for the LLM prompt."""
        lines: List[str] = []
        for i, a in enumerate(articles):
            title = a.title[:120]
            source = f"{a.source_name} ({a.source_bucket})"

            # Include the tonality analysis if available
            tone_info = ""
            if a.emotional_analysis:
                ea = a.emotional_analysis
                tone_info = (
                    f"  [Tonality: {ea.tonality_label} | "
                    f"emotional={ea.emotional_score:.2f} factual={ea.factual_score:.2f} "
                    f"disparity={ea.disparity_score:.2f}]"
                )
                if ea.key_factual_claims:
                    tone_info += f"\n  [Factual claims: {'; '.join(ea.key_factual_claims[:3])}]"
                if ea.key_emotional_phrases:
                    tone_info += f"\n  [Emotional phrases: {'; '.join(ea.key_emotional_phrases[:3])}]"

            # Cluster info
            cluster_info = ""
            if a.related_articles:
                cluster_labels = []
                for ra in a.related_articles:
                    if ra.emotional_analysis:
                        cluster_labels.append(
                            f"{ra.source_name}:{ra.emotional_analysis.tonality_label}"
                        )
                if cluster_labels:
                    cluster_info = f"\n  [Related coverage: {', '.join(cluster_labels[:5])}]"

            lines.append(
                f"[{i+1}] {title}\n  Source: {source}\n"
                f"  Summary: {(a.summary or (a.aggregated_content[:200] if a.aggregated_content else 'N/A'))[:200]}\n"
                f"  Importance: {a.importance_score:.2f} | Query: {a.search_query}"
                f"{tone_info}{cluster_info}"
            )

        return "\n\n".join(lines)

    def _build_system_prompt(
        self,
        article_summaries: str,
        market_state: CurrentMarketState,
    ) -> str:
        """Build the full system prompt for the LLM."""

        # Format market state for the prompt
        macro = market_state.get("macro_baseline", {})
        alloc = market_state.get("portfolio_allocations", {})
        sectors = alloc.get("sectors", {})

        sector_lines = []
        for name, data in sectors.items():
            bias = ", ".join(data.get("sub_sector_bias", []))
            sector_lines.append(
                f"  - {name}: {data.get('weight_percent', 0)}% | Focus: {bias}"
            )
        sector_summary = "\n".join(sector_lines) if sector_lines else "  (No sector data provided)"

        prompt = (
            "You are a senior quantitative macro strategist at a hedge fund.\n"
            "Your job is to act as a STRICT GATEKEEPER: evaluate whether incoming\n"
            "news and data constitute a genuine 'Regime Change' that requires\n"
            "portfolio reallocation.\n\n"
            "A regime change means:\n"
            "- A structural shift in macroeconomic conditions (rates, inflation, growth)\n"
            "- OR identifiable capital rotation between sectors or sub-sectors\n"
            "- OR a significant emotional arbitrage gap (market wildly over/under-reacting)\n\n"
            "Analyze the articles below against the CURRENT MARKET BASELINE and\n"
            "score each factor on a scale of 1 (irrelevant) to 10 (extreme impact).\n\n"
            "=== CURRENT MARKET BASELINE ===\n"
            f"Timestamp: {market_state.get('timestamp', 'unknown')}\n"
            f"Macro Regime: {macro.get('market_regime', 'unknown')}\n"
            f"Rate Trend: {macro.get('interest_rate_trend', 'unknown')}\n"
            f"Inflation Trend: {macro.get('inflation_trend', 'unknown')}\n"
            f"Cash Reserves: {alloc.get('cash_reserves_percent', 0)}%\n"
            f"Portfolio Value: ${alloc.get('total_value', 0):,.2f}\n"
            f"Sector Allocations:\n{sector_summary}\n\n"
            "=== SCORING RUBRIC ===\n\n"
            "1. MACRO IMPACT (M, 1-10):\n"
            "   - Do articles indicate a shift in rate policy, inflation trajectory,\n"
            "     GDP growth, employment, or geopolitical macro risk?\n"
            "   - 1-3: No macro signal. 4-6: Notable but not regime-altering.\n"
            "     7-8: Clear macro shift emerging. 9-10: Structural macro regime change.\n\n"
            "2. ROTATION INTENSITY (R, 1-10):\n"
            "   - Is there verifiable evidence of capital flowing FROM one sector\n"
            "     or sub-sector INTO another? Consider both broad sector rotation\n"
            "     (e.g. Tech -> Healthcare) and intra-sector shifts\n"
            "     (e.g. Semiconductors -> Software, or AI infra -> AI apps).\n"
            "   - Pay special attention to our CURRENT portfolio sector weights\n"
            "     when evaluating if a rotation threatens our positioning.\n"
            "   - 1-3: No rotation signal. 4-6: Early rotation hints.\n"
            "     7-8: Clear rotation underway. 9-10: Massive capital migration.\n\n"
            "3. EMOTIONAL ARBITRAGE (E, 1-10):\n"
            "   - Using the provided emotional/factual disparity scores and\n"
            "     related article clusters, assess whether the market narrative\n"
            "     is ignoring a structural shift (underreaction) or violently\n"
            "     overreacting to noise (overreaction).\n"
            "   - High disparity + isolated reaction = potential arbitrage opportunity.\n"
            "   - Low disparity + broad consensus = efficient pricing, low arbitrage.\n"
            "   - 1-3: No arbitrage gap. 4-6: Moderate narrative distortion.\n"
            "     7-8: Significant over/under-reaction. 9-10: Extreme narrative disconnect.\n\n"
            "=== ARTICLES TO ANALYZE ===\n\n"
            f"{article_summaries}\n\n"
            "=== REQUIRED OUTPUT FORMAT ===\n\n"
            "Respond ONLY with valid JSON in this exact format:\n"
            "{\n"
            '  "Macro_Analysis": "Brief 1-2 sentence explanation of macro shifts vs baseline.",\n'
            '  "Rotation_Analysis": "Brief 1-2 sentence identification of sector/capital flows.",\n'
            '  "Emotional_Arbitrage_Analysis": "Brief 1-2 sentence on over/under-reaction.",\n'
            '  "macro_score": 5,\n'
            '  "rotation_score": 7,\n'
            '  "emotional_arbitrage_score": 4\n'
            "}\n\n"
            "CRITICAL: All three scores MUST be integers between 1 and 10 inclusive.\n"
            "Do not include Significance_Score or proceed_to_portfolio_manager -\n"
            "those are computed by the system, not by you."
        )
        return prompt

    # ------------------------------------------------------------------
    # Heuristic fallback (no Gemini key or API failure)
    # ------------------------------------------------------------------
    def _heuristic_analysis(
        self,
        articles: List[ScoutArticle],
        market_state: CurrentMarketState,
    ) -> RegimeAnalysis:
        """Heuristic scoring based on Agent 1's existing metrics."""

        # Macro: aggregate importance scores
        avg_importance = (
            sum(a.importance_score for a in articles) / len(articles)
            if articles else 0
        )
        macro_score = max(1, min(10, int(round(avg_importance * 10))))

        # Rotation: look for ticker tags and sector mentions in titles
        rotation_signals = 0
        rotation_keywords = [
            "rotation", "shift", "flow", "into", "out of", "sector",
            "semiconductor", "software", "tech", "healthcare", "energy",
            "financial", "industrial", "consumer", "reit", "utility",
        ]
        for a in articles:
            text = (a.title + " " + (a.summary or "") + " " + (a.aggregated_content or "")).lower()
            hits = sum(1 for kw in rotation_keywords if kw in text)
            if hits >= 3:
                rotation_signals += 2
            elif hits >= 1:
                rotation_signals += 1
            # Ticker tags suggest asset-specific content
            if a.ticker_tags:
                rotation_signals += 1
        rotation_score = max(1, min(10, rotation_signals * 2))

        # Emotional: aggregate disparity scores from ToneAnalystNode
        avg_disparity = 0.0
        disp_count = 0
        for a in articles:
            if a.emotional_analysis:
                avg_disparity += a.emotional_analysis.disparity_score
                disp_count += 1
        if disp_count:
            avg_disparity /= disp_count
        # Disparity 0-1 maps to E 1-10
        emotional_score = max(1, min(10, int(round(avg_disparity * 10)) + 1))

        significance = int(
            round(
                (macro_score / 10.0) * REGIME_WEIGHT_MACRO * 100
                + (rotation_score / 10.0) * REGIME_WEIGHT_ROTATION * 100
                + (emotional_score / 10.0) * REGIME_WEIGHT_EMOTIONAL * 100
            )
        )
        significance = max(0, min(100, significance))
        composite_trigger = significance > REGIME_SIGNIFICANCE_THRESHOLD
        individual_trigger = (
            macro_score >= REGIME_INDIVIDUAL_TRIGGER_THRESHOLD
            or rotation_score >= REGIME_INDIVIDUAL_TRIGGER_THRESHOLD
            or emotional_score >= REGIME_INDIVIDUAL_TRIGGER_THRESHOLD
        )
        proceed = composite_trigger or individual_trigger

        analysis = RegimeAnalysis(
            Macro_Analysis=(
                f"Heuristic: average importance score {avg_importance:.2f} across {len(articles)} articles. "
                f"Macro score {macro_score}/10."
            ),
            Rotation_Analysis=(
                f"Heuristic: {rotation_signals} rotation keyword/ticker signals detected. "
                f"Rotation score {rotation_score}/10."
            ),
            Emotional_Arbitrage_Analysis=(
                f"Heuristic: average emotional disparity {avg_disparity:.2f}. "
                f"Emotional arbitrage score {emotional_score}/10."
            ),
            macro_score=macro_score,
            rotation_score=rotation_score,
            emotional_arbitrage_score=emotional_score,
            Significance_Score=significance,
            proceed_to_portfolio_manager=proceed,
        )
        print_analysis(analysis, composite_trigger, individual_trigger)
        return analysis


# ------------------------------------------------------------------
# Conditional routing
# ------------------------------------------------------------------
def route_after_regime_analyst(state: GraphState) -> str:
    """
    Conditional edge function for LangGraph.
    Returns 'agent_3' if the regime analysis indicates portfolio manager
    action is needed, otherwise returns 'end'.
    """
    if state.get("proceed_to_portfolio_manager", False):
        return "portfolio_manager"
    return "__end__"


# ------------------------------------------------------------------
# Pretty-printing
# ------------------------------------------------------------------
def print_analysis(
    analysis: RegimeAnalysis,
    composite_trigger: bool = False,
    individual_trigger: bool = False,
) -> None:
    """Print a formatted summary of the regime analysis.

    Args:
        analysis: The completed RegimeAnalysis.
        composite_trigger: True if the weighted significance score exceeded threshold.
        individual_trigger: True if any single factor score hit the independent threshold.
    """
    if analysis.proceed_to_portfolio_manager:
        trigger_parts: list[str] = []
        if composite_trigger:
            trigger_parts.append(f"Composite S={analysis.Significance_Score}>70")
        if individual_trigger:
            top_factor = ""
            if analysis.macro_score >= REGIME_INDIVIDUAL_TRIGGER_THRESHOLD:
                top_factor = f"M:{analysis.macro_score}"
            elif analysis.rotation_score >= REGIME_INDIVIDUAL_TRIGGER_THRESHOLD:
                top_factor = f"R:{analysis.rotation_score}"
            elif analysis.emotional_arbitrage_score >= REGIME_INDIVIDUAL_TRIGGER_THRESHOLD:
                top_factor = f"E:{analysis.emotional_arbitrage_score}"
            trigger_parts.append(f"Individual spike {top_factor}≥{REGIME_INDIVIDUAL_TRIGGER_THRESHOLD}")
        trigger_detail = " + ".join(trigger_parts)
        verdict = "🚨 REGIME CHANGE DETECTED → route to Portfolio Manager"
    else:
        verdict = "✅ No regime change — market state stable"
        trigger_detail = ""
    print(f"    ╔══════════════════════════════════════════════╗")
    print(f"    ║  REGIME ANALYST VERDICT                      ║")
    print(f"    ╠══════════════════════════════════════════════╣")
    print(f"    ║  M (Macro):      {analysis.macro_score:2d}/10  |  R (Rotation):  {analysis.rotation_score:2d}/10  |  E (Arbitrage): {analysis.emotional_arbitrage_score:2d}/10 ║")
    print(f"    ║  Significance Score: {analysis.Significance_Score:3d}/100                     ║")
    print(f"    ║  {verdict} ║")
    if trigger_detail:
        trigger_display = f"    ║  Trigger: {trigger_detail}"
        # Pad to match box width
        print(trigger_display + " " * max(0, 50 - len(trigger_display)) + "║")
    print(f"    ╠══════════════════════════════════════════════╣")
    print(f"    ║  Macro:     {analysis.Macro_Analysis[:80]}...{' ' * max(0, 74 - len(analysis.Macro_Analysis[:80]))}║")
    print(f"    ║  Rotation:  {analysis.Rotation_Analysis[:80]}...{' ' * max(0, 74 - len(analysis.Rotation_Analysis[:80]))}║")
    print(f"    ║  Arbitrage: {analysis.Emotional_Arbitrage_Analysis[:80]}...{' ' * max(0, 74 - len(analysis.Emotional_Arbitrage_Analysis[:80]))}║")
    print(f"    ╚══════════════════════════════════════════════╝")