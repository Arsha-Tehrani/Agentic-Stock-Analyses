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

Native JSON mode: Uses response_mime_type="application/json" + response_schema
to force structured output from the API. System instruction is separated from
user content to prevent instruction-echo responses. Safety-filtered responses
(text=None) are caught and fall back to heuristic analysis.

All tunable constants are in src/config.py.
"""

from typing import List, Optional

from google import genai
from pydantic import BaseModel, Field

from src.state import ScoutArticle, RegimeAnalysis, GraphState, CurrentMarketState
from src.config import (
    GEMINI_API_KEY,
    REGIME_GEMINI_MODEL,
    REGIME_LLM_TEMPERATURE,
    REGIME_LLM_MAX_TOKENS,
    REGIME_WEIGHT_MACRO,
    REGIME_WEIGHT_ROTATION,
    REGIME_WEIGHT_EMOTIONAL,
    REGIME_SIGNIFICANCE_THRESHOLD,
    REGIME_INDIVIDUAL_TRIGGER_THRESHOLD,
    REGIME_DEFAULT_MARKET_STATE,
)
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)


# ── Native JSON Structured-Output Schema ─────────────────────────────────
class RegimeResponse(BaseModel):
    """Pydantic schema for Gemini native JSON mode — regime analysis.

    The Significance_Score and proceed_to_portfolio_manager fields are
    computed by the Python orchestrator, NOT by the LLM. The LLM only
    provides the three analysis texts and three integer scores (1-10).
    """
    Macro_Analysis: str = Field(description="Brief explanation of macro shifts vs baseline")
    Rotation_Analysis: str = Field(description="Brief identification of sector/capital flows")
    Emotional_Arbitrage_Analysis: str = Field(description="Brief explanation of over/under-reaction")
    macro_score: int = Field(ge=1, le=10, description="Macro impact score 1-10")
    rotation_score: int = Field(ge=1, le=10, description="Rotation intensity score 1-10")
    emotional_arbitrage_score: int = Field(ge=1, le=10, description="Emotional arbitrage score 1-10")


class RegimeAnalystNode:
    """
    Evaluates enriched articles against the current market baseline to
    detect macro regime shifts and capital rotation signals.

    Native JSON mode prevents instruction-echo responses and guarantees
    well-formed output.
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
    # Core LLM-based analysis (native JSON mode)
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
        system_instruction = self._build_system_instruction()
        contents = self._build_contents(article_summaries, market_state)

        try:
            llm_result = self._call_regime_gemini(system_instruction, contents)

            # Guard against safety-filtered responses
            if llm_result is None:
                print("    ⚠️  Gemini returned safety-filtered response (text=None). Falling back to heuristic.")
                return self._heuristic_analysis(articles, market_state)

            # Extract LLM scores (1-10 per factor)
            macro_score = max(1, min(10, int(llm_result.macro_score)))
            rotation_score = max(1, min(10, int(llm_result.rotation_score)))
            emotional_score = max(1, min(10, int(llm_result.emotional_arbitrage_score)))

            # Compute weighted Significance Score (0-100 scale)
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
                Macro_Analysis=llm_result.Macro_Analysis or "No macro analysis provided.",
                Rotation_Analysis=llm_result.Rotation_Analysis or "No rotation analysis provided.",
                Emotional_Arbitrage_Analysis=llm_result.Emotional_Arbitrage_Analysis or "No emotional arbitrage analysis provided.",
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

    def _build_system_instruction(self) -> str:
        """Build the system instruction (role definition + scoring rubric).

        Kept separate from the data payload to prevent instruction-echo
        responses where the model recites prompt rules instead of producing JSON.
        """
        return (
            "You are a senior quantitative macro strategist at a hedge fund. "
            "Your job is to act as a STRICT GATEKEEPER: evaluate whether incoming "
            "news and data constitute a genuine 'Regime Change' that requires "
            "portfolio reallocation.\n\n"
            "A regime change means:\n"
            "- A structural shift in macroeconomic conditions (rates, inflation, growth)\n"
            "- OR identifiable capital rotation between sectors or sub-sectors\n"
            "- OR a significant emotional arbitrage gap (market wildly over/under-reacting)\n\n"
            "Analyze the articles provided against the CURRENT MARKET BASELINE and "
            "score each factor on a scale of 1 (irrelevant) to 10 (extreme impact).\n\n"
            "=== SCORING RUBRIC ===\n\n"
            "1. MACRO IMPACT (M, 1-10):\n"
            "   - Do articles indicate a shift in rate policy, inflation trajectory,\n"
            "     GDP growth, employment, or geopolitical macro risk?\n"
            "   - 1-3: No macro signal. 4-6: Notable but not regime-altering.\n"
            "     7-8: Clear macro shift emerging. 9-10: Structural macro regime change.\n\n"
            "2. ROTATION INTENSITY (R, 1-10):\n"
            "   - Is there verifiable evidence of capital flowing FROM one sector\n"
            "     or sub-sector INTO another? Consider both broad sector rotation\n"
            "     and intra-sector shifts (e.g. Semiconductors -> Software, AI infra -> AI apps).\n"
            "   - Pay special attention to the portfolio's current sector weights\n"
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
            "     7-8: Significant over/under-reaction. 9-10: Extreme narrative disconnect."
        )

    def _build_contents(
        self,
        article_summaries: str,
        market_state: CurrentMarketState,
    ) -> str:
        """Build the user-facing content (market baseline data + articles).

        Contains only data — no instructions, no rules, no "Respond ONLY with..."
        boilerplate. The schema enforces JSON formatting at the API level."
        """
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

        return (
            "=== CURRENT MARKET BASELINE ===\n"
            f"Timestamp: {market_state.get('timestamp', 'unknown')}\n"
            f"Macro Regime: {macro.get('market_regime', 'unknown')}\n"
            f"Rate Trend: {macro.get('interest_rate_trend', 'unknown')}\n"
            f"Inflation Trend: {macro.get('inflation_trend', 'unknown')}\n"
            f"Cash Reserves: {alloc.get('cash_reserves_percent', 0)}%\n"
            f"Portfolio Value: ${alloc.get('total_value', 0):,.2f}\n"
            f"Sector Allocations:\n{sector_summary}\n\n"
            "=== ARTICLES TO ANALYZE ===\n\n"
            f"{article_summaries}"
        )

    # ------------------------------------------------------------------
    # Gemini call with native JSON mode + retry for transient errors
    # ------------------------------------------------------------------
    @staticmethod
    def _retry_regime_decorator(func):
        """Exponential-backoff retry for transient Gemini errors (429/503/5xx)."""
        return retry(
            wait=wait_exponential(multiplier=1, min=2, max=10),
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(Exception),
            after=lambda retry_state: print(
                f"    🔄 Regime Gemini retry attempt {retry_state.attempt_number}/3 "
                f"(waited {retry_state.outcome_timestamp - retry_state.start_time:.1f}s) "
                f"— error was: {retry_state.outcome.exception()}"
            ) if retry_state.outcome and retry_state.outcome.failed else None,
            before_sleep=lambda retry_state: print(
                f"    ⏳ Backing off {retry_state.next_action.sleep:.1f}s before retry..."
            ) if retry_state.next_action and retry_state.next_action.sleep else None,
        )(func)

    def _call_regime_gemini(self, system_instruction: str, contents: str) -> Optional[RegimeResponse]:
        """Call Gemini with native JSON mode. Returns typed Pydantic model or None."""
        @self._retry_regime_decorator
        def _sync_call():
            response = self._client.models.generate_content(
                model=REGIME_GEMINI_MODEL,
                contents=contents,
                config={
                    "temperature": REGIME_LLM_TEMPERATURE,
                    "max_output_tokens": REGIME_LLM_MAX_TOKENS,
                    "response_mime_type": "application/json",
                    "response_schema": RegimeResponse,
                    "system_instruction": system_instruction,
                },
            )

            # Guard against safety-filtered responses where .text is None
            if response.text is None:
                return None

            # Native JSON mode: response.parsed contains the typed model
            if hasattr(response, "parsed") and response.parsed is not None:
                return response.parsed
            # Fallback: parse from text
            return RegimeResponse.model_validate_json(response.text.strip())

        return _sync_call()

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
    Returns 'portfolio_manager' if:
      1. The regime analysis indicates portfolio manager action is needed, OR
      2. User theses from Slack gateway are pending (force_trigger_pm or user_theses populated),
         bypassing the significance-score gate entirely.
    Otherwise returns '__end__'.
    """
    # Check for forced PM trigger (user theses from Slack gateway)
    if state.get("force_trigger_pm", False) or state.get("user_theses"):
        return "portfolio_manager"
    # Standard regime-significance-based routing
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
        print(trigger_display + " " * max(0, 50 - len(trigger_display)) + "║")
    print(f"    ╠══════════════════════════════════════════════╣")
    print(f"    ║  Macro:     {analysis.Macro_Analysis[:80]}...{' ' * max(0, 74 - len(analysis.Macro_Analysis[:80]))}║")
    print(f"    ║  Rotation:  {analysis.Rotation_Analysis[:80]}...{' ' * max(0, 74 - len(analysis.Rotation_Analysis[:80]))}║")
    print(f"    ║  Arbitrage: {analysis.Emotional_Arbitrage_Analysis[:80]}...{' ' * max(0, 74 - len(analysis.Emotional_Arbitrage_Analysis[:80]))}║")
    print(f"    ╚══════════════════════════════════════════════╝")