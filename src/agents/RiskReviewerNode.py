"""
RiskReviewerNode.py - Agent 4: The Risk Reviewer (The Critic).

Triggered by the LangGraph conditional edge `route_after_portfolio_manager`
whenever Agent 3 emits a recommendation that survives its own no-trade
heuristic (`proceed_to_risk_reviewer == True` and `Proposed_Actions` non-empty).

This node is the LAST line of defense before a trade plan leaves the
system. It performs a "Dual-Check Evaluation":

  1. OPTIMIZATION CHECK
     Did the PM capitalize on the regime shift via SMART PROXIES (cheaper,
     purer, less obvious exposure), or did it chase the obvious overvalued
     headline ticker? Are the proposed actions internally consistent?
     Is the funding logic sound?

  2. FLAW DETECTION
     Scrutinize the broader news payload for contradictory articles,
     liquidity traps, sector concentration risks, or macro headwinds that
     punch holes in the PM's thesis.

Output: a `CriticFeedback` with approval_status, optimization_verdict,
risk_flaw_analysis, and (when rejected) a specific `critic_feedback` string
that gets piped back to Agent 3 for revision.

The revision loop is bounded by `RISK_REVIEW_MAX_ITERATIONS` (default 3) to
prevent the agent chain from getting stuck in an infinite disagreement.

All tunable constants live in `src/config.py`.
"""

from typing import List, Optional

from google import genai

from src.state import (
    ScoutArticle,
    GraphState,
    CurrentMarketState,
    RegimeAnalysis,
    PortfolioRecommendation,
    ProposedAction,
    CriticFeedback,
)
from src.config import (
    GEMINI_API_KEY,
    RISK_REVIEWER_GEMINI_MODEL,
    RISK_REVIEWER_TEMPERATURE,
    RISK_REVIEWER_MAX_TOKENS,
    RISK_REVIEW_MAX_ITERATIONS,
    RISK_MAX_SINGLE_POSITION_PCT,
    RISK_MAX_SECTOR_PCT,
)
from src.utils.json_repair import parse_json_with_repair


# =============================================================================
# Public LangGraph routing
# =============================================================================

def route_after_risk_reviewer(state: GraphState) -> str:
    """
    Conditional edge emitted after the Risk Reviewer node.

    Returns:
        "output_reporter" → if the most recent critic_feedback approved.
        "portfolio_manager" → if rejected AND we have iterations left.
        "__end__" → if rejected AND the iteration cap was hit.
    """
    feedback = state.get("critic_feedback")
    iterations = state.get("risk_review_iterations", 0)

    if feedback is None or feedback.approval_status:
        return "output_reporter"

    # Rejected. Have we burned through our iteration budget?
    if iterations >= RISK_REVIEW_MAX_ITERATIONS:
        return "__end__"

    return "portfolio_manager"


# =============================================================================
# Main node class
# =============================================================================

class RiskReviewerNode:
    """
    The Critic. One LLM call, low temperature, deterministic veto power.

    Public API:
        run_risk_reviewer_node(state) -> state
            Reads portfolio_recommendation + regime + articles from state,
            runs the dual-check prompt, writes `critic_feedback` back, and
            leaves routing to the caller via `route_after_risk_reviewer`.
    """

    def __init__(self):
        self._client: Optional[genai.Client] = None
        if GEMINI_API_KEY:
            self._client = genai.Client(api_key=GEMINI_API_KEY)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_risk_reviewer_node(self, state: GraphState) -> GraphState:
        """
        Single-pass critic invocation.

        State contract:
          IN:  state["portfolio_recommendation"], state["regime_analysis"],
               state["articles"], state["market_state"],
               state.get("previous_critic_feedback") (optional)
          OUT: state["critic_feedback"]   ← new CriticFeedback
               state["risk_review_iterations"] incremented
        """
        rec = state.get("portfolio_recommendation")
        regime = state.get("regime_analysis")
        articles = state.get("articles", [])
        market_state = state.get("market_state", {})

        # Nothing to review → auto-approve (no-op) so the graph proceeds.
        if rec is None or not rec.Proposed_Actions:
            iteration = (state.get("risk_review_iterations", 0) or 0) + 1
            feedback = CriticFeedback(
                optimization_verdict=(
                    "No portfolio recommendation to review (no-trade signal "
                    "from the PM). Auto-approving graph exit."
                ),
                risk_flaw_analysis="N/A — nothing proposed.",
                approval_status=True,
                critic_feedback="",
                iteration=iteration,
            )
            state["critic_feedback"] = feedback
            state["risk_review_iterations"] = iteration
            return state

        iteration = (state.get("risk_review_iterations", 0) or 0) + 1
        previous = state.get("previous_critic_feedback")

        print("\n  🛡️  Risk Reviewer (Agent 4) — Dual-Check Evaluation "
              f"(iteration {iteration}/{RISK_REVIEW_MAX_ITERATIONS})...")

        try:
            payload = self._build_review_payload(
                rec=rec, regime=regime, articles=articles, market_state=market_state,
                previous_feedback=previous, iteration=iteration,
            )
            prompt = self._build_critic_prompt(payload)
            feedback = self._run_critic(prompt, iteration)
        except Exception as e:
            print(f"    ⚠️  Risk Reviewer chain failed: {e}")
            feedback = self._fallback_feedback(iteration, reason=str(e))

        state["critic_feedback"] = feedback
        state["risk_review_iterations"] = iteration
        print_critic_verdict(feedback)
        return state

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------
    def _build_review_payload(
        self,
        rec: PortfolioRecommendation,
        regime: Optional[RegimeAnalysis],
        articles: List[ScoutArticle],
        market_state: CurrentMarketState,
        previous_feedback: Optional[CriticFeedback],
        iteration: int,
    ) -> str:
        """Compact text view of everything the critic needs to judge."""
        # ---- Portfolio Manager's recommendation ----
        rec_lines = [
            f"PM Portfolio_Impact_Assessment: {rec.Portfolio_Impact_Assessment}",
            f"PM Abstract_Proxy_Discoveries: {rec.Abstract_Proxy_Discoveries}",
            f"PM Momentum_vs_Valuation_Analysis: {rec.Momentum_vs_Valuation_Analysis}",
            "",
            "PM Proposed_Actions:",
        ]
        for i, a in enumerate(rec.Proposed_Actions, 1):
            rec_lines.append(
                f"  [{i}] {a.ticker} {a.action} {a.time_horizon}\n"
                f"      Reasoning: {a.reasoning}"
            )
        rec_text = "\n".join(rec_lines)

        # ---- Regime Analyst verdict ----
        regime_text = "N/A"
        if regime is not None:
            regime_text = (
                f"Macro: {regime.Macro_Analysis} (M={regime.macro_score}/10)\n"
                f"Rotation: {regime.Rotation_Analysis} (R={regime.rotation_score}/10)\n"
                f"Emotional_Arbitrage: {regime.Emotional_Arbitrage_Analysis} "
                f"(E={regime.emotional_arbitrage_score}/10)\n"
                f"Significance_Score: {regime.Significance_Score}/100"
            )

        # ---- Current portfolio snapshot ----
        alloc = market_state.get("portfolio_allocations", {})
        macro = market_state.get("macro_baseline", {})
        port_lines = [
            f"Cash: {alloc.get('cash_reserves_percent', 0)}% | "
            f"Total Value: ${alloc.get('total_value', 0):,.2f}",
            f"Macro Regime: {macro.get('market_regime', 'unknown')}",
        ]
        for sname, sdata in alloc.get("sectors", {}).items():
            port_lines.append(
                f"  - {sname}: {sdata.get('weight_percent', 0)}% "
                f"({', '.join(sdata.get('sub_sector_bias', []))})"
            )
            for h in sdata.get("holdings", []):
                port_lines.append(
                    f"      • {h.get('ticker')}: {h.get('concentration_percent', 0)}%"
                )
        port_text = "\n".join(port_lines)

        # ---- Top 5 articles by importance (for contradiction scanning) ----
        top_articles = sorted(articles, key=lambda a: a.importance_score, reverse=True)[:5]
        art_lines = []
        for i, a in enumerate(top_articles, 1):
            tone = ""
            if a.emotional_analysis:
                tone = (
                    f" | tonality={a.emotional_analysis.tonality_label} "
                    f"disparity={a.emotional_analysis.disparity_score:.2f}"
                )
            tickers = ", ".join(a.ticker_tags) if a.ticker_tags else "—"
            art_lines.append(
                f"  [{i}] {a.title[:140]}{tone}\n"
                f"      Source: {a.source_name} ({a.source_bucket}) | Tickers: {tickers}\n"
                f"      Summary: {(a.summary or (a.aggregated_content[:200] if a.aggregated_content else 'N/A'))[:200]}"
            )
        art_text = "\n".join(art_lines) if art_lines else "  (no articles)"

        # ---- Prior feedback (if this is a revision) ----
        prior_text = "None — first round of review."
        if previous_feedback is not None:
            prior_text = (
                f"Prior Optimisation Verdict: {previous_feedback.optimization_verdict}\n"
                f"Prior Risk Flaw Analysis: {previous_feedback.risk_flaw_analysis}\n"
                f"Prior Critic Feedback (what the PM was asked to fix):\n"
                f"  >>> {previous_feedback.critic_feedback}"
            )

        return (
            f"=== REVIEW ITERATION: {iteration} / {RISK_REVIEW_MAX_ITERATIONS} ===\n\n"
            f"=== PORTFOLIO MANAGER'S RECOMMENDATION (Agent 3) ===\n"
            f"{rec_text}\n\n"
            f"=== REGIME ANALYST VERDICT (Agent 2) ===\n"
            f"{regime_text}\n\n"
            f"=== CURRENT PORTFOLIO (Ground Truth) ===\n"
            f"{port_text}\n\n"
            f"=== ORIGINAL NEWS PAYLOAD (top 5 by importance) ===\n"
            f"{art_text}\n\n"
            f"=== PRIOR CRITIC FEEDBACK (if any) ===\n"
            f"{prior_text}\n"
        )

    def _build_critic_prompt(self, payload: str) -> str:
        return (
            "You are the Risk Reviewer (the Critic) at a hedge fund. You are the "
            "LAST line of defense before a trade plan is executed. You have VETO "
            "POWER. Be ruthless.\n\n"
            "Your job is to STRESS-TEST the Portfolio Manager's proposal against "
            "the original news flow, the current portfolio, and the regime verdict. "
            "You must perform a DUAL-CHECK EVALUATION:\n\n"
            "1. OPTIMIZATION CHECK\n"
            "   - Did the PM capitalize on the regime shift via SMART PROXIES "
            "(cheaper, purer, less obvious exposure) — or did it just chase the "
            "obvious, overvalued headline ticker?\n"
            "   - Are the proposed actions INTERNALLY CONSISTENT? (e.g. no EXPAND "
            "and DILUTE on the same ticker; no ADD funded without a DILUTE)\n"
            "   - Is the funding logic sound? If a new ADD is proposed, where is "
            "the money coming from?\n"
            "   - Are the time horizons appropriate for the catalyst?\n\n"
            "2. FLAW DETECTION\n"
            "   - Scan the news payload for CONTRADICTORY articles that invalidate "
            "the PM's thesis (e.g. macro headwinds, rate fears, negative earnings, "
            "regulatory action, fraud allegations).\n"
            "   - Look for LIQUIDITY TRAPS: low-float names, momentum rides that "
            "will reverse on a 5% down-day, illiquid option-chain names.\n"
            "   - Check for CONCENTRATION RISK: any single position or sector that "
            "after the trade would exceed reasonable tolerance for the portfolio.\n"
            "   - Verify TIME_HORIZON matches the catalyst — don't ride momentum "
            "tactically on a structural-thesis name, and vice versa.\n\n"
            "REVISION LOOP BEHAVIOR:\n"
            "  - This is iteration N out of N_MAX. If N >= N_MAX you will be "
            "automatically overridden regardless of your verdict — so be precise.\n"
            "  - If prior_critic_feedback is provided, the PM has already seen "
            "it. DO NOT repeat the same critique. Either:\n"
            "      (a) confirm the PM has addressed the prior issue and approve, OR\n"
            "      (b) identify a NEW, distinct flaw the PM missed.\n\n"
            f"{payload}\n"
            "=== REQUIRED JSON SCHEMA (strict) ===\n"
            "{\n"
            '  "optimization_verdict": "1-3 sentence verdict on proxy selection and consistency.",\n'
            '  "risk_flaw_analysis": "1-3 sentence identification of missed risks, contradictions, or concentration issues.",\n'
            '  "approval_status": true,\n'
            '  "critic_feedback": "EMPTY STRING IF approved. If rejected, a SPECIFIC, ACTIONABLE instruction to the PM (e.g. \\"Replace the EXPAND on NVDA with a DILUTE \\u2014 your cash sources do not add up\\"). Must be at least 30 characters and name specific tickers / actions to change."\n'
            '}\n\n'
            "DECISION RULES:\n"
            "  - Set approval_status=true ONLY if you find no material flaw.\n"
            "  - Set approval_status=false if ANY of: invalid funding logic, "
            "contradictory thesis, concentration risk above tolerance, or "
            "missed contradictory article in the payload.\n"
            "  - When rejecting, critic_feedback MUST be specific (ticker, action, "
            "reason). Vague feedback like 'improve reasoning' is not acceptable.\n"
            "  - Return ONLY the JSON object, no markdown, no preamble."
        )

    # ------------------------------------------------------------------
    # LLM call + validator
    # ------------------------------------------------------------------
    def _run_critic(self, prompt: str, iteration: int) -> CriticFeedback:
        if not self._client:
            return self._fallback_feedback(iteration, reason="no Gemini client available")

        response = self._client.models.generate_content(
            model=RISK_REVIEWER_GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": RISK_REVIEWER_TEMPERATURE,
                "max_output_tokens": RISK_REVIEWER_MAX_TOKENS,
            },
        )
        if response.text is None:
            raise ValueError("Risk Reviewer LLM response was safety-filtered (text=None)")
        text = self._strip_code_fences(response.text)
        data = parse_json_with_repair(text)

        approval = bool(data.get("approval_status", False))
        feedback = str(data.get("critic_feedback", "")).strip()

        # Rule: feedback must be non-empty when rejected, empty when approved.
        if not approval and len(feedback) < 30:
            feedback = (
                feedback
                + " [Critic feedback auto-extended — the original was too brief "
                  "to be actionable. The PM must make a specific, named change.]"
            )
        if approval:
            feedback = ""

        return CriticFeedback(
            optimization_verdict=str(data.get("optimization_verdict", "")).strip(),
            risk_flaw_analysis=str(data.get("risk_flaw_analysis", "")).strip(),
            approval_status=approval,
            critic_feedback=feedback,
            iteration=iteration,
        )

    # ------------------------------------------------------------------
    # Heuristic fallback (no Gemini)
    # ------------------------------------------------------------------
    def _fallback_feedback(self, iteration: int, reason: str = "") -> CriticFeedback:
        """
        No-LLM fallback. Returns an auto-approval so the graph can continue
        without an LLM.  A human reviewer remains the real Critic in that
        scenario, just like for every other agent's heuristic path.
        """
        return CriticFeedback(
            optimization_verdict=(
                f"Heuristic (no-LLM) auto-approval issued. Reason: {reason or 'unknown'}. "
                "No critic LLM available to dual-check the PM's plan; the human "
                "reviewer at the terminal step is the real Critic for this run."
            ),
            risk_flaw_analysis=(
                "Heuristic fallback did not perform flaw detection. "
                "Default position: trust the PM unless the proposal violates "
                "hard-rule thresholds."
            ),
            approval_status=True,
            critic_feedback="",
            iteration=iteration,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_code_fences(text: str) -> str:
        s = text.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[-1]
            if s.endswith("```"):
                s = s.rsplit("```", 1)[0]
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
        return s.strip()


# =============================================================================
# Pretty-printing
# =============================================================================

def print_critic_verdict(feedback: CriticFeedback) -> None:
    """Formatted console output of the critic's verdict."""
    print(f"\n    ╔══════════════════════════════════════════════════════════════╗")
    print(f"    ║  RISK REVIEWER VERDICT (Agent 4)  — iter {feedback.iteration:<2} of "
          f"{RISK_REVIEW_MAX_ITERATIONS}                ║")
    print(f"    ╠══════════════════════════════════════════════════════════════╣")
    verdict = "✅ APPROVED" if feedback.approval_status else "❌ REJECTED — revise"
    print(f"    ║  {verdict:<60} ║")
    print(f"    ╠══════════════════════════════════════════════════════════════╣")
    print(f"    ║  OPTIMIZATION VERDICT:                                       ║")
    for line in _wrap(feedback.optimization_verdict, 60):
        print(f"    ║    {line:<60} ║")
    print(f"    ║  RISK FLAW ANALYSIS:                                        ║")
    for line in _wrap(feedback.risk_flaw_analysis, 60):
        print(f"    ║    {line:<60} ║")
    print(f"    ╠══════════════════════════════════════════════════════════════╣")
    if feedback.approval_status:
        print(f"    ║  → Forward to Agent 5 (Output Reporter)                      ║")
    else:
        print(f"    ║  CRITIC FEEDBACK (to be sent back to Agent 3):               ║")
        for line in _wrap(feedback.critic_feedback, 60):
            print(f"    ║    {line:<60} ║")
        print(f"    ║  → Loop back to Agent 3 (Portfolio Manager) for revision    ║")
    print(f"    ╚══════════════════════════════════════════════════════════════╝")


def _wrap(text: str, width: int) -> List[str]:
    """Simple word-wrap into a list of strings of at most `width` chars."""
    if not text:
        return [""]
    words = text.split()
    lines: List[str] = []
    current = ""
    for w in words:
        if not current:
            current = w
        elif len(current) + 1 + len(w) <= width:
            current += " " + w
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines if lines else [""]
