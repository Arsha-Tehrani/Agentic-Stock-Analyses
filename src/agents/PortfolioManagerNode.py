"""
PortfolioManagerNode.py - Agent 3: The Optimizer & Execution Strategist.

This agent is invoked by the LangGraph conditional edge `route_after_regime_analyst`
whenever the Regime Analyst (Agent 2) flags a Significance_Score > 70.

Because the cognitive load of combining "abstract market research" with "portfolio
mathematics" is too high for a single LLM call, the Portfolio Manager is implemented
as TWO sequential LLM chains inside a single node:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STEP 1 — THE QUANT RESEARCHER (Abstract Discovery & Proxy Hunting)    │
    │  ------------------------------------------------------------------------│
    │  - Reads the Significant_Regime_Payload from Agent 2                    │
    │  - Reads my Current_Market_State                                        │
    │  - Asks the LLM to formulate 3-4 DuckDuckGo search queries              │
    │  - Executes those queries via the `ddgs` library                        │
    │  - Optional enrichment: market caps + prices via Finnhub (aiohttp)     │
    │  - Asks the LLM a SECOND time to synthesize snippets into a structured  │
    │    Target_Research_List of HEADLINE / PROXY / COMPETITOR / SUPPLIER     │
    │    tickers with momentum & valuation theses                             │
    │  - Output: TargetResearchList                                          │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STEP 2 — THE BOOK RUNNER (Allocation & Risk Management)               │
    │  ------------------------------------------------------------------------│
    │  - Receives the Target_Research_List AND the Current_Market_State      │
    │  - Compares proposed positions against current sector weightings       │
    │  - Asks the LLM to emit a strict JSON trade plan using the required     │
    │    schema: Portfolio_Impact_Assessment, Abstract_Proxy_Discoveries,    │
    │    Momentum_vs_Valuation_Analysis, and Proposed_Actions[]              │
    │  - Python validator enforces: valid action vocabulary, valid horizons,│
    │    cap on number of actions, ticker in research list or current port.  │
    │  - Output: PortfolioRecommendation  →  routes to Agent 4 (Risk Reviewer)│
    └─────────────────────────────────────────────────────────────────────────┘

The whole node is wrapped in a single try/except boundary so partial failures
are not committed to GraphState.

All tunable constants live in src/config.py.
"""

import asyncio
import json
from typing import List, Optional

import aiohttp
from ddgs import DDGS
from google import genai

from src.state import (
    ScoutArticle,
    GraphState,
    CurrentMarketState,
    RegimeAnalysis,
    TargetStock,
    TargetResearchList,
    ProposedAction,
    PortfolioRecommendation,
)
from src.config import (
    GEMINI_API_KEY,
    PORTFOLIO_GEMINI_MODEL,
    PORTFOLIO_RESEARCHER_TEMPERATURE,
    PORTFOLIO_RESEARCHER_MAX_TOKENS,
    PORTFOLIO_ALLOCATOR_TEMPERATURE,
    PORTFOLIO_ALLOCATOR_MAX_TOKENS,
    PORTFOLIO_DDG_MAX_RESULTS,
    PORTFOLIO_DDG_QUERIES,
    PORTFOLIO_MAX_PROPOSED_ACTIONS,
    PORTFOLIO_VALID_ACTIONS,
    PORTFOLIO_VALID_HORIZONS,
    PORTFOLIO_FINNHUB_BASE,
    PORTFOLIO_FINNHUB_TIMEOUT,
    WIRE_API_KEY,
    REGIME_DEFAULT_MARKET_STATE,
)


# =============================================================================
# Public LangGraph routing
# =============================================================================

def route_after_portfolio_manager(state: GraphState) -> str:
    """
    Conditional edge emitted after the Portfolio Manager node.

    Returns the name of the next node when the agent chain should continue,
    or the special ``__end__`` sentinel when it should stop.

    Today only Agent 4 (The Risk Reviewer) is the legitimate next hop, but
    that node is not yet implemented; we still register the edge so the
    graph wiring is correct and the pipeline can be completed once Agent 4
    lands. When a non-recommendation is produced (e.g. heuristic fallback
    with no actions) we return ``__end__`` so the graph stops cleanly.
    """
    rec = state.get("portfolio_recommendation")
    if rec is None:
        return "__end__"
    if not rec.Proposed_Actions:
        return "__end__"
    return "risk_reviewer"


# =============================================================================
# Main node class
# =============================================================================

class PortfolioManagerNode:
    """
    Two-chain portfolio decision engine. Mirrors the structure of
    RegimeAnalystNode (single class, single ``run_*_node`` entry point)
    while internally splitting the work into a Researcher and an Allocator.
    """

    def __init__(self):
        self._client: Optional[genai.Client] = None
        if GEMINI_API_KEY:
            self._client = genai.Client(api_key=GEMINI_API_KEY)

    # ------------------------------------------------------------------
    # Public API (called by run_pipeline.py / LangGraph)
    # ------------------------------------------------------------------
    def run_portfolio_manager_node(self, state: GraphState) -> GraphState:
        """
        Main entry point. Accepts the full GraphState, runs both internal
        chains, and writes the resulting ``PortfolioRecommendation`` back
        into state.

        Returns the mutated GraphState with:
          - portfolio_recommendation populated
          - proceed_to_risk_reviewer-style flag implied by recommendation
        """
        articles = state.get("articles", [])
        market_state = state.get("market_state", REGIME_DEFAULT_MARKET_STATE)
        regime_analysis = state.get("regime_analysis")

        if regime_analysis is None:
            print("\n  ⚠️  No regime_analysis in state. Skipping Portfolio Manager.")
            state["portfolio_recommendation"] = None
            return state

        print("\n  💼 Portfolio Manager: spinning up Researcher + Book Runner...")

        try:
            # Step 1 — the Quant Researcher
            target_list = self._run_researcher_chain(articles, regime_analysis, market_state)
            print(f"    🔍 Researcher surfaced {len(target_list.targets)} target(s) across "
                  f"{len(target_list.queries_used)} DDG query/queries.")

            # Optional: enrich with live market caps from Finnhub (aiohttp)
            target_list = self._enrich_with_finnhub(target_list)

            # Step 2 — the Book Runner
            recommendation = self._run_allocator_chain(
                target_list, market_state, regime_analysis
            )
            print(f"    📋 Book Runner emitted {len(recommendation.Proposed_Actions)} "
                  f"validated action(s).")
            print_recommendation(recommendation)

        except Exception as e:
            print(f"    ⚠️  Portfolio Manager chain failed: {e}")
            recommendation = self._fallback_recommendation(regime_analysis, str(e))
            print_recommendation(recommendation)

        state["portfolio_recommendation"] = recommendation
        return state

    # ==================================================================
    # STEP 1 — THE QUANT RESEARCHER
    # ==================================================================

    def _run_researcher_chain(
        self,
        articles: List[ScoutArticle],
        regime_analysis: RegimeAnalysis,
        market_state: CurrentMarketState,
    ) -> TargetResearchList:
        """
        1a. Ask LLM for 3-4 DuckDuckGo search queries.
        1b. Execute those queries via ddgs.
        1c. Ask LLM to synthesize snippets + original payload into a structured
            Target_Research_List with HEADLINE/PROXY/COMPETITOR/SUPPLIER roles.
        """
        if not self._client:
            return self._heuristic_researcher(articles, market_state)

        # Build the payload for the LLM
        regime_payload = self._build_regime_payload(articles, regime_analysis, market_state)

        # 1a — LLM formulates the search queries
        queries_prompt = self._build_queries_prompt(regime_payload)
        try:
            response = self._client.models.generate_content(
                model=PORTFOLIO_GEMINI_MODEL,
                contents=queries_prompt,
                config={
                    "temperature": PORTFOLIO_RESEARCHER_TEMPERATURE,
                    "max_output_tokens": PORTFOLIO_RESEARCHER_MAX_TOKENS,
                },
            )
            queries_text = self._strip_code_fences(response.text)
            queries_data = json.loads(queries_text)
            raw_queries = queries_data.get("queries", [])
            # Filter to clean, non-empty strings; cap at config limit
            queries = [str(q).strip() for q in raw_queries if str(q).strip()]
            queries = queries[:PORTFOLIO_DDG_QUERIES]
            if not queries:
                raise ValueError("LLM returned no usable search queries")
        except Exception as e:
            print(f"    ⚠️  Researcher query generation failed: {e}")
            # Fall back to extracting the article's ticker tags as pseudo-queries
            queries = self._fallback_queries(articles)

        # 1b — Execute DDG searches (sync DDGS is fine for the small number of queries)
        snippets_by_query = self._execute_ddg_queries(queries)

        # 1c — LLM synthesizes the snippets into a Target_Research_List
        synth_prompt = self._build_synthesis_prompt(
            regime_payload=regime_payload,
            queries=queries,
            snippets_by_query=snippets_by_query,
        )
        try:
            response = self._client.models.generate_content(
                model=PORTFOLIO_GEMINI_MODEL,
                contents=synth_prompt,
                config={
                    "temperature": PORTFOLIO_RESEARCHER_TEMPERATURE,
                    "max_output_tokens": PORTFOLIO_RESEARCHER_MAX_TOKENS,
                },
            )
            synth_text = self._strip_code_fences(response.text)
            synth_data = json.loads(synth_text)

            targets: List[TargetStock] = []
            for t in synth_data.get("targets", []):
                try:
                    targets.append(
                        TargetStock(
                            ticker=str(t.get("ticker", "")).upper().strip(),
                            company_name=str(t.get("company_name", "")).strip(),
                            role=str(t.get("role", "HEADLINE")).upper().strip(),
                            momentum_flag=bool(t.get("momentum_flag", False)),
                            valuation_thesis=str(t.get("valuation_thesis", "")).strip(),
                            momentum_thesis=str(t.get("momentum_thesis", "")).strip(),
                            evidence=[str(e) for e in t.get("evidence", [])][:5],
                        )
                    )
                except Exception:
                    continue
            # Drop entries with no ticker
            targets = [t for t in targets if t.ticker]

            return TargetResearchList(
                targets=targets,
                queries_used=queries,
                research_summary=str(synth_data.get("research_summary", "")).strip(),
            )

        except Exception as e:
            print(f"    ⚠️  Researcher synthesis failed: {e}")
            # Fall back to heuristic list built from article ticker tags
            return self._heuristic_researcher(articles, market_state, queries)

    # ------------------------------------------------------------------
    # Researcher: prompt builders
    # ------------------------------------------------------------------
    def _build_regime_payload(
        self,
        articles: List[ScoutArticle],
        regime_analysis: RegimeAnalysis,
        market_state: CurrentMarketState,
    ) -> str:
        """Compact text view of Agent 2's payload for the Researcher LLM."""
        macro = market_state.get("macro_baseline", {})
        alloc = market_state.get("portfolio_allocations", {})
        sectors = alloc.get("sectors", {})

        sector_lines = []
        for name, data in sectors.items():
            bias = ", ".join(data.get("sub_sector_bias", []))
            top_holdings = sorted(
                data.get("holdings", []),
                key=lambda h: h.get("concentration_percent", 0),
                reverse=True,
            )[:3]
            top_str = ", ".join(
                f"{h.get('ticker')} ({h.get('concentration_percent', 0)}%)"
                for h in top_holdings
            )
            sector_lines.append(
                f"  - {name}: {data.get('weight_percent', 0)}% | "
                f"Focus: {bias} | Top: {top_str}"
            )
        sector_summary = "\n".join(sector_lines) if sector_lines else "  (none)"

        # Pull the most significant articles (cap at 5 for prompt size)
        sig_articles = sorted(
            articles,
            key=lambda a: a.importance_score,
            reverse=True,
        )[:5]

        article_lines = []
        for i, a in enumerate(sig_articles, 1):
            tone = ""
            if a.emotional_analysis:
                tone = (
                    f" | tonality={a.emotional_analysis.tonality_label} "
                    f"disparity={a.emotional_analysis.disparity_score:.2f}"
                )
            tickers = ", ".join(a.ticker_tags) if a.ticker_tags else "—"
            article_lines.append(
                f"[{i}] {a.title[:140]}\n"
                f"    Source: {a.source_name} ({a.source_bucket}){tone}\n"
                f"    Tickers: {tickers}\n"
                f"    Summary: {(a.summary or (a.aggregated_content[:200] if a.aggregated_content else 'N/A'))[:200]}"
            )
        articles_summary = "\n\n".join(article_lines) if article_lines else "  (no articles)"

        return (
            "=== REGIME ANALYST VERDICT (Agent 2) ===\n"
            f"Macro: {regime_analysis.Macro_Analysis}\n"
            f"Rotation: {regime_analysis.Rotation_Analysis}\n"
            f"Emotional Arbitrage: {regime_analysis.Emotional_Arbitrage_Analysis}\n"
            f"Significance Score: {regime_analysis.Significance_Score}/100 "
            f"(M={regime_analysis.macro_score}/10, R={regime_analysis.rotation_score}/10, "
            f"E={regime_analysis.emotional_arbitrage_score}/10)\n\n"
            "=== CURRENT MARKET BASELINE ===\n"
            f"Macro Regime: {macro.get('market_regime', 'unknown')}\n"
            f"Rate Trend: {macro.get('interest_rate_trend', 'unknown')}\n"
            f"Inflation Trend: {macro.get('inflation_trend', 'unknown')}\n"
            f"Cash Reserves: {alloc.get('cash_reserves_percent', 0)}%\n"
            f"Total Portfolio Value: ${alloc.get('total_value', 0):,.2f}\n"
            f"Sector Allocations:\n{sector_summary}\n\n"
            "=== SIGNIFICANT ARTICLES (top 5 by importance) ===\n"
            f"{articles_summary}\n"
        )

    def _build_queries_prompt(self, regime_payload: str) -> str:
        """Ask the LLM for 3-4 web search queries to find proxies and momentum clues."""
        return (
            "You are a senior research analyst at a hedge fund preparing a "
            "second-order thesis on a significant market event.\n\n"
            "Given the REGIME PAYLOAD below, formulate exactly "
            f"{PORTFOLIO_DDG_QUERIES} highly targeted DuckDuckGo search queries "
            "that will help you find:\n"
            "  1. Upstream SUPPLIERS or downstream DISTRIBUTORS (the value chain)\n"
            "  2. Direct COMPETITORS that may offer cheaper exposure\n"
            "  3. Historical PATTERNS or PRECEDENTS of similar rotations\n"
            "  4. PROXY stocks that have historically tracked the headline theme\n\n"
            "Rules:\n"
            "- Each query is 4-9 keywords.\n"
            "- Include specific company / ticker / sector names where possible.\n"
            "- Mix supply-chain queries, competitor queries, and historical-pattern queries.\n"
            "- DO NOT include generic queries like 'stock market news'.\n"
            "- Return ONLY valid JSON in this exact format:\n"
            '{"queries": ["query one", "query two", "query three", "query four"]}\n\n'
            f"{regime_payload}"
        )

    def _build_synthesis_prompt(
        self,
        regime_payload: str,
        queries: List[str],
        snippets_by_query: dict,
    ) -> str:
        """Ask the LLM to turn DDG snippets into a structured Target_Research_List."""
        snippet_lines = []
        for q in queries:
            snippets = snippets_by_query.get(q, [])
            if not snippets:
                snippet_lines.append(f"--- Query: {q} ---\n(no results)\n")
                continue
            joined = "\n".join(f"  - {s}" for s in snippets)
            snippet_lines.append(f"--- Query: {q} ---\n{joined}\n")

        snippets_text = "\n".join(snippet_lines)

        return (
            "You are a senior research analyst. You just executed a batch of "
            "DuckDuckGo searches related to a significant market event.\n\n"
            "Your job is to SYNTHESIZE the snippets below into a structured "
            "Target_Research_List that the portfolio book-runner can act on.\n\n"
            "For EACH target you surface, classify its role:\n"
            "  - HEADLINE: the actual ticker named in the news\n"
            "  - PROXY: a substitute that offers cheaper / safer / purer exposure\n"
            "  - COMPETITOR: a direct peer to the headline ticker\n"
            "  - SUPPLIER: an upstream beneficiary in the value chain\n\n"
            "For each target, also evaluate MOMENTUM:\n"
            "  - momentum_flag=true if the asset looks overvalued BUT is currently "
            "    riding a strong institutional / earnings momentum wave\n"
            "  - momentum_flag=false if it's a pure value play\n\n"
            "=== REGIME PAYLOAD ===\n"
            f"{regime_payload}\n"
            "=== DDG SEARCH RESULTS ===\n"
            f"{snippets_text}\n"
            "=== REQUIRED OUTPUT FORMAT ===\n\n"
            "Respond ONLY with valid JSON in this exact format:\n"
            "{\n"
            '  "research_summary": "One-paragraph synthesis of what the searches revealed.",\n'
            '  "targets": [\n'
            '    {\n'
            '      "ticker": "TICK",\n'
            '      "company_name": "Company Inc.",\n'
            '      "role": "HEADLINE | PROXY | COMPETITOR | SUPPLIER",\n'
            '      "momentum_flag": false,\n'
            '      "valuation_thesis": "1-2 sentence value analysis",\n'
            '      "momentum_thesis": "1-2 sentence momentum analysis",\n'
            '      "evidence": ["short snippet 1", "short snippet 2"]\n'
            '    }\n'
            '  ]\n'
            '}\n\n'
            "Surface AT LEAST the headline ticker(s) plus 2-3 PROXY/COMPETITOR/SUPPLIER picks.\n"
            "Each evidence string must be 5-25 words copied or paraphrased from the snippets."
        )

    # ------------------------------------------------------------------
    # Researcher: tool execution (DDG + Finnhub)
    # ------------------------------------------------------------------
    def _execute_ddg_queries(self, queries: List[str]) -> dict:
        """Run each query against DuckDuckGo and collect a flat snippet list per query."""
        results: dict = {}
        for q in queries:
            snippets: List[str] = []
            try:
                with DDGS() as ddgs:
                    for i, r in enumerate(ddgs.text(q, max_results=PORTFOLIO_DDG_MAX_RESULTS)):
                        title = (r.get("title") or "").strip()
                        body = (r.get("body") or "").strip()
                        if body:
                            if title:
                                snippets.append(f"{title}: {body}")
                            else:
                                snippets.append(body)
                results[q] = snippets[:PORTFOLIO_DDG_MAX_RESULTS]
            except Exception as e:
                print(f"    ⚠️  DDG query failed for '{q}': {e}")
                results[q] = []
        return results

    def _enrich_with_finnhub(self, target_list: TargetResearchList) -> TargetResearchList:
        """
        Optional: fire concurrent aiohttp requests to Finnhub for live market cap + price
        on every distinct ticker in the target list. Failures are non-fatal; missing
        values stay None. Requires an active event loop or runs via asyncio.run().
        """
        tickers = sorted(target_list.ticker_set())
        if not tickers or not WIRE_API_KEY:
            return target_list

        try:
            enrichments = asyncio.run(
                self._async_finnhub_enrich(tickers)
            )
        except RuntimeError:
            # Already inside an event loop (e.g. inside run_pipeline's main_pipeline).
            # Try the alternative: nest_asyncio / new loop. We do a simple fallback.
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Cannot run nested loop → fall back to synchronous-ish fetching
                    # with a short per-ticker timeout using a fresh session.
                    return self._sync_finnhub_enrich(target_list, tickers)
                enrichments = loop.run_until_complete(
                    self._async_finnhub_enrich(tickers)
                )
            except Exception as e:
                print(f"    ⚠️  Finnhub enrichment loop error: {e}")
                return target_list
        except Exception as e:
            print(f"    ⚠️  Finnhub enrichment failed: {e}")
            return target_list

        # Apply enrichments back to the targets
        for t in target_list.targets:
            data = enrichments.get(t.ticker.upper())
            if data:
                t.market_cap = data.get("market_cap")
                t.current_price = data.get("current_price")
        return target_list

    async def _async_finnhub_enrich(self, tickers: List[str]) -> dict:
        """Concurrent aiohttp fetches for profile2 (market cap) + quote (price)."""
        timeout = aiohttp.ClientTimeout(total=PORTFOLIO_FINNHUB_TIMEOUT)
        results: dict = {}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [self._fetch_one_ticker(session, t) for t in tickers]
            pairs = await asyncio.gather(*tasks, return_exceptions=True)
            for t, pair in zip(tickers, pairs):
                if isinstance(pair, Exception) or pair is None:
                    results[t.upper()] = None
                else:
                    results[t.upper()] = pair
        return results

    async def _fetch_one_ticker(
        self, session: "aiohttp.ClientSession", ticker: str
    ) -> Optional[dict]:
        """Fetch market cap from /stock/profile2 and current price from /quote."""
        out: dict = {}
        try:
            profile_url = (
                f"{PORTFOLIO_FINNHUB_BASE}/stock/profile2"
                f"?symbol={ticker}&token={WIRE_API_KEY}"
            )
            async with session.get(profile_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    mc = data.get("marketCapitalization")
                    if isinstance(mc, (int, float)):
                        # Finnhub returns market cap in MILLIONS of USD
                        out["market_cap"] = float(mc) * 1_000_000
        except Exception:
            pass
        try:
            quote_url = (
                f"{PORTFOLIO_FINNHUB_BASE}/quote"
                f"?symbol={ticker}&token={WIRE_API_KEY}"
            )
            async with session.get(quote_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    cp = data.get("c")  # current price
                    if isinstance(cp, (int, float)):
                        out["current_price"] = float(cp)
        except Exception:
            pass
        return out if out else None

    def _sync_finnhub_enrich(
        self, target_list: TargetResearchList, tickers: List[str]
    ) -> TargetResearchList:
        """Best-effort sync fallback when called from inside a running event loop."""
        try:
            import requests  # type: ignore
        except ImportError:
            print("    ⚠️  `requests` not installed — skipping Finnhub enrichment.")
            return target_list

        for t in target_list.targets:
            try:
                r1 = requests.get(
                    f"{PORTFOLIO_FINNHUB_BASE}/stock/profile2",
                    params={"symbol": t.ticker, "token": WIRE_API_KEY},
                    timeout=PORTFOLIO_FINNHUB_TIMEOUT,
                )
                if r1.status_code == 200:
                    mc = r1.json().get("marketCapitalization")
                    if isinstance(mc, (int, float)):
                        t.market_cap = float(mc) * 1_000_000
            except Exception:
                pass
            try:
                r2 = requests.get(
                    f"{PORTFOLIO_FINNHUB_BASE}/quote",
                    params={"symbol": t.ticker, "token": WIRE_API_KEY},
                    timeout=PORTFOLIO_FINNHUB_TIMEOUT,
                )
                if r2.status_code == 200:
                    cp = r2.json().get("c")
                    if isinstance(cp, (int, float)):
                        t.current_price = float(cp)
            except Exception:
                pass
        return target_list

    # ------------------------------------------------------------------
    # Researcher: heuristics & fallbacks
    # ------------------------------------------------------------------
    def _heuristic_researcher(
        self,
        articles: List[ScoutArticle],
        market_state: CurrentMarketState,
        queries: Optional[List[str]] = None,
    ) -> TargetResearchList:
        """
        No-LLM fallback: build a minimal Target_Research_List from the ticker
        tags already present in the ScoutArticles, marked as HEADLINE role.
        """
        seen: set = set()
        targets: List[TargetStock] = []
        for a in articles:
            for t in a.ticker_tags:
                tk = t.upper().strip()
                if not tk or tk in seen:
                    continue
                seen.add(tk)
                targets.append(
                    TargetStock(
                        ticker=tk,
                        company_name=tk,  # unknown without LLM
                        role="HEADLINE",
                        momentum_flag=False,
                        valuation_thesis="Heuristic: no LLM available for thesis generation.",
                        momentum_thesis="Heuristic: momentum not evaluated.",
                        evidence=[],
                    )
                )

        return TargetResearchList(
            targets=targets,
            queries_used=queries or [],
            research_summary=(
                "Heuristic fallback — no LLM available. Only tickers explicitly "
                "tagged in source articles are surfaced; no proxy discovery was "
                "performed."
            ),
        )

    def _fallback_queries(self, articles: List[ScoutArticle]) -> List[str]:
        """Build pseudo-queries from the most-mentioned tickers in the article batch."""
        tickers: List[str] = []
        for a in articles:
            tickers.extend(a.ticker_tags)
        # Deduplicate while preserving order
        seen: set = set()
        unique = []
        for t in tickers:
            tk = t.upper().strip()
            if tk and tk not in seen:
                seen.add(tk)
                unique.append(tk)
        queries = [f"{tk} stock price news" for tk in unique[:PORTFOLIO_DDG_QUERIES]]
        if not queries:
            queries = ["market rotation capital flows news"]
        return queries[:PORTFOLIO_DDG_QUERIES]

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Robustly strip ```json ... ``` or ``` ... ``` markdown fences from LLM output."""
        s = text.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[-1]
            if s.endswith("```"):
                s = s.rsplit("```", 1)[0]
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
        return s.strip()

    # ==================================================================
    # STEP 2 — THE BOOK RUNNER (Allocation & Risk Management)
    # ==================================================================

    def _run_allocator_chain(
        self,
        target_list: TargetResearchList,
        market_state: CurrentMarketState,
        regime_analysis: RegimeAnalysis,
    ) -> PortfolioRecommendation:
        """
        Single LLM call that compares the Target_Research_List against the
        Current_Market_State and emits the strict JSON trade plan.

        The output is then run through a Python validator that:
          - Coerces Action into {EXPAND, DILUTE, ADD}
          - Coerces Time_Horizon into {SHORT_TERM_MOMENTUM, LONG_TERM_HOLD}
          - Caps the action list at PORTFOLIO_MAX_PROPOSED_ACTIONS
          - Requires the ticker to be in either the research list OR the
            current portfolio (otherwise the action is dropped)
        """
        if not self._client:
            return self._fallback_recommendation(
                regime_analysis, reason="no Gemini client available"
            )

        alloc_prompt = self._build_allocator_prompt(
            target_list, market_state, regime_analysis
        )

        response = self._client.models.generate_content(
            model=PORTFOLIO_GEMINI_MODEL,
            contents=alloc_prompt,
            config={
                "temperature": PORTFOLIO_ALLOCATOR_TEMPERATURE,
                "max_output_tokens": PORTFOLIO_ALLOCATOR_MAX_TOKENS,
            },
        )
        text = self._strip_code_fences(response.text)
        data = json.loads(text)

        # Validate & coerce the proposed actions
        valid_actions = self._validate_proposed_actions(
            data.get("Proposed_Actions", []),
            target_list,
            market_state,
        )

        return PortfolioRecommendation(
            Portfolio_Impact_Assessment=str(
                data.get("Portfolio_Impact_Assessment", "")
            ).strip(),
            Abstract_Proxy_Discoveries=str(
                data.get("Abstract_Proxy_Discoveries", "")
            ).strip(),
            Momentum_vs_Valuation_Analysis=str(
                data.get("Momentum_vs_Valuation_Analysis", "")
            ).strip(),
            Proposed_Actions=valid_actions,
            proceed_to_risk_reviewer=len(valid_actions) > 0,
            regime_significance_score=regime_analysis.Significance_Score,
            research_summary=target_list.research_summary,
            queries_used=target_list.queries_used,
        )

    def _build_allocator_prompt(
        self,
        target_list: TargetResearchList,
        market_state: CurrentMarketState,
        regime_analysis: RegimeAnalysis,
    ) -> str:
        # ---- Current portfolio view ----
        alloc = market_state.get("portfolio_allocations", {})
        macro = market_state.get("macro_baseline", {})
        sectors = alloc.get("sectors", {})

        portfolio_lines = [f"Cash: {alloc.get('cash_reserves_percent', 0)}%",
                           f"Total Value: ${alloc.get('total_value', 0):,.2f}"]
        for sname, sdata in sectors.items():
            portfolio_lines.append(
                f"  - {sname}: {sdata.get('weight_percent', 0)}% "
                f"({', '.join(sdata.get('sub_sector_bias', []))})"
            )
            for h in sdata.get("holdings", []):
                portfolio_lines.append(
                    f"      • {h.get('ticker')}: {h.get('concentration_percent', 0)}%"
                )
        portfolio_text = "\n".join(portfolio_lines)

        # ---- Target list view ----
        target_lines = []
        for i, t in enumerate(target_list.targets, 1):
            mc_str = f" | mkt cap ${t.market_cap/1e9:.2f}B" if t.market_cap else ""
            px_str = f" | px ${t.current_price:.2f}" if t.current_price else ""
            target_lines.append(
                f"  [{i}] {t.ticker} ({t.role}) — {t.company_name}{mc_str}{px_str}\n"
                f"      Momentum: {'YES' if t.momentum_flag else 'no'}\n"
                f"      Valuation: {t.valuation_thesis}\n"
                f"      Momentum thesis: {t.momentum_thesis}"
            )
        targets_text = "\n".join(target_lines) if target_lines else "  (no targets)"

        return (
            "You are a senior book-runner at a hedge fund. You have just received "
            "a Target_Research_List from the Research desk after a major regime-change "
            "event. Your job is to produce a STRICT JSON trade plan that the risk "
            "reviewer can stress-test.\n\n"
            "DECISION LOGIC:\n"
            "  1. COMPARE the targets against the CURRENT PORTFOLIO. If we are already "
            "     heavily weighted in a sector, do NOT just ADD another position in that "
            "     sector — DILUTE an existing overweight to fund a new proxy.\n"
            "  2. DILUTE positions whose underlying thesis is invalidated by the news.\n"
            "  3. EXPAND positions where the news CONFIRMS the structural thesis.\n"
            "  4. ADD new positions only when the research surfaced a ticker NOT currently held.\n"
            "  5. When a target has momentum_flag=true, lean SHORT_TERM_MOMENTUM unless "
            "     the underlying thesis is structural, in which case LONG_TERM_HOLD.\n\n"
            f"REGIME SIGNIFICANCE: {regime_analysis.Significance_Score}/100\n"
            f"MACRO REGIME: {macro.get('market_regime', 'unknown')}\n\n"
            "=== CURRENT PORTFOLIO ===\n"
            f"{portfolio_text}\n\n"
            "=== TARGET RESEARCH LIST (from Agent 3 Researcher) ===\n"
            f"{targets_text}\n\n"
            "=== REQUIRED JSON SCHEMA (strict — do not deviate) ===\n"
            "{\n"
            '  "Portfolio_Impact_Assessment": "Detailed analysis of how the current portfolio stacks up against the incoming news and where our blind spots / over-exposures are.",\n'
            '  "Abstract_Proxy_Discoveries": "Explanation of the deep-dive research. Document the unmentioned companies, supply chain beneficiaries, or competitors discovered during the DDG search.",\n'
            '  "Momentum_vs_Valuation_Analysis": "Identification of historical patterns. Note which assets are pure value plays and which are momentum ride-the-wave plays despite high valuations.",\n'
            '  "Proposed_Actions": [\n'
            '    {\n'
            '      "Ticker": "Stock Symbol",\n'
            '      "Action": "EXPAND | DILUTE | ADD",\n'
            '      "Time_Horizon": "SHORT_TERM_MOMENTUM | LONG_TERM_HOLD",\n'
            '      "Reasoning": "Highly detailed justification tying the proxy discovery, momentum pattern, and current portfolio risk tolerance together to justify this exact move."\n'
            '    }\n'
            '  ]\n'
            '}\n\n'
            "STRICT RULES:\n"
            f"- Proposed_Actions length MUST be between 1 and {PORTFOLIO_MAX_PROPOSED_ACTIONS}.\n"
            "- Every Ticker in Proposed_Actions must appear in EITHER the current portfolio OR the Target Research List above.\n"
            "- Action MUST be one of EXPAND, DILUTE, or ADD (case-sensitive).\n"
            "- Time_Horizon MUST be one of SHORT_TERM_MOMENTUM, LONG_TERM_HOLD.\n"
            "- If you are already 30%+ in a sector, the new position in that sector must be funded by a DILUTE elsewhere.\n"
            "- Reasoning must be at least 30 words and tie proxy + momentum + portfolio context together.\n"
            "- Return ONLY the JSON object, no markdown, no explanation."
        )

    # ------------------------------------------------------------------
    # Allocator: validator
    # ------------------------------------------------------------------
    def _validate_proposed_actions(
        self,
        raw_actions: list,
        target_list: TargetResearchList,
        market_state: CurrentMarketState,
    ) -> List[ProposedAction]:
        """
        Coerce, sanitize, and cap the LLM-emitted actions.

        Rules enforced:
          1. Action ∈ {EXPAND, DILUTE, ADD} (case-insensitive). Invalid → "HOLD" dropped.
          2. Time_Horizon ∈ {SHORT_TERM_MOMENTUM, LONG_TERM_HOLD}. Invalid → LONG_TERM_HOLD.
          3. Ticker must be present in EITHER the research list OR current portfolio. Otherwise drop.
          4. Cap list at PORTFOLIO_MAX_PROPOSED_ACTIONS.
          5. Deduplicate (same ticker+action) — keep first occurrence.
        """
        # Build the "universe" of acceptable tickers
        research_set = target_list.ticker_set()
        portfolio_set: set = set()
        alloc = market_state.get("portfolio_allocations", {})
        for sdata in alloc.get("sectors", {}).values():
            for h in sdata.get("holdings", []):
                portfolio_set.add(str(h.get("ticker", "")).upper().strip())
        for h in alloc.get("cash_holdings", []):
            portfolio_set.add(str(h.get("ticker", "")).upper().strip())
        universe = research_set | portfolio_set

        valid: List[ProposedAction] = []
        seen_pairs: set = set()
        for raw in raw_actions:
            if not isinstance(raw, dict):
                continue
            ticker = str(raw.get("Ticker", "")).upper().strip()
            if not ticker or ticker not in universe:
                # Drop the action — ticker unknown to the system
                continue

            action = str(raw.get("Action", "")).upper().strip()
            if action not in PORTFOLIO_VALID_ACTIONS:
                # Coerce invalid action to a safe "HOLD" — which we don't ship, so drop
                continue

            horizon = str(raw.get("Time_Horizon", "")).upper().strip()
            if horizon not in PORTFOLIO_VALID_HORIZONS:
                horizon = "LONG_TERM_HOLD"

            # Deduplicate (ticker, action) tuples
            key = (ticker, action)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            reasoning = str(raw.get("Reasoning", "")).strip()
            if len(reasoning) < 30:
                reasoning = (
                    reasoning + " (Reasoning auto-extended; original too brief.)"
                ) if reasoning else (
                    "Heuristic extension: this action was generated against the "
                    "incoming regime signal and the researcher's Target_Research_List."
                )

            valid.append(
                ProposedAction(
                    ticker=ticker,
                    action=action,
                    time_horizon=horizon,
                    reasoning=reasoning,
                )
            )

            if len(valid) >= PORTFOLIO_MAX_PROPOSED_ACTIONS:
                break

        return valid

    # ==================================================================
    # Shared fallbacks
    # ==================================================================
    def _fallback_recommendation(
        self,
        regime_analysis: RegimeAnalysis,
        reason: str = "",
    ) -> PortfolioRecommendation:
        """
        Conservative recommendation when LLM calls fail or the universe of
        valid actions is empty. Issues no trades and stops the graph.
        """
        return PortfolioRecommendation(
            Portfolio_Impact_Assessment=(
                f"Fallback (no-trade) recommendation issued. Reason: {reason or 'unknown'}. "
                "Conservative posture: do not commit capital until the portfolio manager "
                "LLM chain is available to validate proxy discovery against the current portfolio."
            ),
            Abstract_Proxy_Discoveries=(
                "No proxy discovery performed — LLM chain unavailable."
            ),
            Momentum_vs_Valuation_Analysis=(
                "No momentum/valuation analysis performed — LLM chain unavailable."
            ),
            Proposed_Actions=[],
            proceed_to_risk_reviewer=False,
            regime_significance_score=regime_analysis.Significance_Score if regime_analysis else 0,
            research_summary="Heuristic fallback — no LLM results.",
            queries_used=[],
        )


# =============================================================================
# Pretty-printing
# =============================================================================

def print_recommendation(rec: PortfolioRecommendation) -> None:
    """Print a formatted summary of the portfolio recommendation."""
    print(f"\n    ╔══════════════════════════════════════════════════════════════╗")
    print(f"    ║  PORTFOLIO MANAGER VERDICT (Agent 3)                        ║")
    print(f"    ╠══════════════════════════════════════════════════════════════╣")
    print(f"    ║  Regime significance: {rec.regime_significance_score:3d}/100                             ║")
    print(f"    ║  Proxies discovered:   {len([t for t in rec.queries_used]):3d} queries executed                 ║")
    print(f"    ║  Proposed actions:     {len(rec.Proposed_Actions):3d}                                      ║")
    verdict = (
        "→ Route to Agent 4 (Risk Reviewer)" if rec.proceed_to_risk_reviewer
        else "→ No-trade signal; graph ends."
    )
    print(f"    ║  {verdict}              ║")
    print(f"    ╠══════════════════════════════════════════════════════════════╣")
    print(f"    ║  PORTFOLIO IMPACT:                                            ║")
    for line in _wrap(rec.Portfolio_Impact_Assessment, 60):
        print(f"    ║    {line}   ║")
    print(f"    ║  PROXY DISCOVERIES:                                           ║")
    for line in _wrap(rec.Abstract_Proxy_Discoveries, 60):
        print(f"    ║    {line}   ║")
    print(f"    ║  MOMENTUM vs VALUATION:                                       ║")
    for line in _wrap(rec.Momentum_vs_Valuation_Analysis, 60):
        print(f"    ║    {line}   ║")
    print(f"    ╠══════════════════════════════════════════════════════════════╣")
    if rec.Proposed_Actions:
        for i, a in enumerate(rec.Proposed_Actions, 1):
            print(f"    ║  ACTION [{i}] {a.ticker:<6} {a.action:<7} {a.time_horizon:<20}     ║")
            for line in _wrap(a.reasoning, 60):
                print(f"    ║    {line}   ║")
    else:
        print(f"    ║  (no actions — hold posture)                                 ║")
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
