"""
debug_portfolio_manager.py — Isolated Portfolio Manager (Agent 3) debugger.

Loads saved articles from data/news.db, constructs a realistic GraphState
including RegimeAnalysis, and runs ONLY the Portfolio Manager chains:
  1. Quant Researcher (query generation → DDG → synthesis)
  2. Book Runner (allocation → trade plan)

Usage:
    python3 debug_portfolio_manager.py                        # Uses top 3 recent articles
    python3 debug_portfolio_manager.py --count 5               # Uses 5 articles
    python3 debug_portfolio_manager.py --significance 85       # Override S threshold
    python3 debug_portfolio_manager.py --skip-ddg              # Skip live DDG, use heuristic
    python3 debug_portfolio_manager.py --verbose               # Show every JSON parse attempt
"""

import argparse
import json
import sys
import time
from datetime import datetime
from typing import List, Optional

from src.db.DatabaseSink import DatabaseSink
from src.state import (
    ScoutArticle,
    EmotionalAnalysis,
    RegimeAnalysis,
    GraphState,
    PortfolioState,
)
from src.agents.PortfolioManagerNode import (
    PortfolioManagerNode,
    print_recommendation,
)
from src.config import (
    REGIME_DEFAULT_MARKET_STATE,
    REGIME_WEIGHT_MACRO,
    REGIME_WEIGHT_ROTATION,
    REGIME_WEIGHT_EMOTIONAL,
    REGIME_SIGNIFICANCE_THRESHOLD,
)

DIVIDER = "=" * 60


def build_scout_article_from_db_row(row: dict) -> ScoutArticle:
    """
    Convert a raw DB row (from fetch_recent or find_articles_with_emotional_analysis)
    into a ScoutArticle suitable for the Portfolio Manager.
    """
    def _parse_json_field(val):
        if val is None:
            return None
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return None
        return val

    emotional_analysis = None
    if row.get("emotional_score") is not None:
        emotional_analysis = EmotionalAnalysis(
            emotional_score=row["emotional_score"],
            factual_score=row["factual_score"] if row.get("factual_score") is not None else 0.0,
            disparity_score=row["disparity_score"] if row.get("disparity_score") is not None else 0.0,
            tonality_label=row.get("tonality_label", "unknown") or "unknown",
            reasoning=row.get("emotional_reasoning", "") or "",
            key_emotional_phrases=(
                _parse_json_field(row.get("emotional_phrases"))
                if row.get("emotional_phrases") else []
            ) or [],
            key_factual_claims=(
                _parse_json_field(row.get("factual_claims"))
                if row.get("factual_claims") else []
            ) or [],
        )

    return ScoutArticle(
        source_bucket=row.get("source_bucket", "DB_Debug"),
        source_name=row.get("source_name", "Database"),
        title=row.get("title", "No Title"),
        summary=row.get("summary", row.get("content", "")[:300]) or "",
        url=row.get("url"),
        timestamp=(
            datetime.fromisoformat(row["timestamp"])
            if isinstance(row.get("timestamp"), str)
            else datetime.now()
        ),
        ticker_tags=(
            json.loads(row["ticker_tags"])
            if isinstance(row.get("ticker_tags"), str)
            else (row.get("ticker_tags") or [])
        ),
        importance_score=row.get("importance_score", 0.0) or 0.0,
        importance_reasoning="Loaded from DB debug run",
        aggregated_content=row.get("content", "") or "",
        emotional_analysis=emotional_analysis,
    )


def build_regime_analysis_from_db(row: dict, significance_override: Optional[int] = None) -> RegimeAnalysis:
    """
    Build a RegimeAnalysis from a significant_articles DB row, or create a
    synthetic one with high scores to trigger the Portfolio Manager.
    """
    if row and row.get("macro_score") is not None:
        macro = int(row["macro_score"]) if row["macro_score"] else 7
        rotation = int(row["rotation_score"]) if row["rotation_score"] else 7
        emotional = int(row["emotional_arbitrage_score"]) if row["emotional_arbitrage_score"] else 7
        sig = significance_override or int(row.get("significance_score", 0) or 70)
    else:
        # Synthetic scores that will trigger the Portfolio Manager
        macro = 8
        rotation = 7
        emotional = 6
        sig = significance_override or 75

    # Recompute S if override was given
    if significance_override is not None:
        sig = significance_override

    return RegimeAnalysis(
        Macro_Analysis=(
            row.get("macro_analysis", "") if row else
            "Synthetic debug: Inflation cooling, rate hold expected, "
            "GDP growth moderating. Market pricing in soft landing."
        ),
        Rotation_Analysis=(
            row.get("rotation_analysis", "") if row else
            "Synthetic debug: Capital rotating from defensive to cyclical. "
            "Tech sector seeing increased inflows. Semis leading."
        ),
        Emotional_Arbitrage_Analysis=(
            row.get("emotional_arbitrage_analysis", "") if row else
            "Synthetic debug: Bearish sentiment overdone relative to fundamentals. "
            "Disparity scores suggest opportunity in quality names."
        ),
        macro_score=macro,
        rotation_score=rotation,
        emotional_arbitrage_score=emotional,
        Significance_Score=sig,
        proceed_to_portfolio_manager=sig > REGIME_SIGNIFICANCE_THRESHOLD,
    )


def load_articles(db_sink: DatabaseSink, count: int, skip_ddg: bool) -> List[ScoutArticle]:
    """
    Load articles from the database. Prefers articles with emotional analysis data.
    Falls back to recent articles if none have emotional data.
    """
    # Try to get articles with emotional analysis first
    emotional_rows = db_sink.find_articles_with_emotional_analysis(limit=count)
    if emotional_rows:
        print(f"  📰 Loaded {len(emotional_rows)} article(s) with emotional analysis from DB")
        return [build_scout_article_from_db_row(r) for r in emotional_rows]

    # Fall back to recent articles
    recent = db_sink.fetch_recent(limit=count)
    if recent:
        print(f"  📰 Loaded {len(recent)} recent article(s) from DB (no emotional data)")
        rows = [
            {
                "source_bucket": a.source_bucket,
                "source_name": a.source_name,
                "title": a.title,
                "summary": a.summary,
                "url": a.url,
                "timestamp": a.timestamp.isoformat() if isinstance(a.timestamp, datetime) else str(a.timestamp),
                "ticker_tags": json.dumps(a.ticker_tags) if a.ticker_tags else "[]",
                "importance_score": a.importance_score,
                "content": "",
                "emotional_score": None,
                "factual_score": None,
                "disparity_score": None,
                "tonality_label": None,
                "emotional_reasoning": None,
                "emotional_phrases": None,
                "factual_claims": None,
            }
            for a in recent
        ]
        return [build_scout_article_from_db_row(r) for r in rows]

    print("  ❌ No articles found in database. Run the pipeline first.")
    sys.exit(1)


def load_significant_articles(db_sink: DatabaseSink) -> List[dict]:
    """Load significant articles for RegimeAnalysis data."""
    return db_sink.fetch_significant_articles(limit=10)


def main():
    parser = argparse.ArgumentParser(
        description="Debug Portfolio Manager (Agent 3) in isolation using saved DB data."
    )
    parser.add_argument(
        "--count", type=int, default=3,
        help="Number of articles to load from DB (default: 3)"
    )
    parser.add_argument(
        "--significance", type=int, default=None,
        help="Override Significance_Score (default: from DB or 75)"
    )
    parser.add_argument(
        "--skip-ddg", action="store_true",
        help="Skip live DuckDuckGo queries, use heuristic researcher only"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show verbose JSON parse attempt details"
    )
    args = parser.parse_args()

    print(DIVIDER)
    print("🐞 DEBUG PORTFOLIO MANAGER (Agent 3) — Isolated Debug Run")
    print(f"   Article count:              {args.count}")
    print(f"   Significance override:      {args.significance or 'from DB'}")
    print(f"   Skip DDG (heuristic only):  {args.skip_ddg}")
    print(f"   Verbose mode:               {args.verbose}")
    print(DIVIDER)

    t0 = time.monotonic()

    # ── Step 1: Connect to DB and load data ──
    print("\n📡 Phase 1: Loading data from database...")
    db_sink = DatabaseSink()
    total_articles = db_sink.article_count()
    print(f"   Total articles in DB: {total_articles}")

    articles = load_articles(db_sink, args.count, args.skip_ddg)
    sig_rows = load_significant_articles(db_sink)

    if sig_rows:
        print(f"   Significant articles found: {len(sig_rows)}")
    else:
        print("   ⚠️  No significant articles in DB — using synthetic regime analysis")

    # ── Step 2: Print the loaded articles ──
    print(f"\n📋 Loaded {len(articles)} article(s) for Portfolio Manager:")
    for i, a in enumerate(articles, 1):
        tone_info = ""
        if a.emotional_analysis:
            tone_info = (
                f" | tonality={a.emotional_analysis.tonality_label} "
                f"disp={a.emotional_analysis.disparity_score:.2f}"
            )
        print(f"  [{i}] {a.title[:80]} | {a.source_name}{tone_info}")
        if a.ticker_tags:
            print(f"       Tickers: {', '.join(a.ticker_tags)}")

    # ── Step 3: Build GraphState ──
    print("\n" + DIVIDER)
    print("🏛️  Phase 2: Building GraphState with RegimeAnalysis")
    print(DIVIDER)

    # Use the first significant article's regime data, or build synthetic
    regime_analysis = build_regime_analysis_from_db(
        sig_rows[0] if sig_rows else None,
        significance_override=args.significance,
    )

    print(f"   Regime significance:  {regime_analysis.Significance_Score}/100")
    print(f"   M={regime_analysis.macro_score}/10  "
          f"R={regime_analysis.rotation_score}/10  "
          f"E={regime_analysis.emotional_arbitrage_score}/10")
    print(f"   Proceed to PM:        {regime_analysis.proceed_to_portfolio_manager}")

    if not regime_analysis.proceed_to_portfolio_manager:
        print("\n⚠️  Significance score too low — Portfolio Manager would not be triggered.")
        print("   Use --significance N to override (e.g., --significance 85)")
        print("   Exiting.")
        sys.exit(0)

    # Build portfolio state from config
    portfolio_state = PortfolioState.from_dict(REGIME_DEFAULT_MARKET_STATE)
    market_state = portfolio_state.to_market_state()

    state: GraphState = {
        "articles": articles,
        "market_state": market_state,
        "regime_analysis": regime_analysis,
        "proceed_to_portfolio_manager": True,
    }

    # ── Step 4: Run Portfolio Manager ──
    print("\n" + DIVIDER)
    print("💼 Phase 3: Running Portfolio Manager (Researcher → Book Runner)")
    print(DIVIDER)

    pm = PortfolioManagerNode()

    if args.skip_ddg:
        # Monkey-patch to skip DDG if desired
        original_run = pm._run_researcher_chain
        def skip_ddg_researcher(articles, regime_analysis, market_state):
            print("\n    ⚠️  --skip-ddg mode: using heuristic researcher (no LLM queries)")
            return pm._heuristic_researcher(articles, market_state)
        pm._run_researcher_chain = skip_ddg_researcher

    print("\n   Starting Portfolio Manager chain...")
    try:
        t_pm_start = time.monotonic()
        state = pm.run_portfolio_manager_node(state)
        t_pm = time.monotonic() - t_pm_start

        recommendation = state.get("portfolio_recommendation")
        if recommendation:
            print(f"\n📋 Portfolio Manager completed in {t_pm:.1f}s")
            print(f"   Actions proposed: {len(recommendation.Proposed_Actions)}")
            print_recommendation(recommendation)
        else:
            print("\n❌ No recommendation produced — Portfolio Manager returned None")
    except Exception as e:
        print(f"\n❌ Portfolio Manager crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Step 5: Summary ──
    elapsed = time.monotonic() - t0
    print(f"\n{DIVIDER}")
    print("📋 DEBUG RUN SUMMARY")
    print(DIVIDER)
    print(f"   Articles fed to PM:      {len(articles)}")
    print(f"   Elapsed time:            {elapsed:.1f}s")
    rec = state.get("portfolio_recommendation")
    if rec:
        print(f"   Actions proposed:        {len(rec.Proposed_Actions)}")
        print(f"   Proceed to risk review:  {rec.proceed_to_risk_reviewer}")
        print(f"   DDG queries used:        {len(rec.queries_used)}")
        for i, q in enumerate(rec.queries_used, 1):
            print(f"      [{i}] {q}")
    else:
        print("   Recommendation:          NONE (fallback issued)")
    print(DIVIDER)


if __name__ == "__main__":
    main()