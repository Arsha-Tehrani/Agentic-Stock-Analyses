"""
run_pipeline.py – Multi-bucket news ingestion pipeline with Scout enrichment,
emotional tonality analysis, and related-article clustering.

Usage:
    python3 run_pipeline.py
"""

import asyncio
from typing import List

from src.NewsArticle import NewsArticle
from src.ingestors.WireIngestor import WireIngestor
from src.ingestors.MacroBlogs import MacroBlogIngestor
from src.ingestors.GlobalOutlets import RegionalRSSIngestor
from src.db.DatabaseSink import DatabaseSink
from src.agents.ScoutNode import ScoutNode, ScoutArticle
from src.agents.ToneAnalystNode import ToneAnalystNode
from src.agents.ClusterFinder import ClusterFinder
from src.agents.RegimeAnalystNode import RegimeAnalystNode
from src.agents.PortfolioManagerNode import (
    PortfolioManagerNode,
    route_after_portfolio_manager,
)
from src.state import GraphState
from src.config import (
    WIRE_API_KEY,
    BLOG_TARGETS,
    REGIONAL_FEEDS,
    IMPORTANCE_HIGH_THRESHOLD,
    DISPARITY_HIGH_THRESHOLD,
    REGIME_DEFAULT_MARKET_STATE,
)
from src.state import PortfolioState


async def main_pipeline() -> List[ScoutArticle]:
    wire_client = WireIngestor(api_key=WIRE_API_KEY)
    blog_client = MacroBlogIngestor(target_urls=BLOG_TARGETS)
    regional_client = RegionalRSSIngestor(rss_feeds=REGIONAL_FEEDS)

    print("=" * 60)
    print("🚀 Starting multi-bucket news ingestion pipeline...")
    print("=" * 60)

    wire_task = wire_client.fetch_latest_wires()
    blog_task = blog_client.fetch_blog_posts()
    regional_task = regional_client.fetch_all_regional()
    raw_results = await asyncio.gather(wire_task, blog_task, regional_task)
    bucket_names = ["Wires", "Macro_Blogs", "Regional"]

    all_articles: List[NewsArticle] = []
    seen_urls: set = set()

    for bucket_idx, bucket_result in enumerate(raw_results):
        print(f"\n  📡 Processing {bucket_names[bucket_idx]} ({len(bucket_result)} raw items) ...")
        for item in bucket_result:
            if item.get("url") and item["url"] in seen_urls:
                continue
            try:
                validated_article = NewsArticle(**item)
                all_articles.append(validated_article)
                if item.get("url"):
                    seen_urls.add(item["url"])
            except Exception as e:
                print(f"    ⚠️  Validation error: {e}")

    print(f"\n✅ Validation complete. {len(all_articles)} unique articles ready.")

    print("\n" + "=" * 60)
    print("🔍 Scout enrichment phase — evaluating importance & gathering context...")
    print("=" * 60)

    scout = ScoutNode()
    enriched_articles: List[ScoutArticle] = scout.enrich_batch(all_articles)

    print("\n" + "=" * 60)
    print("🎭 Emotional tonality analysis — separating emotion from facts...")
    print("=" * 60)

    tone_analyst = ToneAnalystNode()
    enriched_articles = tone_analyst.analyze_batch(enriched_articles)

    print("\n" + "=" * 60)
    print("🔗 Cluster search — finding related articles for high-disparity pieces...")
    print("=" * 60)

    db_sink = DatabaseSink()
    cluster_finder = ClusterFinder(db_sink=db_sink, tone_analyst=tone_analyst)
    enriched_articles = cluster_finder.find_clusters(enriched_articles)

    # ── Build NewsArticle list with emotional analysis data for DB persistence ──
    db_articles: List[NewsArticle] = []
    for sa in enriched_articles:
        article = NewsArticle(
            source_bucket=sa.source_bucket,
            source_name=sa.source_name,
            title=sa.title,
            content=sa.aggregated_content if sa.aggregated_content else sa.summary,
            summary=sa.summary,
            url=sa.url,
            timestamp=sa.timestamp,
            ticker_tags=sa.ticker_tags,
            importance_score=sa.importance_score,
        )
        if sa.emotional_analysis:
            article.emotional_score = sa.emotional_analysis.emotional_score
            article.factual_score = sa.emotional_analysis.factual_score
            article.disparity_score = sa.emotional_analysis.disparity_score
            article.tonality_label = sa.emotional_analysis.tonality_label
            article.emotional_reasoning = sa.emotional_analysis.reasoning
            article.emotional_phrases = sa.emotional_analysis.key_emotional_phrases
            article.factual_claims = sa.emotional_analysis.key_factual_claims
        db_articles.append(article)

    print("\n" + "=" * 60)
    print("💾 Persisting enriched articles to database...")
    print("=" * 60)

    rows_inserted = db_sink.insert_articles(db_articles)

    # ── Load portfolio state from database (or initialize from config) ──
    print("\n" + "=" * 60)
    print("📊 Loading portfolio state from database...")
    print("=" * 60)

    db_sink = DatabaseSink()
    db_sink.initialize_portfolio_state(REGIME_DEFAULT_MARKET_STATE)
    portfolio_state_dict = db_sink.load_portfolio_state()
    if portfolio_state_dict:
        portfolio_state = PortfolioState.from_dict(portfolio_state_dict)
        print(f"  📦 Loaded portfolio state (v{portfolio_state.version}) from {portfolio_state.timestamp}")
        print(f"     Updated by: {portfolio_state.updated_by}")
        print(f"     Macro regime: {portfolio_state.macro_baseline.market_regime}")
    else:
        portfolio_state = PortfolioState.from_dict(REGIME_DEFAULT_MARKET_STATE)
        print("  ⚠️  No portfolio state in DB, using config defaults")

    # ── Regime Analyst ──
    print("\n" + "=" * 60)
    print("🏛️  Regime Analyst — detecting macro shifts & capital rotation...")
    print("=" * 60)

    regime_analyst = RegimeAnalystNode()
    state: GraphState = {
        "articles": enriched_articles,
        "market_state": portfolio_state.to_market_state(),
        "regime_analysis": None,
        "proceed_to_portfolio_manager": False,
    }
    state = regime_analyst.run_regime_analyst_node(state)
    significant_count = 0

    portfolio_recommendation = None
    if state["proceed_to_portfolio_manager"]:
        print("\n  🚨 REGIME CHANGE DETECTED! → routing to Portfolio Manager (Agent 3)")
        print("  📋 Saving significant articles for later analysis...")
        significant_count = db_sink.insert_significant_articles(
            articles=db_articles,
            signed_articles=enriched_articles,
            regime_analysis=state["regime_analysis"],
        )

        # ── Agent 3: Portfolio Manager (Researcher + Book Runner) ──
        print("\n" + "=" * 60)
        print("💼 Portfolio Manager (Agent 3) — Researcher + Book Runner")
        print("=" * 60)
        portfolio_manager = PortfolioManagerNode()
        state = portfolio_manager.run_portfolio_manager_node(state)
        portfolio_recommendation = state.get("portfolio_recommendation")

        # Conditional routing for the LangGraph edge (documented even though
        # Agent 4 doesn't exist yet — the routing function is the canonical
        # graph-wiring contract).
        next_hop = route_after_portfolio_manager(state)
        if next_hop == "risk_reviewer":
            print("\n  → Next hop would be Agent 4 (Risk Reviewer) — not yet implemented.")
            print("     The PortfolioRecommendation above is the final output of this run.")
        else:
            print("\n  🟡 No-trade signal from Portfolio Manager — graph ends here.")
    else:
        print("\n  ✅ No regime change — graph execution ends here.")

    total_in_db = db_sink.article_count()
    high_importance = sum(1 for a in db_articles if a.importance_score and a.importance_score >= IMPORTANCE_HIGH_THRESHOLD)
    high_disparity = sum(1 for a in enriched_articles if a.emotional_analysis and a.emotional_analysis.disparity_score >= DISPARITY_HIGH_THRESHOLD)
    clustered = sum(1 for a in enriched_articles if a.related_articles)

    print(f"\n📊 Database summary: {total_in_db} articles total in data/news.db")
    print(f"   High-importance (≥{IMPORTANCE_HIGH_THRESHOLD}):      {high_importance}")
    print(f"   High emotional disparity:    {high_disparity}")
    print(f"   Articles with clusters:      {clustered}")
    print(f"   Significant articles saved:  {significant_count}")
    print(f"   Scout-enriched:              {len(enriched_articles)}")
    print("=" * 60)
    print("Pipeline finished. Regime-analyzed articles ready for review.")
    print("=" * 60)

    return enriched_articles


if __name__ == "__main__":
    asyncio.run(main_pipeline())