"""
debug_pipeline.py — Lightweight test runner for the full pipeline.
Processes only 3 articles with verbose debug output to verify:
   1. JSON truncation fix (REGIME_LLM_MAX_TOKENS=1500, TONALITY_MAX_TOKENS=2000)
  2. OR-gate individual trigger threshold
  3. Jittered staggered concurrency (no 503 bursts)

Usage:
    python3 debug_pipeline.py

Token cost: ~3/32 ≈ 9% of a full run. No database writes.
"""

import asyncio
import json
import time
from typing import List

from src.NewsArticle import NewsArticle
from src.ingestors.WireIngestor import WireIngestor
from src.ingestors.MacroBlogs import MacroBlogIngestor
from src.ingestors.GlobalOutlets import RegionalRSSIngestor
from src.db.DatabaseSink import DatabaseSink
from src.agents.ScoutNode import ScoutNode
from src.agents.ToneAnalystNode import ToneAnalystNode
from src.agents.ClusterFinder import ClusterFinder
from src.agents.RegimeAnalystNode import RegimeAnalystNode
from src.state import GraphState, ScoutArticle, EmotionalAnalysis
from src.config import (
    WIRE_API_KEY,
    BLOG_TARGETS,
    REGIONAL_FEEDS,
    REGIME_DEFAULT_MARKET_STATE,
    REGIME_SIGNIFICANCE_THRESHOLD,
    REGIME_INDIVIDUAL_TRIGGER_THRESHOLD,
    SCOUT_CONCURRENCY_LIMIT,
    TONE_CONCURRENCY_LIMIT,
    TONALITY_MAX_TOKENS,
    REGIME_LLM_MAX_TOKENS,
)

# ── Debug config ──────────────────────────────────────────────────────────
DEBUG_ARTICLE_COUNT = 5   # How many articles to run through the pipeline
# Articles are persisted to the database (with dedup by URL) so that
# saved articles can be re-used by debug_portfolio_manager.py later.
# Set SKIP_DB=True to skip database writes during quick iteration.
SKIP_DB = False           # Set True to skip database writes entirely
# ───────────────────────────────────────────────────────────────────────────

DIVIDER = "=" * 60


async def main():
    print(DIVIDER)
    print("🐞 DEBUG PIPELINE — 3-article test run")
    print(f"   Scout concurrency limit:  {SCOUT_CONCURRENCY_LIMIT}")
    print(f"   Tone  concurrency limit:  {TONE_CONCURRENCY_LIMIT}")
    print(f"   Tonality max tokens:      {TONALITY_MAX_TOKENS}")
    print(f"   Regime  max tokens:       {REGIME_LLM_MAX_TOKENS}")
    print(f"   Composite threshold:      >{REGIME_SIGNIFICANCE_THRESHOLD}")
    print(f"   Individual trigger:       ≥{REGIME_INDIVIDUAL_TRIGGER_THRESHOLD}")
    print(DIVIDER)

    t0 = time.monotonic()

    # ── 1. Fetch all articles ──────────────────────────────────────────
    print("\n📡 Phase 1: Fetching articles from all sources...")
    wire_client = WireIngestor(api_key=WIRE_API_KEY)
    blog_client = MacroBlogIngestor(target_urls=BLOG_TARGETS)
    regional_client = RegionalRSSIngestor(rss_feeds=REGIONAL_FEEDS)

    wire_task = wire_client.fetch_latest_wires()
    blog_task = blog_client.fetch_blog_posts()
    regional_task = regional_client.fetch_all_regional()
    raw_results = await asyncio.gather(wire_task, blog_task, regional_task)
    bucket_names = ["Wires", "Macro_Blogs", "Regional"]

    all_articles: List[NewsArticle] = []
    seen_urls: set = set()

    for bucket_idx, bucket_result in enumerate(raw_results):
        print(f"  {bucket_names[bucket_idx]}: {len(bucket_result)} raw items")
        for item in bucket_result:
            if item.get("url") and item["url"] in seen_urls:
                continue
            try:
                validated = NewsArticle(**item)
                all_articles.append(validated)
                if item.get("url"):
                    seen_urls.add(item["url"])
            except Exception as e:
                pass  # Skip validation errors silently in debug mode

    # ── 2. Slice to debug count ────────────────────────────────────────
    total_available = len(all_articles)
    all_articles = all_articles[:DEBUG_ARTICLE_COUNT]
    print(f"\n✅ Fetched {total_available} total, using first {len(all_articles)} for debug run.")

    if not all_articles:
        print("❌ No articles available — aborting debug run.")
        return

    for i, a in enumerate(all_articles):
        print(f"  [{i+1}] {a.title[:80]} | {a.source_name} ({a.source_bucket})")

    # ── Filter out already-enriched articles ───────────────────────────
    if not SKIP_DB:
        print("\n" + DIVIDER)
        print("🔍 Phase 2: Checking for already-enriched articles in database...")
        print(DIVIDER)
        db_sink_check = DatabaseSink()
        articles_to_process: List[NewsArticle] = []
        articles_from_db: List[ScoutArticle] = []
        for a in all_articles:
            if a.url and db_sink_check.is_article_enriched(a.url):
                db_row = db_sink_check.fetch_enriched_article(a.url)
                if db_row:
                    print(f"  ⏭️  Already enriched: {a.title[:60]}... (loading from DB)")
                    # Reconstruct ScoutArticle from the DB row
                    sa = ScoutArticle(
                        source_bucket=db_row.get("source_bucket", a.source_bucket),
                        source_name=db_row.get("source_name", a.source_name),
                        title=db_row.get("title", a.title),
                        summary=db_row.get("summary", a.summary) or "",
                        url=db_row.get("url", a.url),
                        timestamp=a.timestamp,
                        ticker_tags=json.loads(db_row["ticker_tags"]) if isinstance(db_row.get("ticker_tags"), str) else (db_row.get("ticker_tags") or a.ticker_tags),
                        importance_score=db_row.get("importance_score", 0.0) or 0.0,
                        importance_reasoning="Loaded from DB (previously enriched)",
                        aggregated_content=db_row.get("content", "") or "",
                        emotional_analysis=EmotionalAnalysis(
                            emotional_score=db_row["emotional_score"],
                            factual_score=db_row["factual_score"] if db_row.get("factual_score") is not None else 0.0,
                            disparity_score=db_row["disparity_score"] if db_row.get("disparity_score") is not None else 0.0,
                            tonality_label=db_row.get("tonality_label", "unknown") or "unknown",
                            reasoning=db_row.get("emotional_reasoning", "") or "",
                            key_emotional_phrases=(
                                json.loads(db_row["emotional_phrases"]) if isinstance(db_row.get("emotional_phrases"), str) else (db_row.get("emotional_phrases") or [])
                            ),
                            key_factual_claims=(
                                json.loads(db_row["factual_claims"]) if isinstance(db_row.get("factual_claims"), str) else (db_row.get("factual_claims") or [])
                            ),
                        ) if db_row.get("emotional_score") is not None else None,
                    )
                    articles_from_db.append(sa)
                else:
                    articles_to_process.append(a)
            else:
                articles_to_process.append(a)
        print(f"  → {len(articles_to_process)} new article(s) to process, {len(articles_from_db)} already enriched (skipped)")
    else:
        articles_to_process = list(all_articles)
        articles_from_db = []

    # ── 3. Scout enrichment (only for new articles) ────────────────────
    print("\n" + DIVIDER)
    print("🔍 Phase 3: Scout enrichment (importance scoring + DDG context)")
    print(DIVIDER)

    t_scout_start = time.monotonic()
    scout = ScoutNode()
    newly_enriched = await scout.enrich_batch(articles_to_process)
    # Merge: DB-loaded articles come first, then newly enriched
    enriched = articles_from_db + newly_enriched
    t_scout = time.monotonic() - t_scout_start

    scout_ok = sum(1 for a in enriched if a.importance_score >= 0.0)
    print(f"\n📊 Scout complete in {t_scout:.1f}s: {scout_ok}/{len(enriched)} articles scored "
          f"({len(articles_from_db)} from DB)")
    for a in enriched:
        print(f"  [{a.importance_score:.2f}] {a.importance_reasoning[:100]}")

    # ── 4. Tonality analysis ───────────────────────────────────────────
    print("\n" + DIVIDER)
    print("🎭 Phase 3: Emotional tonality analysis")
    print(DIVIDER)

    t_tone_start = time.monotonic()
    tone_analyst = ToneAnalystNode()
    enriched = await tone_analyst.analyze_batch(enriched)
    t_tone = time.monotonic() - t_tone_start

    tone_ok = sum(1 for a in enriched if a.emotional_analysis)
    high_disp = sum(
        1 for a in enriched
        if a.emotional_analysis and a.emotional_analysis.disparity_score >= 0.35
    )
    print(f"\n📊 Tone complete in {t_tone:.1f}s: {tone_ok}/{len(enriched)} analyzed, {high_disp} high-disparity")
    for a in enriched:
        if a.emotional_analysis:
            ea = a.emotional_analysis
            print(f"  [{ea.tonality_label:>12s}] e={ea.emotional_score:+.2f} f={ea.factual_score:.2f} disp={ea.disparity_score:.2f} | {ea.reasoning[:80]}")

    # ── DB Persistence (unless SKIP_DB is set) ─────────────────────────
    if not SKIP_DB:
        # 5a. Persist raw articles after Scout enrichment (dedup by URL)
        print("\n" + DIVIDER)
        print("💾 Phase 4a: Persisting articles to database...")
        print(DIVIDER)
        db_sink = DatabaseSink()
        # Convert ScoutArticle -> NewsArticle for the raw insert
        raw_for_db: List[NewsArticle] = []
        for sa in enriched:
            raw_for_db.append(NewsArticle(
                source_bucket=sa.source_bucket,
                source_name=sa.source_name,
                title=sa.title,
                content=sa.aggregated_content or sa.summary or "",
                summary=sa.summary or "",
                url=sa.url,
                timestamp=sa.timestamp,
                ticker_tags=sa.ticker_tags,
                importance_score=sa.importance_score,
            ))
        inserted = db_sink.insert_articles_batch(raw_for_db)
        print(f"  Debug DB: {inserted} new article(s) saved (duplicates skipped).")
    else:
        print("\n  ⏭️  SKIP_DB=True — skipping database writes.")

    # ── 5. Cluster search ──────────────────────────────────────────────
    print("\n" + DIVIDER)
    print("🔗 Phase 4b: Cluster search (related article discovery)")
    print(DIVIDER)

    if SKIP_DB:
        # Need a db_sink anyway for the cluster finder even in skip mode
        db_sink = DatabaseSink()
    cluster_finder = ClusterFinder(db_sink=db_sink, tone_analyst=tone_analyst)
    enriched = await cluster_finder.find_clusters(enriched)
    clustered = sum(1 for a in enriched if a.related_articles)
    print(f"\n📊 Cluster search complete: {clustered}/{len(enriched)} articles have related coverage")

    # ── Backfill enrichment data into DB (unless SKIP_DB) ──
    if not SKIP_DB:
        print("\n" + DIVIDER)
        print("💾 Phase 4c: Backfilling emotional enrichment data...")
        print(DIVIDER)
        enrichment_updated = db_sink.update_article_enrichment(enriched)
        print(f"  Debug DB: {enrichment_updated} article(s) enriched.")

    # ── 6. Regime Analyst ──────────────────────────────────────────────
    print("\n" + DIVIDER)
    print("🏛️  Phase 5: Regime Analyst (gatekeeping evaluation)")
    print(DIVIDER)

    regime_analyst = RegimeAnalystNode()
    state: GraphState = {
        "articles": enriched,
        "market_state": REGIME_DEFAULT_MARKET_STATE,
        "regime_analysis": None,
        "proceed_to_portfolio_manager": False,
    }
    state = regime_analyst.run_regime_analyst_node(state)

    # ── 7. Summary ─────────────────────────────────────────────────────
    elapsed = time.monotonic() - t0

    print("\n" + DIVIDER)
    print("📋 DEBUG RUN SUMMARY")
    print(DIVIDER)
    print(f"  Articles processed:       {len(enriched)}/{total_available}")
    print(f"  Scout phase:              {t_scout:.1f}s")
    print(f"  Tone phase:               {t_tone:.1f}s")
    print(f"  Total wall-clock:         {elapsed:.1f}s")
    print(f"  High-disparity articles:  {high_disp}")
    print(f"  Clustered articles:       {clustered}")

    ra = state.get("regime_analysis")
    if ra:
        print(f"  Composite S score:        {ra.Significance_Score}/100")
        print(f"  M={ra.macro_score}/10  R={ra.rotation_score}/10  E={ra.emotional_arbitrage_score}/10")
        composite_fired = ra.Significance_Score > REGIME_SIGNIFICANCE_THRESHOLD
        individual_fired = (
            ra.macro_score >= REGIME_INDIVIDUAL_TRIGGER_THRESHOLD
            or ra.rotation_score >= REGIME_INDIVIDUAL_TRIGGER_THRESHOLD
            or ra.emotional_arbitrage_score >= REGIME_INDIVIDUAL_TRIGGER_THRESHOLD
        )
        print(f"  Composite trigger:        {'✅ FIRED' if composite_fired else '❌ not met'} (S>{REGIME_SIGNIFICANCE_THRESHOLD})")
        print(f"  Individual trigger:       {'✅ FIRED' if individual_fired else '❌ not met'} (any ≥{REGIME_INDIVIDUAL_TRIGGER_THRESHOLD})")
        print(f"  Gate result:              {'🚨 PROCEED to Portfolio Manager' if ra.proceed_to_portfolio_manager else '✅ No regime change — graph ends'}")
    else:
        print("  ⚠️  Regime analysis: NONE (likely failed)")

    print(DIVIDER)
    print("🐞 Debug run complete. Check output above for truncation/trigger/503 issues.")
    print(DIVIDER)


if __name__ == "__main__":
    asyncio.run(main())