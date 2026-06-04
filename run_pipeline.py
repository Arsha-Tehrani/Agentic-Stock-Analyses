"""
run_pipeline.py – Multi-bucket news ingestion pipeline with Scout enrichment.

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


async def main_pipeline() -> List[ScoutArticle]:
    blog_targets = [
        "https://www.bespokepremium.com/interactive/blog/",
        "https://macrocompass.substack.com/"
    ]
    regional_feeds = {
        "Nikkei Asia": "https://services.nikkei.com/core/v1/rss/asia/news.xml",
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml"
    }

    wire_client = WireIngestor(api_key="d8g80l9r01qlgcuhr95gd8g80l9r01qlgcuhr960")
    blog_client = MacroBlogIngestor(target_urls=blog_targets)
    regional_client = RegionalRSSIngestor(rss_feeds=regional_feeds)

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

    db_articles: List[NewsArticle] = []
    for sa in enriched_articles:
        db_articles.append(NewsArticle(
            source_bucket=sa.source_bucket,
            source_name=sa.source_name,
            title=sa.title,
            content=sa.aggregated_content if sa.aggregated_content else sa.summary,
            summary=sa.summary,
            url=sa.url,
            timestamp=sa.timestamp,
            ticker_tags=sa.ticker_tags,
            importance_score=sa.importance_score,
        ))

    print("\n" + "=" * 60)
    print("💾 Persisting enriched articles to database...")
    print("=" * 60)

    db_sink = DatabaseSink()
    rows_inserted = db_sink.insert_articles(db_articles)

    total_in_db = db_sink.article_count()
    high_importance = sum(1 for a in db_articles if a.importance_score and a.importance_score >= 0.7)

    print(f"\n📊 Database summary: {total_in_db} articles total in data/news.db")
    print(f"   High-importance (≥0.7): {high_importance}")
    print(f"   Scout-enriched:         {len(enriched_articles)}")
    print("=" * 60)
    print("Pipeline finished. Scout-enriched articles ready for Regime Analyst.")
    print("=" * 60)

    return enriched_articles


if __name__ == "__main__":
    asyncio.run(main_pipeline())