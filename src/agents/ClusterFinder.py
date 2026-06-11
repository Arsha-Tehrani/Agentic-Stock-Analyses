"""
ClusterFinder.py – Related-article clustering for high emotional-disparity articles.

When the ToneAnalystNode identifies an article with high emotional disparity
(the emotional framing dramatically exceeds the factual substance), the
ClusterFinder searches the existing database for articles covering similar
topics to determine if:

  - Multiple outlets show the same emotional reaction (consensus alarm/euphoria)
  - The reaction is isolated to a single source (potential outlier/sensationalism)

This additional context helps downstream agents decide whether the emotional
signal indicates a genuine market sentiment shift or just noise.

All tunable constants are in src/config.py.
"""

import asyncio
from typing import List, Optional

from src.db.DatabaseSink import DatabaseSink
from src.state import ScoutArticle
from src.agents.ToneAnalystNode import ToneAnalystNode
from src.config import (
    CLUSTER_DAYS_WINDOW,
    CLUSTER_MAX_RELATED,
    CLUSTER_DISPARITY_THRESHOLD,
)


class ClusterFinder:
    """
    Searches for related articles to provide context around high-disparity pieces.

    Strategy:
      1. Extract meaningful search keywords from the article's title, ticker_tags,
         and emotional analysis (key factual claims preferred).
      2. Query the database for articles matching those keywords in a ±N day window.
      3. Run tonality analysis on the found related articles so we can compare
         emotional scores across the cluster.
    """

    def __init__(self, db_sink: DatabaseSink, tone_analyst: Optional[ToneAnalystNode] = None):
        self._db = db_sink
        self._tone_analyst = tone_analyst or ToneAnalystNode()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def find_clusters(
        self,
        articles: List[ScoutArticle],
        disparity_threshold: Optional[float] = None,
    ) -> List[ScoutArticle]:
        """
        For every article whose disparity_score exceeds the threshold,
        search the database for related articles and attach them.

        Returns the same list (mutated in-place).
        """
        threshold = disparity_threshold or CLUSTER_DISPARITY_THRESHOLD
        high_disparity = [
            a for a in articles
            if a.emotional_analysis and a.emotional_analysis.disparity_score >= threshold
        ]

        if not high_disparity:
            print("\n  ✅ No articles exceed the emotional disparity threshold — skipping cluster search.")
            return articles

        print(f"\n  🔗 Cluster search: {len(high_disparity)} article(s) have high emotional disparity — finding related coverage...")

        for i, article in enumerate(high_disparity):
            print(f"\n    🔍 [{i+1}/{len(high_disparity)}] {article.title[:70]}...")
            try:
                related = self._find_related(article)
                if related:
                    # Run tonality on the related articles so we can compare
                    await self._tone_analyst.analyze_batch(related)
                    article.related_articles = related
                    print(f"      📰 Found {len(related)} related article(s) — emotional comparison ready")
                else:
                    print(f"      ⚠️  No related articles found in database (within ±{CLUSTER_DAYS_WINDOW} days)")
            except Exception as e:
                print(f"      ❌ Cluster search failed: {e}")

        return articles

    # ------------------------------------------------------------------
    # Related article discovery
    # ------------------------------------------------------------------
    def _find_related(self, article: ScoutArticle) -> List[ScoutArticle]:
        keywords = self._extract_search_keywords(article)
        if not keywords:
            return []

        raw_articles = self._db.find_related_by_keywords(
            keywords=keywords,
            exclude_url=article.url,
            days_window=CLUSTER_DAYS_WINDOW,
            limit=CLUSTER_MAX_RELATED,
        )

        # Convert NewsArticle -> ScoutArticle (lightweight, no Scout enrichment needed)
        related: List[ScoutArticle] = []
        for raw in raw_articles:
            related.append(ScoutArticle(
                source_bucket=raw.source_bucket,
                source_name=raw.source_name,
                title=raw.title,
                summary=raw.summary or (raw.content[:300] if raw.content else ""),
                url=raw.url,
                timestamp=raw.timestamp,
                ticker_tags=raw.ticker_tags,
                aggregated_content=raw.content,  # Use DB content as context for tone analysis
                importance_score=raw.importance_score or 0.0,
            ))

        return related

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_search_keywords(article: ScoutArticle) -> List[str]:
        """
        Extract high-signal keywords from the article for database search.

        Priority order:
          1. Factual claims from emotional analysis (most specific)
          2. Ticker tags
          3. Key nouns from the title
        """
        keywords: List[str] = []

        # 1. Factual claims — extract key entities
        if article.emotional_analysis and article.emotional_analysis.key_factual_claims:
            for claim in article.emotional_analysis.key_factual_claims:
                # Extract meaningful words (4+ chars) from each claim
                words = claim.split()
                for w in words:
                    clean = w.strip(".,;:'\"()[]{}!?").lower()
                    if len(clean) >= 4 and clean not in {"with", "from", "that", "this", "have", "been", "were", "they"}:
                        if clean not in keywords:
                            keywords.append(clean)

        # 2. Ticker tags
        for ticker in (article.ticker_tags or []):
            if ticker not in keywords:
                keywords.append(ticker)

        # 3. Title nouns (fallback)
        title_words = article.title.lower().split()
        important_title_words = [
            w.strip(".,;:'\"()[]{}!?").lower() for w in title_words
            if len(w.strip(".,;:'\"()[]{}!?")) >= 4
            and w.strip(".,;:'\"()[]{}!?").lower() not in {
                "with", "from", "that", "this", "have", "been", "were", "they",
                "what", "when", "where", "which", "there", "their", "about",
                "over", "after", "before", "could", "would", "should",
            }
        ]
        for w in important_title_words:
            if w not in keywords:
                keywords.append(w)

        # Cap at 15 keywords to avoid overly broad searches
        return keywords[:15]