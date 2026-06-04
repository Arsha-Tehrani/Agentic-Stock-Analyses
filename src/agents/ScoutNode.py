"""
ScoutNode.py – Event-centric enrichment engine.

For every raw article (from any bucket), the Scout node performs a three-stage
enrichment pipeline:

  1. evaluate_importance() — Gemini LLM scores the article's market importance (0-1)
  2. generate_search_query() — Gemini LLM crafts a highly specific search query
  3. aggregate_snippets() — DuckDuckGo returns top-5 snippets, concatenated

The enriched result is a ScoutArticle ready for the Regime Analyst.
"""

import os
import json
from typing import List, Optional

from ddgs import DDGS
from google import genai

from src.NewsArticle import NewsArticle
from src.state import ScoutArticle

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


class ScoutNode:
    """
    Enriches raw articles with importance scoring, contextual search queries,
    and aggregated web snippets.
    """

    _DDG_MAX_RESULTS = 5

    def __init__(self):
        self._client: Optional[genai.Client] = None
        if GEMINI_API_KEY:
            self._client = genai.Client(api_key=GEMINI_API_KEY)

    def enrich(self, article: NewsArticle) -> ScoutArticle:
        importance = self._evaluate_importance(article)
        print(f"    📊 Importance: {importance['score']:.2f} — {importance['reasoning']}")

        search_query = self._generate_search_query(article)
        print(f"    🔍 Query: {search_query}")

        aggregated = self._aggregate_snippets(search_query)
        print(f"    📰 Aggregated context: {len(aggregated)} chars")

        return ScoutArticle(
            source_bucket=article.source_bucket,
            source_name=article.source_name,
            title=article.title,
            summary=article.summary or (article.content[:300] if article.content else ""),
            url=article.url,
            timestamp=article.timestamp,
            ticker_tags=article.ticker_tags,
            importance_score=importance["score"],
            importance_reasoning=importance["reasoning"],
            search_query=search_query,
            aggregated_content=aggregated,
        )

    # ------------------------------------------------------------------
    # Stage 1 – Importance evaluation via Gemini
    # ------------------------------------------------------------------
    def _evaluate_importance(self, article: NewsArticle) -> dict:
        if not self._client:
            return self._heuristic_importance(article)

        prompt = (
            "You are a senior macro hedge fund analyst. Rate the market importance "
            "of this news article on a scale from 0.0 (completely irrelevant) to 1.0 "
            "(extremely market-moving). Consider impact on equities, bonds, currencies, "
            "and commodities.\n\n"
            "Respond ONLY with valid JSON in this exact format:\n"
            '{"score": 0.85, "reasoning": "Brief 1-sentence justification"}\n\n'
            f"Title: {article.title}\n"
            f"Source: {article.source_name} ({article.source_bucket})\n"
            f"Summary: {article.summary or (article.content[:500] if article.content else 'N/A')}\n"
        )
        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={"temperature": 0.1, "max_output_tokens": 100},
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            return {
                "score": max(0.0, min(1.0, float(result.get("score", 0.5)))),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            print(f"    ⚠️  Gemini importance evaluation failed: {e}")
            return self._heuristic_importance(article)

    def _heuristic_importance(self, article: NewsArticle) -> dict:
        text = (article.title + " " + (article.summary or article.content or "")).lower()
        high_impact = [
            "fed", "federal reserve", "interest rate", "cpi", "inflation",
            "recession", "gdp", "employment", "nonfarm", "payroll",
            "central bank", "policymaker", "tariff", "sanction",
            "earnings", "corporate profit", "acquisition", "merger",
            "stimulus", "jolts", "pmi", "manufacturing",
        ]
        medium_impact = [
            "housing", "consumer", "retail", "trade deficit", "treasury",
            "bond yield", "sp500", "nasdaq", "dow", "volatility",
            "currency", "dollar", "euro", "yen", "emerging market",
        ]
        score = 0.3
        for kw in high_impact:
            if kw in text:
                score = max(score, 0.7)
                break
        for kw in medium_impact:
            if kw in text:
                score = max(score, 0.5)
        if article.source_name.lower() in ("bloomberg", "reuters", "wsj", "financial times"):
            score = max(score, 0.5)
        return {"score": round(score, 2), "reasoning": "Heuristic fallback (no Gemini key)"}

    # ------------------------------------------------------------------
    # Stage 2 – Search-query generation via Gemini
    # ------------------------------------------------------------------
    def _generate_search_query(self, article: NewsArticle) -> str:
        if not self._client:
            return self._heuristic_query(article)

        prompt = (
            "You are a research analyst. Based on the article below, generate a single, "
            "highly specific Google-search-style query that would find the most relevant "
            "and authoritative articles, analysis, and context about this topic.\n\n"
            "Rules:\n"
            "- Use 5-8 keywords maximum.\n"
            "- Include key entity names (people, companies, agencies, tickers).\n"
            "- Prefer recent event framing.\n"
            "- Return ONLY the query string, no explanation, no quotes.\n\n"
            f"Title: {article.title}\n"
            f"Summary: {article.summary or (article.content[:400] if article.content else 'N/A')}\n"
        )
        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config={"temperature": 0.2, "max_output_tokens": 60},
            )
            query = response.text.strip().strip('"').strip("'")
            return query if query else self._heuristic_query(article)
        except Exception as e:
            print(f"    ⚠️  Gemini query generation failed: {e}")
            return self._heuristic_query(article)

    @staticmethod
    def _heuristic_query(article: NewsArticle) -> str:
        title = article.title
        for sep in [" - ", " | ", " — "]:
            if sep in title:
                title = title.split(sep)[0]
        return title[:150]

    # ------------------------------------------------------------------
    # Stage 3 – DuckDuckGo search & snippet aggregation
    # ------------------------------------------------------------------
    def _aggregate_snippets(self, query: str) -> str:
        snippets: List[str] = []
        try:
            with DDGS() as ddgs:
                for i, result in enumerate(ddgs.text(query, max_results=self._DDG_MAX_RESULTS)):
                    body = result.get("body", "").strip()
                    if body:
                        snippets.append(f"[{i+1}] {result.get('title', '')}\n{body}")
            if snippets:
                return "\n\n".join(snippets)
            if len(query.split()) > 3:
                shorter = " ".join(query.split()[:3])
                with DDGS() as ddgs:
                    for i, result in enumerate(ddgs.text(shorter, max_results=self._DDG_MAX_RESULTS)):
                        body = result.get("body", "").strip()
                        if body:
                            snippets.append(f"[{i+1}] {result.get('title', '')}\n{body}")
                if snippets:
                    return "\n\n".join(snippets)
        except Exception as e:
            print(f"    ⚠️  DuckDuckGo search failed: {e}")
        return ""

    # ------------------------------------------------------------------
    # Batch enrichment
    # ------------------------------------------------------------------
    def enrich_batch(self, articles: List[NewsArticle]) -> List[ScoutArticle]:
        results: List[ScoutArticle] = []
        for i, article in enumerate(articles):
            print(f"\n  🔍 Scout [{i+1}/{len(articles)}] {article.title[:70]}...")
            try:
                results.append(self.enrich(article))
            except Exception as e:
                print(f"    ❌ Scout enrichment failed: {e}")
                results.append(ScoutArticle(
                    source_bucket=article.source_bucket,
                    source_name=article.source_name,
                    title=article.title,
                    summary=article.summary or (article.content[:300] if article.content else ""),
                    url=article.url,
                    timestamp=article.timestamp,
                    ticker_tags=article.ticker_tags,
                ))
        return results