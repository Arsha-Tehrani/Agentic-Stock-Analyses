"""
ScoutNode.py – Event-centric enrichment engine.

For every raw article (from any bucket), the Scout node performs a three-stage
enrichment pipeline:

  1. evaluate_importance() — Gemini LLM scores the article's market importance (0-1)
  2. generate_search_query() — Gemini LLM crafts a highly specific search query
  3. aggregate_snippets() — DuckDuckGo returns top-5 snippets, concatenated

The enriched result is a ScoutArticle ready for the Regime Analyst.

Concurrency: enriches articles with asyncio.gather capped by asyncio.Semaphore(N)
to avoid overwhelming the Gemini API. Retries 429/503 with exponential backoff.

All tunable constants are in src/config.py.
"""

import asyncio
import json
import random
from typing import List, Optional

from ddgs import DDGS
from google import genai
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

from src.NewsArticle import NewsArticle
from src.state import ScoutArticle
from src.utils.json_repair import parse_json_with_repair
from src.config import (
    GEMINI_API_KEY,
    SCOUT_GEMINI_MODEL,
    SCOUT_DDG_MAX_RESULTS,
    SCOUT_SUMMARY_MAX_CHARS,
    SCOUT_IMPORTANCE_TEMPERATURE,
    SCOUT_IMPORTANCE_MAX_TOKENS,
    SCOUT_IMPORTANCE_PROMPT_CHARS,
    SCOUT_QUERY_TEMPERATURE,
    SCOUT_QUERY_MAX_TOKENS,
    SCOUT_QUERY_PROMPT_CHARS,
    SCOUT_CONCURRENCY_LIMIT,
    SCOUT_HIGH_IMPACT_KWS,
    SCOUT_MEDIUM_IMPACT_KWS,
    SCOUT_HIGH_IMPACT_SOURCES,
)



def _is_retryable_error(exception: Exception) -> bool:
    """Return True for HTTP 429 / 503 / 5xx errors that should be retried."""
    status = getattr(exception, "code", None)
    if status is not None:
        return status in (429, 503) or (isinstance(status, int) and 500 <= status < 600)
    msg = str(exception).lower()
    retryable_keywords = ["429", "503", "unavailable", "resource_exhausted", "rate", "quota"]
    return any(kw in msg for kw in retryable_keywords)


def _llm_retry_decorator(func):
    """Exponential-backoff retry for transient Gemini errors (429/503/5xx).
    
    IMPORTANT: Only retries on genuine API errors (429, 503, 5xx). JSON parse
    failures are NOT retried — they indicate malformed/truncated LLM output
    that will produce the same result on every attempt. Use json_repair instead.
    """
    return retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),  # tenacity filters via _is_retryable_error in _sync_call
        after=lambda retry_state: print(
            f"    🔄 Gemini retry attempt {retry_state.attempt_number}/5 "
            f"(waited {retry_state.outcome_timestamp - retry_state.start_time:.1f}s) "
            f"— error was: {retry_state.outcome.exception()}"
        ) if retry_state.outcome and retry_state.outcome.failed else None,
        before_sleep=lambda retry_state: print(
            f"    ⏳ Backing off {retry_state.next_action.sleep:.1f}s before retry..."
        ) if retry_state.next_action and retry_state.next_action.sleep else None,
    )(func)


class ScoutNode:
    """
    Enriches raw articles with importance scoring, contextual search queries,
    and aggregated web snippets.

    Uses async concurrency with a semaphore to limit simultaneous LLM calls.
    """

    def __init__(self):
        self._client: Optional[genai.Client] = None
        if GEMINI_API_KEY:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._semaphore = asyncio.Semaphore(SCOUT_CONCURRENCY_LIMIT)

    # ------------------------------------------------------------------
    # Public API — async batch enrichment
    # ------------------------------------------------------------------
    async def enrich_batch(self, articles: List[NewsArticle]) -> List[ScoutArticle]:
        """Enrich all articles concurrently, capped by a semaphore."""
        sem = self._semaphore

        async def enrich_one(idx: int, article: NewsArticle) -> ScoutArticle:
            # Jittered stagger: spaces out semaphore acquisition so the API
            # sees a gentle trickle instead of a burst. The random jitter
            # (±0.1s around the deterministic delay) prevents Tenacity retries
            # from re-synchronizing and spiking the API again.
            await asyncio.sleep(idx * 0.1 + random.uniform(0, 0.2))
            async with sem:
                print(f"\n  🔍 Scout [{idx+1}/{len(articles)}] {article.title[:70]}...")
                try:
                    return await self._enrich(article)
                except Exception as e:
                    print(f"    ❌ Scout enrichment failed: {e}")
                    return ScoutArticle(
                        source_bucket=article.source_bucket,
                        source_name=article.source_name,
                        title=article.title,
                        summary=article.summary or (article.content[:SCOUT_SUMMARY_MAX_CHARS] if article.content else ""),
                        url=article.url,
                        timestamp=article.timestamp,
                        ticker_tags=article.ticker_tags,
                    )

        tasks = [enrich_one(i, a) for i, a in enumerate(articles)]
        return list(await asyncio.gather(*tasks))

    async def _enrich(self, article: NewsArticle) -> ScoutArticle:
        importance = await self._evaluate_importance(article)
        print(f"    📊 Importance: {importance['score']:.2f} — {importance['reasoning']}")

        search_query = await self._generate_search_query(article)
        print(f"    🔍 Query: {search_query}")

        aggregated = self._aggregate_snippets(search_query)
        print(f"    📰 Aggregated context: {len(aggregated)} chars")

        return ScoutArticle(
            source_bucket=article.source_bucket,
            source_name=article.source_name,
            title=article.title,
            summary=article.summary or (article.content[:SCOUT_SUMMARY_MAX_CHARS] if article.content else ""),
            url=article.url,
            timestamp=article.timestamp,
            ticker_tags=article.ticker_tags,
            importance_score=importance["score"],
            importance_reasoning=importance["reasoning"],
            search_query=search_query,
            aggregated_content=aggregated,
        )

    # ------------------------------------------------------------------
    # Stage 1 – Importance evaluation via Gemini (async + retry)
    # ------------------------------------------------------------------
    async def _evaluate_importance(self, article: NewsArticle) -> dict:
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
            f"Summary: {article.summary or (article.content[:SCOUT_IMPORTANCE_PROMPT_CHARS] if article.content else 'N/A')}\n"
        )
        try:
            result = await self._call_gemini_with_retry(
                model=SCOUT_GEMINI_MODEL,
                prompt=prompt,
                temperature=SCOUT_IMPORTANCE_TEMPERATURE,
                max_tokens=SCOUT_IMPORTANCE_MAX_TOKENS,
            )
            return {
                "score": max(0.0, min(1.0, float(result.get("score", 0.5)))),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            print(f"    ⚠️  Gemini importance evaluation failed after retries: {e}")
            return self._heuristic_importance(article)

    def _heuristic_importance(self, article: NewsArticle) -> dict:
        text = (article.title + " " + (article.summary or article.content or "")).lower()
        score = 0.3
        for kw in SCOUT_HIGH_IMPACT_KWS:
            if kw in text:
                score = max(score, 0.7)
                break
        for kw in SCOUT_MEDIUM_IMPACT_KWS:
            if kw in text:
                score = max(score, 0.5)
        if article.source_name.lower() in SCOUT_HIGH_IMPACT_SOURCES:
            score = max(score, 0.5)
        return {"score": round(score, 2), "reasoning": "Heuristic fallback (no Gemini key or exhausted retries)"}

    # ------------------------------------------------------------------
    # Stage 2 – Search-query generation via Gemini (async + retry)
    # ------------------------------------------------------------------
    async def _generate_search_query(self, article: NewsArticle) -> str:
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
            f"Summary: {article.summary or (article.content[:SCOUT_QUERY_PROMPT_CHARS] if article.content else 'N/A')}\n"
        )
        try:
            text = await self._call_gemini_raw(
                model=SCOUT_GEMINI_MODEL,
                prompt=prompt,
                temperature=SCOUT_QUERY_TEMPERATURE,
                max_tokens=SCOUT_QUERY_MAX_TOKENS,
            )
            query = text.strip().strip('"').strip("'")
            return query if query else self._heuristic_query(article)
        except Exception as e:
            print(f"    ⚠️  Gemini query generation failed after retries: {e}")
            return self._heuristic_query(article)

    @staticmethod
    def _heuristic_query(article: NewsArticle) -> str:
        title = article.title
        for sep in [" - ", " | ", " — "]:
            if sep in title:
                title = title.split(sep)[0]
        return title[:150]

    # ------------------------------------------------------------------
    # Stage 3 – DuckDuckGo search & snippet aggregation (sync, no API key needed)
    # ------------------------------------------------------------------
    def _aggregate_snippets(self, query: str) -> str:
        snippets: List[str] = []
        try:
            with DDGS() as ddgs:
                for i, result in enumerate(ddgs.text(query, max_results=SCOUT_DDG_MAX_RESULTS)):
                    body = result.get("body", "").strip()
                    if body:
                        snippets.append(f"[{i+1}] {result.get('title', '')}\n{body}")
            if snippets:
                return "\n\n".join(snippets)
            if len(query.split()) > 3:
                shorter = " ".join(query.split()[:3])
                with DDGS() as ddgs:
                    for i, result in enumerate(ddgs.text(shorter, max_results=SCOUT_DDG_MAX_RESULTS)):
                        body = result.get("body", "").strip()
                        if body:
                            snippets.append(f"[{i+1}] {result.get('title', '')}\n{body}")
                if snippets:
                    return "\n\n".join(snippets)
        except Exception as e:
            print(f"    ⚠️  DuckDuckGo search failed: {e}")
        return ""

    # ------------------------------------------------------------------
    # Gemini helpers — async calls with retry for transient errors
    # ------------------------------------------------------------------
    @_llm_retry_decorator
    async def _call_gemini_with_retry(
        self, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> dict:
        """Call Gemini, return parsed JSON dict. Retries on 429/503/5xx."""
        text = await self._call_gemini_raw(model, prompt, temperature, max_tokens)
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if text.startswith("json"):
            text = text[4:].strip()
        return parse_json_with_repair(text)

    async def _call_gemini_raw(
        self, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """Call Gemini and return raw text. Uses run_in_executor for the sync SDK."""
        loop = asyncio.get_running_loop()

        def _sync_call():
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"temperature": temperature, "max_output_tokens": max_tokens},
                )
                return response.text.strip()
            except Exception as e:
                # Re-raise only retryable errors; others propagate to caller
                if _is_retryable_error(e):
                    raise  # Let tenacity retry it
                raise  # Propagate non-retryable errors too (caught by caller w/ heuristic fallback)

        return await loop.run_in_executor(None, _sync_call)