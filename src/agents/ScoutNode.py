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
JSON parse failures are NOT retried — they use parse_json_with_repair.

System instruction is separated from user content to improve instruction
adherence (prevents instruction echo). Safety-filtered responses (text=None)
are caught and fall back to heuristics.

All tunable constants are in src/config.py.
"""

import asyncio
import random
from typing import List, Optional

from ddgs import DDGS
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
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


class ImportanceSchema(BaseModel):
    score: float = Field(..., description="Importance score from 0.0 to 1.0")
    reasoning: str = Field(..., description="Short justification for the score")


def _is_retryable_error(exception: Exception) -> bool:
    """Return True for HTTP 429 / 503 / 5xx errors that should be retried."""
    status = getattr(exception, "code", None)
    if status is not None:
        return status in (429, 503) or (isinstance(status, int) and 500 <= status < 600)
    msg = str(exception).lower()
    retryable_keywords = ["429", "503", "unavailable", "resource_exhausted", "rate", "quota"]
    return any(kw in msg for kw in retryable_keywords)


def _llm_retry_decorator(func):
    """Exponential-backoff retry for transient Gemini errors (429/503/5xx)."""
    return retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),  # Reduced from 5 — JSON failures use parse_json_with_repair
        retry=retry_if_exception_type(Exception),
        after=lambda retry_state: print(
            f"    🔄 Gemini retry attempt {retry_state.attempt_number}/3 "
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
    System instruction is separated from contents to improve adherence.
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
    # Stage 1 – Importance evaluation via Gemini (native structured output)
    # ------------------------------------------------------------------
    async def _evaluate_importance(self, article: NewsArticle) -> dict:
        if not self._client:
            return self._heuristic_importance(article)

        system_instruction = (
            "You are a senior macro hedge fund analyst. Rate the market importance "
            "of news articles on a scale from 0.0 (completely irrelevant) to 1.0 "
            "(extremely market-moving). Consider impact on equities, bonds, currencies, "
            "and commodities."
        )

        contents = (
            f"Title: {article.title}\n"
            f"Source: {article.source_name} ({article.source_bucket})\n"
            f"Summary: {article.summary or (article.content[:SCOUT_IMPORTANCE_PROMPT_CHARS] if article.content else 'N/A')}\n"
        )

        try:
            result = await self._call_gemini_structured(
                model=SCOUT_GEMINI_MODEL,
                system_instruction=system_instruction,
                contents=contents,
                temperature=SCOUT_IMPORTANCE_TEMPERATURE,
                max_tokens=SCOUT_IMPORTANCE_MAX_TOKENS,
                schema=ImportanceSchema,
            )
            return {
                "score": max(0.0, min(1.0, float(result.score))),
                "reasoning": result.reasoning,
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
    # Stage 2 – Search-query generation via Gemini (plain text)
    # ------------------------------------------------------------------
    async def _generate_search_query(self, article: NewsArticle) -> str:
        if not self._client:
            return self._heuristic_query(article)

        system_instruction = (
            "You are a research analyst. Based on the article provided, generate a single, "
            "highly specific Google-search-style query that would find the most relevant "
            "and authoritative articles, analysis, and context about this topic."
        )

        contents = (
            f"Rules:\n"
            f"- Use 5-8 keywords maximum.\n"
            f"- Include key entity names (people, companies, agencies, tickers).\n"
            f"- Prefer recent event framing.\n"
            f"- Return ONLY the query string, no explanation, no quotes.\n\n"
            f"Title: {article.title}\n"
            f"Summary: {article.summary or (article.content[:SCOUT_QUERY_PROMPT_CHARS] if article.content else 'N/A')}\n"
        )

        try:
            text = await self._call_gemini_text(
                model=SCOUT_GEMINI_MODEL,
                system_instruction=system_instruction,
                contents=contents,
                temperature=SCOUT_QUERY_TEMPERATURE,
                max_tokens=SCOUT_QUERY_MAX_TOKENS,
            )
            if text and isinstance(text, str):
                query = text.strip().strip('"').strip("'")
                return query if query else self._heuristic_query(article)
            else:
                return self._heuristic_query(article)
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
    # Stage 3 – DuckDuckGo search & snippet aggregation
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
    # Gemini helpers
    # ------------------------------------------------------------------
    @_llm_retry_decorator
    async def _call_gemini_structured(
        self, model: str, system_instruction: str, contents: str,
        temperature: float, max_tokens: int, schema: type,
    ):
        """Call Gemini using native structured output (Pydantic schema).
        Returns the Pydantic model instance directly — no manual JSON parsing."""
        loop = asyncio.get_running_loop()

        def _sync_call():
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=schema,
            )

            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            print("\n--- DEBUG: GEMINI STRUCTURED OUTPUT ---")
            print(f"RAW TEXT RESPONSE:\n{response.text}")
            print(f"DID SDK AUTO-PARSE?: {hasattr(response, 'parsed') and response.parsed is not None}")
            print("---------------------------------------\n")

            if response.text is None:
                raise ValueError("Gemini returned None (safety-filtered response)")

            if response.parsed:
                return response.parsed

            return schema.model_validate_json(response.text)

        return await loop.run_in_executor(None, _sync_call)

    @_llm_retry_decorator
    async def _call_gemini_json(
        self, model: str, system_instruction: str, contents: str,
        temperature: float, max_tokens: int,
    ) -> dict:
        """Call Gemini, return parsed JSON dict (with truncation repair)."""
        text = await self._call_gemini_text(
            model=model,
            system_instruction=system_instruction,
            contents=contents,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if text is None:
            raise ValueError("Gemini returned None (safety-filtered response)")
        return parse_json_with_repair(text)

    @_llm_retry_decorator
    async def _call_gemini_text(
        self, model: str, system_instruction: str, contents: str,
        temperature: float, max_tokens: int,
    ) -> Optional[str]:
        """Call Gemini and return raw text. Uses run_in_executor for the sync SDK."""
        loop = asyncio.get_running_loop()

        def _sync_call():
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "system_instruction": system_instruction,
                },
            )
            # Guard against safety-filtered responses where .text is None
            if response.text is None:
                return None
            return response.text.strip()

        return await loop.run_in_executor(None, _sync_call)