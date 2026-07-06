"""
ToneAnalystNode.py – Emotional tonality analysis engine.

For every ScoutArticle, this node uses an LLM to:
  1. Detect the emotional tone of the writing (fear, euphoria, panic, calm, etc.)
  2. Quantify the density of factual/numeric content
  3. Compute a disparity score showing how much the emotional framing
     diverges from the factual substance

When the emotional_disparity is high, it signals potential market
overreaction or underreaction that downstream agents can act on.

Concurrency: analyses articles with asyncio.gather capped by asyncio.Semaphore(N)
to avoid overwhelming the Gemini API. Retries 429/503 with exponential backoff.
JSON parsing uses Gemini's native structured output (Pydantic schema) to
eliminate markdown fences, preamble contamination, and unterminated strings.

System instruction is separated from user content to improve instruction
adherence. Safety-filtered responses (text=None) are caught and fall back
to heuristics.

All tunable constants are in src/config.py.
"""

import asyncio
import random
import re
from typing import List, Optional

from google import genai
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

from src.state import ScoutArticle, EmotionalAnalysis
from src.utils.cost_logger import log_gemini_usage
from src.config import (
    GEMINI_API_KEY,
    TONALITY_GEMINI_MODEL,
    DISPARITY_THRESHOLD,
    TONALITY_TEMPERATURE,
    TONALITY_MAX_TOKENS,
    TONALITY_ANALYSIS_MAX_CHARS,
    TONE_CONCURRENCY_LIMIT,
    TONALITY_NEGATIVE_EMOTIONAL,
    TONALITY_POSITIVE_EMOTIONAL,
    TONALITY_FACTUAL_INDICATORS,
)


class TonalitySchema(BaseModel):
    """Native structured output schema for Gemini tonality analysis.
    Gemini is forced to return data matching this exact structure."""
    emotional_score: float = Field(..., description="Emotional tone from -1.0 (fear/panic) to +1.0 (euphoria/greed)")
    factual_score: float = Field(..., description="Factual density from 0.0 (pure opinion) to 1.0 (data-driven)")
    tonality_label: str = Field(..., description="One of: alarmist, measured, euphoric, clinical, balanced, sensationalist")
    reasoning: str = Field(..., description="1-2 sentence explanation of emotional/factual divergence")
    key_emotional_phrases: List[str] = Field(default_factory=list, description="Up to 3 key emotional phrases from the text")
    key_factual_claims: List[str] = Field(default_factory=list, description="Up to 3 key factual claims from the text")


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
        stop=stop_after_attempt(3),
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


class ToneAnalystNode:
    """
    Analyses the emotional tonality of enriched articles, separating
    emotional language from factual/numeric content to compute a
    disparity score.

    Uses async concurrency with a semaphore to limit simultaneous LLM calls.
    System instruction is separated from contents to improve adherence.
    JSON output enforced via Gemini native structured output (Pydantic schema).
    """

    def __init__(self):
        self._client: Optional[genai.Client] = None
        if GEMINI_API_KEY:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._semaphore = asyncio.Semaphore(TONE_CONCURRENCY_LIMIT)

    # ------------------------------------------------------------------
    # Public API — async batch analysis
    # ------------------------------------------------------------------
    async def analyze_batch(self, articles: List[ScoutArticle]) -> List[ScoutArticle]:
        """Analyze a batch of ScoutArticles concurrently, capped by a semaphore."""
        sem = self._semaphore

        async def analyze_one(idx: int, article: ScoutArticle) -> ScoutArticle:
            await asyncio.sleep(idx * 0.1 + random.uniform(0, 0.2))
            async with sem:
                print(f"\n  🎭 Tone [{idx+1}/{len(articles)}] {article.title[:70]}...")
                try:
                    return await self._analyze(article)
                except Exception as e:
                    print(f"    ❌ Tone analysis failed: {e}")
                    article.emotional_analysis = self._neutral_fallback()
                    return article

        tasks = [analyze_one(i, a) for i, a in enumerate(articles)]
        results = await asyncio.gather(*tasks)

        high_disparity_count = sum(
            1 for a in results
            if a.emotional_analysis and a.emotional_analysis.disparity_score >= DISPARITY_THRESHOLD
        )
        print(f"\n  📊 Tone analysis complete: {high_disparity_count}/{len(articles)} articles show high emotional disparity")
        return list(results)

    async def _analyze(self, article: ScoutArticle) -> ScoutArticle:
        analysis = await self._run_tonality_analysis(article)
        article.emotional_analysis = analysis

        if analysis.disparity_score >= DISPARITY_THRESHOLD:
            print(f"    ⚠️  HIGH DISPARITY ({analysis.disparity_score:.2f}): {analysis.tonality_label} — {analysis.reasoning[:100]}")
        else:
            print(f"    😶 Tonality: {analysis.tonality_label} | emotional={analysis.emotional_score:.2f} factual={analysis.factual_score:.2f} disparity={analysis.disparity_score:.2f}")

        return article

    # ------------------------------------------------------------------
    # Core LLM-based analysis (native structured output)
    # ------------------------------------------------------------------
    async def _run_tonality_analysis(self, article: ScoutArticle) -> EmotionalAnalysis:
        if not self._client:
            return self._heuristic_tonality(article)

        analysis_text = (
            article.aggregated_content
            or article.summary
            or (article.title or "")
        )

        system_instruction = (
            "You are an expert media analyst specializing in financial journalism. "
            "Analyze news article text on TWO separate dimensions:\n\n"
            "1. EMOTIONAL TONE: How emotionally charged is the language? "
            "Rate from -1.0 (extreme fear/panic/doom) to +1.0 (extreme euphoria/greed/elation). "
            "0.0 means completely neutral/objective language.\n\n"
            "2. FACTUAL DENSITY: How much of the text is substantive facts, data, "
            "statistics, percentages, dollar amounts, or specific named entities? "
            "Rate from 0.0 (pure opinion/emotion, no factual backing) to 1.0 "
            "(almost entirely data-driven with little emotional framing).\n\n"
            "3. Extract up to 3 key emotional phrases and up to 3 key factual claims.\n\n"
            "4. Explain in 1-2 sentences WHY the emotional tone and factual density "
            "diverge (if they do).\n\n"
            "Valid tonality labels: alarmist, measured, euphoric, clinical, balanced, sensationalist"
        )

        contents = (
            f"ARTICLE TEXT:\n{analysis_text[:TONALITY_ANALYSIS_MAX_CHARS]}\n\n"
            f"Title: {article.title}\n"
            f"Source: {article.source_name} ({article.source_bucket})\n"
        )

        try:
            result = await self._call_gemini_structured(
                model=TONALITY_GEMINI_MODEL,
                system_instruction=system_instruction,
                contents=contents,
                temperature=TONALITY_TEMPERATURE,
                max_tokens=TONALITY_MAX_TOKENS,
                schema=TonalitySchema,
            )

            emotional_score = max(-1.0, min(1.0, float(result.emotional_score)))
            factual_score = max(0.0, min(1.0, float(result.factual_score)))

            factual_dampening = max(0.0, 1.0 - factual_score * 0.5)
            disparity = max(0.0, round(abs(emotional_score) * factual_dampening, 2))

            valid_labels = {"alarmist", "measured", "euphoric", "clinical", "balanced", "sensationalist"}
            tonality_label = result.tonality_label
            if tonality_label not in valid_labels:
                tonality_label = "balanced"

            return EmotionalAnalysis(
                emotional_score=round(emotional_score, 2),
                factual_score=round(factual_score, 2),
                disparity_score=disparity,
                tonality_label=tonality_label,
                reasoning=result.reasoning,
                key_emotional_phrases=result.key_emotional_phrases,
                key_factual_claims=result.key_factual_claims,
            )
        except Exception as e:
            print(f"    ⚠️  LLM tonality analysis failed after retries: {e}")
            return self._heuristic_tonality(article)

    # ------------------------------------------------------------------
    # Heuristic fallback (no Gemini key or exhausted retries)
    # ------------------------------------------------------------------
    def _heuristic_tonality(self, article: ScoutArticle) -> EmotionalAnalysis:
        text = (
            (article.title or "")
            + " "
            + (article.summary or "")
            + " "
            + (article.aggregated_content or "")
        ).lower()

        emotional_hits = 0
        emotional_valence = 0.0
        for kw in TONALITY_NEGATIVE_EMOTIONAL:
            count = len(re.findall(r'\b' + re.escape(kw) + r'\b', text))
            if count:
                emotional_hits += count
                emotional_valence -= 0.15 * count
        for kw in TONALITY_POSITIVE_EMOTIONAL:
            count = len(re.findall(r'\b' + re.escape(kw) + r'\b', text))
            if count:
                emotional_hits += count
                emotional_valence += 0.15 * count

        emotional_score = max(-1.0, min(1.0, round(emotional_valence, 2)))

        factual_hits = 0
        for kw in TONALITY_FACTUAL_INDICATORS:
            factual_hits += text.count(kw)
        factual_score = max(0.0, min(1.0, round(factual_hits * 0.08, 2)))

        factual_dampening = max(0.0, 1.0 - factual_score * 0.5)
        disparity = max(0.0, round(abs(emotional_score) * factual_dampening, 2))

        if emotional_hits == 0 and factual_hits == 0:
            tonality_label = "balanced"
        elif emotional_hits > factual_hits * 2 and abs(emotional_score) > 0.4:
            tonality_label = "alarmist" if emotional_score < 0 else "euphoric"
        elif factual_hits > emotional_hits * 3:
            tonality_label = "clinical"
        elif abs(emotional_score) < 0.2:
            tonality_label = "measured"
        else:
            tonality_label = "sensationalist" if abs(emotional_score) > 0.5 else "balanced"

        return EmotionalAnalysis(
            emotional_score=emotional_score,
            factual_score=factual_score,
            disparity_score=disparity,
            tonality_label=tonality_label,
            reasoning="Heuristic fallback (keyword-based analysis, no LLM available).",
            key_emotional_phrases=[],
            key_factual_claims=[],
        )

    # ------------------------------------------------------------------
    # Neutral fallback for unexpected errors
    # ------------------------------------------------------------------
    @staticmethod
    def _neutral_fallback() -> EmotionalAnalysis:
        return EmotionalAnalysis(
            emotional_score=0.0,
            factual_score=0.5,
            disparity_score=0.0,
            tonality_label="balanced",
            reasoning="Analysis failed — defaulting to neutral.",
            key_emotional_phrases=[],
            key_factual_claims=[],
        )

    # ------------------------------------------------------------------
    # Gemini helpers
    # ------------------------------------------------------------------
    @_llm_retry_decorator
    async def _call_gemini_structured(
        self, model: str, system_instruction: str, contents: str,
        temperature: float, max_tokens: int, schema: type,
    ):
        """Call Gemini using native structured output (Pydantic schema).
        Returns the Pydantic model instance directly — no manual JSON parsing,
        no code fence stripping, no unterminated-string repair."""
        loop = asyncio.get_running_loop()

        def _sync_call():
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "system_instruction": system_instruction,
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )
            log_gemini_usage("ToneAnalystNode", model, response)
            if response.text is None:
                raise ValueError("Gemini returned None (safety-filtered response)")
            return schema.model_validate_json(response.text)

        return await loop.run_in_executor(None, _sync_call)

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
            log_gemini_usage("ToneAnalystNode", model, response)
            # Guard against safety-filtered responses where .text is None
            if response.text is None:
                return None
            return response.text.strip()

        return await loop.run_in_executor(None, _sync_call)