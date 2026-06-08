"""
ToneAnalystNode.py – Emotional tonality analysis engine.

For every ScoutArticle, this node uses an LLM to:
  1. Detect the emotional tone of the writing (fear, euphoria, panic, calm, etc.)
  2. Quantify the density of factual/numeric content
  3. Compute a disparity score showing how much the emotional framing
     diverges from the factual substance

When the emotional_disparity is high, it signals potential market
overreaction or underreaction that downstream agents can act on.

All tunable constants are in src/config.py.
"""

import json
import re
from typing import List, Optional

from google import genai

from src.state import ScoutArticle, EmotionalAnalysis
from src.config import (
    GEMINI_API_KEY,
    TONALITY_GEMINI_MODEL,  # Per-agent model config
    DISPARITY_THRESHOLD,
    TONALITY_TEMPERATURE,
    TONALITY_MAX_TOKENS,
    TONALITY_ANALYSIS_MAX_CHARS,
    TONALITY_NEGATIVE_EMOTIONAL,
    TONALITY_POSITIVE_EMOTIONAL,
    TONALITY_FACTUAL_INDICATORS,
)


class ToneAnalystNode:
    """
    Analyses the emotional tonality of enriched articles, separating
    emotional language from factual/numeric content to compute a
    disparity score.
    """

    def __init__(self):
        self._client: Optional[genai.Client] = None
        if GEMINI_API_KEY:
            self._client = genai.Client(api_key=GEMINI_API_KEY)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, article: ScoutArticle) -> ScoutArticle:
        """
        Analyse a single ScoutArticle and attach its EmotionalAnalysis.
        Returns the same article (mutated in-place for convenience).
        """
        analysis = self._run_tonality_analysis(article)
        article.emotional_analysis = analysis

        if analysis.disparity_score >= DISPARITY_THRESHOLD:
            print(f"    ⚠️  HIGH DISPARITY ({analysis.disparity_score:.2f}): {analysis.tonality_label} — {analysis.reasoning[:100]}")
        else:
            print(f"    😶 Tonality: {analysis.tonality_label} | emotional={analysis.emotional_score:.2f} factual={analysis.factual_score:.2f} disparity={analysis.disparity_score:.2f}")

        return article

    def analyze_batch(self, articles: List[ScoutArticle]) -> List[ScoutArticle]:
        """Analyze a batch of ScoutArticles for emotional tonality."""
        high_disparity_count = 0
        for i, article in enumerate(articles):
            print(f"\n  🎭 Tone [{i+1}/{len(articles)}] {article.title[:70]}...")
            try:
                self.analyze(article)
                if article.emotional_analysis and article.emotional_analysis.disparity_score >= DISPARITY_THRESHOLD:
                    high_disparity_count += 1
            except Exception as e:
                print(f"    ❌ Tone analysis failed: {e}")
                article.emotional_analysis = self._neutral_fallback()

        print(f"\n  📊 Tone analysis complete: {high_disparity_count}/{len(articles)} articles show high emotional disparity")
        return articles

    # ------------------------------------------------------------------
    # Core LLM-based analysis
    # ------------------------------------------------------------------
    def _run_tonality_analysis(self, article: ScoutArticle) -> EmotionalAnalysis:
        if not self._client:
            return self._heuristic_tonality(article)

        # Build the richest text we have — prefer aggregated content (Scout's
        # contextual snippets) over the raw summary because it gives the LLM
        # more data points to assess emotional vs factual framing.
        analysis_text = (
            article.aggregated_content
            or article.summary
            or (article.title or "")
        )

        prompt = (
            "You are an expert media analyst specializing in financial journalism. "
            "Analyze the following news article text on TWO separate dimensions:\n\n"
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
            "Respond ONLY with valid JSON in this exact format:\n"
            '{"emotional_score": -0.65, "factual_score": 0.3, '
            '"tonality_label": "alarmist", '
            '"reasoning": "The article uses vivid fear-inducing language about market crash '
            'but cites only a single 0.3% decline as evidence.", '
            '"key_emotional_phrases": ["panic selling", "wiped out"], '
            '"key_factual_claims": ["S&P 500 fell 0.3%", "volume was 2.1M shares"]}\n\n'
            "Valid tonality labels: alarmist, measured, euphoric, clinical, balanced, sensationalist\n\n"
            f"ARTICLE TEXT:\n{analysis_text[:TONALITY_ANALYSIS_MAX_CHARS]}\n\n"
            f"Title: {article.title}\n"
            f"Source: {article.source_name} ({article.source_bucket})\n"
        )

        try:
            response = self._client.models.generate_content(
                model=TONALITY_GEMINI_MODEL,
                contents=prompt,
                config={"temperature": TONALITY_TEMPERATURE, "max_output_tokens": TONALITY_MAX_TOKENS},
            )
            text = response.text.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            # Sometimes the model wraps in ```json ... ```
            if text.startswith("json"):
                text = text[4:].strip()

            result = json.loads(text)

            emotional_score = max(-1.0, min(1.0, float(result.get("emotional_score", 0.0))))
            factual_score = max(0.0, min(1.0, float(result.get("factual_score", 0.5))))

            # Disparity = how much the emotional magnitude exceeds factual backing
            # A score of 0 means emotional intensity matches factual substance
            # Higher values mean the emotion is disproportionate to the facts
            disparity = max(0.0, round(abs(emotional_score) - factual_score, 2))

            valid_labels = {"alarmist", "measured", "euphoric", "clinical", "balanced", "sensationalist"}
            tonality_label = result.get("tonality_label", "balanced")
            if tonality_label not in valid_labels:
                tonality_label = "balanced"

            return EmotionalAnalysis(
                emotional_score=round(emotional_score, 2),
                factual_score=round(factual_score, 2),
                disparity_score=disparity,
                tonality_label=tonality_label,
                reasoning=result.get("reasoning", ""),
                key_emotional_phrases=result.get("key_emotional_phrases", []),
                key_factual_claims=result.get("key_factual_claims", []),
            )
        except Exception as e:
            print(f"    ⚠️  LLM tonality analysis failed: {e}")
            return self._heuristic_tonality(article)

    # ------------------------------------------------------------------
    # Heuristic fallback (no Gemini key or API failure)
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

        disparity = max(0.0, round(abs(emotional_score) - factual_score, 2))

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