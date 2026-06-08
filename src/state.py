"""
state.py – Data structures flowing through the Agentic Workflow pipeline.

After the Scout enrichment phase, every raw article is transformed into a
ScoutArticle containing:
  - The original metadata (source, title, URL, timestamp)
  - The raw summary/excerpt from the source
  - Gemini-derived importance score
  - Gemini-generated search query
  - Aggregated context from top-5 DuckDuckGo search results
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class EmotionalAnalysis:
    """
    LLM-driven emotional tonality analysis that separates the emotional
    language of an article from its factual/numeric content.

    Key insight: emotional_disparity = |emotional_score - factual_score|
    A high disparity suggests the article's tone is significantly more
    charged (positively or negatively) than the actual data warrants.
    """

    emotional_score: float          # -1.0 (very negative) to +1.0 (very positive)
    factual_score: float            # 0.0 (no facts/numbers) to 1.0 (dense factual content)
    disparity_score: float          # abs(emotional_score) - factual_score, clipped to 0-1
    tonality_label: str             # "alarmist", "measured", "euphoric", "clinical", "balanced", "sensationalist"
    reasoning: str                  # LLM explanation of the disparity
    key_emotional_phrases: List[str] = field(default_factory=list)
    key_factual_claims: List[str] = field(default_factory=list)


@dataclass
class ScoutArticle:
    """
    Enriched article returned by the Scout node.

    This is the canonical data unit that flows into the Regime Analyst
    for classification and downstream decision-making.
    """

    # ── Original source metadata ──
    source_bucket: str          # "Wires" | "Macro_Blogs" | "Regional"
    source_name: str            # e.g. "Bloomberg", "Reuters", "Macro Compass"
    title: str
    summary: str                # Raw snippet/excerpt from the original source
    url: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    ticker_tags: List[str] = field(default_factory=list)

    # ── Scout enrichment (populated by ScoutNode) ──
    importance_score: float = 0.0          # Gemini grade 0.0 – 1.0
    importance_reasoning: str = ""         # Gemini's short justification
    search_query: str = ""                 # Gemini-generated Google-style query
    aggregated_content: str = ""           # Concatenated DDG snippets (rich context)

    # ── Emotional tonality analysis (populated by ToneAnalystNode) ──
    emotional_analysis: Optional[EmotionalAnalysis] = None

    # ── Related articles cluster (populated by ClusterFinder for high-disparity articles) ──
    related_articles: List["ScoutArticle"] = field(default_factory=list)


# =============================================================================
# Regime Analyst — Agent 2 data structures
# =============================================================================

from typing import TypedDict, Dict, Any  # noqa: E402


class CurrentMarketState(TypedDict, total=False):
    """Baseline market state used by Regime Analyst as the reference frame."""
    timestamp: str
    macro_baseline: Dict[str, str]
    portfolio_allocations: Dict[str, Any]


@dataclass
class RegimeAnalysis:
    """
    Output of the Regime Analyst node.
    
    The LLM evaluates three factors on a 1-10 scale, then Python computes the
    weighted Significance_Score on a 0-100 integer scale.
    
    S = α*M + β*R + γ*E   (each factor scored 1-10, formula maps to 0-100)
    """

    Macro_Analysis: str                          # Explanation of baseline economic shifts
    Rotation_Analysis: str                       # Identification of sector/intra-sector flows
    Emotional_Arbitrage_Analysis: str             # Overreaction or underreaction analysis

    # Raw LLM scores (1-10 per factor)
    macro_score: int = 1                         # M — Macroeconomic Impact (1-10)
    rotation_score: int = 1                      # R — Capital Rotation Intensity (1-10)
    emotional_arbitrage_score: int = 1           # E — Emotional Arbitrage Gap (1-10)

    # Computed by Python (not the LLM)
    Significance_Score: int = 0                  # 0-100 weighted composite
    proceed_to_portfolio_manager: bool = False   # True if Significance_Score > threshold


class GraphState(TypedDict, total=False):
    """Top-level state object flowing through the agent graph."""
    # Input articles (post-Scout enrichment + tonality analysis)
    articles: List[ScoutArticle]

    # Current market baseline (loaded from config or external source)
    market_state: CurrentMarketState

    # Regime Analyst output
    regime_analysis: Optional[RegimeAnalysis]

    # Final routing decision
    proceed_to_portfolio_manager: bool
