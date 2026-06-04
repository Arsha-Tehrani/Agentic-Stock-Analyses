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