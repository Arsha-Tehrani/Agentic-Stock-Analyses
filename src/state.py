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

    # Portfolio Manager output (Agent 3)
    portfolio_recommendation: Optional["PortfolioRecommendation"]

    # Risk Reviewer output (Agent 4) — most recent critic verdict
    critic_feedback: Optional["CriticFeedback"]

    # Prior critic feedback piped back into Agent 3 prompt (for revision rounds)
    previous_critic_feedback: Optional["CriticFeedback"]

    # How many times Agent 3 ↔ Agent 4 have disagreed so far. Hard-capped
    # by RISK_REVIEW_MAX_ITERATIONS so the graph can never get stuck in a loop.
    risk_review_iterations: int


# =============================================================================
# Portfolio State — Mutable portfolio management data structure
# =============================================================================

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import json


@dataclass
class CashHolding:
    """Represents a cash holding (e.g., money market fund)."""
    ticker: str
    concentration_percent: float

    def to_dict(self) -> dict:
        return {"ticker": self.ticker, "concentration_percent": self.concentration_percent}

    @classmethod
    def from_dict(cls, data: dict) -> "CashHolding":
        return cls(
            ticker=data["ticker"],
            concentration_percent=data["concentration_percent"]
        )


@dataclass
class Position:
    """Represents a single position/holding in a sector."""
    ticker: str
    concentration_percent: float

    def to_dict(self) -> dict:
        return {"ticker": self.ticker, "concentration_percent": self.concentration_percent}

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        return cls(
            ticker=data["ticker"],
            concentration_percent=data["concentration_percent"]
        )


@dataclass
class SectorAllocation:
    """Represents a sector allocation with its holdings."""
    weight_percent: float
    sub_sector_bias: List[str] = field(default_factory=list)
    holdings: List[Position] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "weight_percent": self.weight_percent,
            "sub_sector_bias": self.sub_sector_bias,
            "holdings": [h.to_dict() for h in self.holdings]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SectorAllocation":
        return cls(
            weight_percent=data["weight_percent"],
            sub_sector_bias=data.get("sub_sector_bias", []),
            holdings=[Position.from_dict(h) for h in data.get("holdings", [])]
        )


@dataclass
class PortfolioAllocations:
    """Represents the full portfolio allocation structure."""
    total_value: float
    cash_reserves_percent: float
    cash_holdings: List[CashHolding] = field(default_factory=list)
    sectors: Dict[str, SectorAllocation] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_value": self.total_value,
            "cash_reserves_percent": self.cash_reserves_percent,
            "cash_holdings": [h.to_dict() for h in self.cash_holdings],
            "sectors": {k: v.to_dict() for k, v in self.sectors.items()}
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PortfolioAllocations":
        return cls(
            total_value=data["total_value"],
            cash_reserves_percent=data["cash_reserves_percent"],
            cash_holdings=[CashHolding.from_dict(h) for h in data.get("cash_holdings", [])],
            sectors={k: SectorAllocation.from_dict(v) for k, v in data.get("sectors", {}).items()}
        )


@dataclass
class MacroBaseline:
    """Represents the macroeconomic baseline."""
    interest_rate_trend: str
    inflation_trend: str
    market_regime: str

    def to_dict(self) -> dict:
        return {
            "interest_rate_trend": self.interest_rate_trend,
            "inflation_trend": self.inflation_trend,
            "market_regime": self.market_regime
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MacroBaseline":
        return cls(
            interest_rate_trend=data["interest_rate_trend"],
            inflation_trend=data["inflation_trend"],
            market_regime=data["market_regime"]
        )


@dataclass
class PortfolioState:
    """
    Mutable portfolio state that can be altered and persisted.
    This is the canonical representation of the current portfolio
    that flows through the pipeline and can be updated by the
    Portfolio Manager → Reviewer → Human approval workflow.
    """

    timestamp: str
    macro_baseline: MacroBaseline
    portfolio_allocations: PortfolioAllocations
    version: int = 1
    updated_by: str = "system"  # 'human', 'portfolio_manager', 'reviewer', 'system'
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization / database storage."""
        return {
            "timestamp": self.timestamp,
            "macro_baseline": self.macro_baseline.to_dict(),
            "portfolio_allocations": self.portfolio_allocations.to_dict(),
            "version": self.version,
            "updated_by": self.updated_by,
            "last_updated": self.last_updated
        }

    def to_market_state(self) -> CurrentMarketState:
        """Convert to CurrentMarketState TypedDict for compatibility with existing code."""
        return {
            "timestamp": self.timestamp,
            "macro_baseline": self.macro_baseline.to_dict(),
            "portfolio_allocations": self.portfolio_allocations.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PortfolioState":
        """Create from dictionary (e.g., from database or config)."""
        return cls(
            timestamp=data["timestamp"],
            macro_baseline=MacroBaseline.from_dict(data["macro_baseline"]),
            portfolio_allocations=PortfolioAllocations.from_dict(data["portfolio_allocations"]),
            version=data.get("version", 1),
            updated_by=data.get("updated_by", "system"),
            last_updated=data.get("last_updated", datetime.now().isoformat())
        )

    @classmethod
    def from_market_state(cls, market_state: CurrentMarketState, version: int = 1, updated_by: str = "system") -> "PortfolioState":
        """Create from a CurrentMarketState TypedDict."""
        return cls(
            timestamp=market_state["timestamp"],
            macro_baseline=MacroBaseline.from_dict(market_state["macro_baseline"]),
            portfolio_allocations=PortfolioAllocations.from_dict(market_state["portfolio_allocations"]),
            version=version,
            updated_by=updated_by,
            last_updated=datetime.now().isoformat()
        )

    def update_allocations(self, new_allocations: dict, updated_by: str) -> "PortfolioState":
        """
        Create a new PortfolioState with updated allocations.
        This is immutable - returns a new instance with incremented version.
        """
        new_allocs = PortfolioAllocations.from_dict(new_allocations)
        return PortfolioState(
            timestamp=self.timestamp,
            macro_baseline=self.macro_baseline,
            portfolio_allocations=new_allocs,
            version=self.version + 1,
            updated_by=updated_by,
            last_updated=datetime.now().isoformat()
        )

    def update_macro_baseline(self, new_baseline: dict, updated_by: str) -> "PortfolioState":
        """
        Create a new PortfolioState with updated macro baseline.
        This is immutable - returns a new instance with incremented version.
        """
        new_macro = MacroBaseline.from_dict(new_baseline)
        return PortfolioState(
            timestamp=datetime.now().strftime("%Y-%m-%d"),
            macro_baseline=new_macro,
            portfolio_allocations=self.portfolio_allocations,
            version=self.version + 1,
            updated_by=updated_by,
            last_updated=datetime.now().isoformat()
        )

    def get_total_sector_weight(self) -> float:
        """Calculate total sector weight percentage."""
        return sum(s.weight_percent for s in self.portfolio_allocations.sectors.values())

    def get_cash_weight(self) -> float:
        """Get cash reserves percentage."""
        return self.portfolio_allocations.cash_reserves_percent

    def validate(self) -> List[str]:
        """
        Validate the portfolio state.
        Returns list of validation errors (empty if valid).
        """
        errors = []
        total = self.get_total_sector_weight() + self.get_cash_weight()
        if abs(total - 100.0) > 0.01:
            errors.append(f"Total allocation is {total:.2f}%, expected 100%")

        # Check sector holdings sum to sector weight
        for sector_name, sector in self.portfolio_allocations.sectors.items():
            holdings_sum = sum(h.concentration_percent for h in sector.holdings)
            if abs(holdings_sum - sector.weight_percent) > 0.01:
                errors.append(
                    f"Sector '{sector_name}': holdings sum to {holdings_sum:.2f}%, "
                    f"but sector weight is {sector.weight_percent}%"
                )

        return errors

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Portfolio State (v{self.version}) - {self.timestamp}",
            f"  Macro Regime: {self.macro_baseline.market_regime}",
            f"  Cash: {self.get_cash_weight():.1f}%",
            f"  Sectors: {self.get_total_sector_weight():.1f}%",
        ]
        for name, sector in self.portfolio_allocations.sectors.items():
            lines.append(f"    {name}: {sector.weight_percent:.1f}% ({len(sector.holdings)} holdings)")
        return "\n".join(lines)


# =============================================================================
# Portfolio Manager — Agent 3 data structures
# =============================================================================


@dataclass
class TargetStock:
    """
    One target stock surfaced by the Researcher's proxy-hunt / momentum scan.

    role: how this stock relates to the incoming news flow:
        - "HEADLINE"   → the actual ticker named in the article(s)
        - "PROXY"      → discovered substitute that offers cheaper/safer exposure
        - "COMPETITOR" → direct peer to the headline ticker
        - "SUPPLIER"   → upstream beneficiary in the value chain
    momentum_flag: True when valuation looks rich but the LLM flagged strong
        institutional money flow / earnings momentum worth riding tactically.
    """

    ticker: str
    company_name: str
    role: str                              # HEADLINE | PROXY | COMPETITOR | SUPPLIER
    momentum_flag: bool = False
    valuation_thesis: str = ""
    momentum_thesis: str = ""
    evidence: List[str] = field(default_factory=list)
    market_cap: Optional[float] = None    # Populated by Finnhub enrichment (aiohttp)
    current_price: Optional[float] = None  # Populated by Finnhub enrichment (aiohttp)


@dataclass
class TargetResearchList:
    """
    Step-1 output of the Portfolio Manager: the researcher's annotated watchlist.
    """

    targets: List[TargetStock] = field(default_factory=list)
    queries_used: List[str] = field(default_factory=list)
    research_summary: str = ""

    def ticker_set(self) -> set:
        """Convenience: the set of all tickers surfaced."""
        return {t.ticker.upper() for t in self.targets}


@dataclass
class ProposedAction:
    """
    A single executable instruction emitted by the Book Runner (Step 2).

    Action vocabulary:
        - "EXPAND"  → increase the existing position (sector weight goes up)
        - "DILUTE"  → reduce the existing position to free up cash
        - "ADD"     → open a brand-new position not currently held
    Time_Horizon vocabulary:
        - "SHORT_TERM_MOMENTUM" → tactical ride; expects to exit quickly
        - "LONG_TERM_HOLD"      → structural thesis; not time-sensitive
    """

    ticker: str
    action: str
    time_horizon: str
    reasoning: str


@dataclass
class PortfolioRecommendation:
    """
    The strict, schema-locked final output of the Portfolio Manager node.
    This object is what routes to Agent 4 (The Risk Reviewer).
    """

    # Narrative analyses (Step 2 LLM)
    Portfolio_Impact_Assessment: str
    Abstract_Proxy_Discoveries: str
    Momentum_vs_Valuation_Analysis: str

    # The actual trade list (Step 2 LLM, validated by Python)
    Proposed_Actions: List[ProposedAction] = field(default_factory=list)

    # Routing
    proceed_to_risk_reviewer: bool = True

    # Audit metadata
    regime_significance_score: int = 0
    research_summary: str = ""
    queries_used: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """JSON-serializable view of the recommendation (used for logging / DB)."""
        return {
            "Portfolio_Impact_Assessment": self.Portfolio_Impact_Assessment,
            "Abstract_Proxy_Discoveries": self.Abstract_Proxy_Discoveries,
            "Momentum_vs_Valuation_Analysis": self.Momentum_vs_Valuation_Analysis,
            "Proposed_Actions": [
                {
                    "Ticker": a.ticker,
                    "Action": a.action,
                    "Time_Horizon": a.time_horizon,
                    "Reasoning": a.reasoning,
                }
                for a in self.Proposed_Actions
            ],
            "proceed_to_risk_reviewer": self.proceed_to_risk_reviewer,
            "regime_significance_score": self.regime_significance_score,
            "research_summary": self.research_summary,
            "queries_used": self.queries_used,
            "generated_at": self.generated_at,
        }


# =============================================================================
# Risk Reviewer — Agent 4 data structures
# =============================================================================


@dataclass
class CriticFeedback:
    """
    Output of the Risk Reviewer (Agent 4) — the dual-check critic verdict.

    approval_status=True  → forward to Agent 5 (Output Reporter).
    approval_status=False → pipe critic_feedback back to Agent 3 for revision.
    """

    optimization_verdict: str
    risk_flaw_analysis: str
    approval_status: bool = False
    critic_feedback: str = ""   # Empty when approved; specific fix instructions when rejected.
    iteration: int = 1           # Which round of review this was (1, 2, 3, ...).
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """JSON-serializable view (used for logging / piping back to Agent 3)."""
        return {
            "optimization_verdict": self.optimization_verdict,
            "risk_flaw_analysis": self.risk_flaw_analysis,
            "approval_status": self.approval_status,
            "critic_feedback": self.critic_feedback,
            "iteration": self.iteration,
            "generated_at": self.generated_at,
        }
