"""
config.py – Centralized configuration for the Agentic Workflow pipeline.

All tunable constants live here so they can be adjusted in one place
without digging through individual agent files.

To override API keys, set environment variables before running:
    export GEMINI_API_KEY="your_key_here"
    export FINNHUB_API_KEY="your_key_here"
"""

import os

# =============================================================================
# LLM / Gemini Configuration (shared by ScoutNode & ToneAnalystNode)
# =============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# =============================================================================
# ScoutNode – Enrichment Engine
# =============================================================================
# DuckDuckGo snippet aggregation
SCOUT_DDG_MAX_RESULTS = 7          # How many DDG results to fetch per query
SCOUT_SUMMARY_MAX_CHARS = 300      # Fallback: chars taken from content when summary is empty

# Gemini importance evaluation
SCOUT_IMPORTANCE_TEMPERATURE = 0.1  # Low temp for consistent scoring
SCOUT_IMPORTANCE_MAX_TOKENS = 100   # Enough for a short JSON response
SCOUT_IMPORTANCE_PROMPT_CHARS = 500  # Max summary chars fed into the importance prompt

# Gemini search-query generation
SCOUT_QUERY_TEMPERATURE = 0.2
SCOUT_QUERY_MAX_TOKENS = 60
SCOUT_QUERY_PROMPT_CHARS = 400      # Max summary chars fed into the query prompt

# Heuristic importance fallback (keyword-based when no Gemini key)
SCOUT_HIGH_IMPACT_KWS = [
    "fed", "federal reserve", "interest rate", "cpi", "inflation",
    "recession", "gdp", "employment", "nonfarm", "payroll",
    "central bank", "policymaker", "tariff", "sanction",
    "earnings", "corporate profit", "acquisition", "merger",
    "stimulus", "jolts", "pmi", "manufacturing",
]
SCOUT_MEDIUM_IMPACT_KWS = [
    "housing", "consumer", "retail", "trade deficit", "treasury",
    "bond yield", "sp500", "nasdaq", "dow", "volatility",
    "currency", "dollar", "euro", "yen", "emerging market",
]
SCOUT_HIGH_IMPACT_SOURCES = {"bloomberg", "reuters", "wsj", "financial times"}

# =============================================================================
# ToneAnalystNode – Emotional Tonality Analysis
# =============================================================================
DISPARITY_THRESHOLD = 0.35          # Disparity ≥ this triggers cluster search
TONALITY_TEMPERATURE = 0.15         # Low temp for consistent tonality scores
TONALITY_MAX_TOKENS = 400           # Room for JSON + reasoning + phrases/claims
TONALITY_ANALYSIS_MAX_CHARS = 3000  # Max text chars fed into the tonality prompt

# Heuristic tonality fallback keywords
TONALITY_NEGATIVE_EMOTIONAL = [
    "panic", "crash", "plunge", "fear", "doom", "collapse", "meltdown",
    "turmoil", "chaos", "crisis", "devastating", "catastrophic", "bloodbath",
    "rout", "carnage", "freefall", "tanked", "nosedive", "wiped out",
    "alarm", "scare", "dread", "apocalypse",
]
TONALITY_POSITIVE_EMOTIONAL = [
    "surge", "soar", "euphoria", "boom", "rocket", "skyrocket", "mania",
    "frenzy", "euphoric", "bullish", "explosive", "unstoppable", "roaring",
    "triumph", "bonanza",
]
TONALITY_FACTUAL_INDICATORS = [
    "%", "percent", "bps", "basis points", "$", "dollar", "billion",
    "million", "trillion", "index", "points", "yield", "rate",
    "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9",
    "gdp", "cpi", "ppi", "pmi", "unemployment", "nonfarm", "payrolls",
    "earnings per share", "revenue", "quarter", "fiscal",
]

# =============================================================================
# ClusterFinder – Related-Article Clustering
# =============================================================================
CLUSTER_DAYS_WINDOW = 7              # ±N day window for related article search
CLUSTER_MAX_RELATED = 10             # Max related articles to retrieve per cluster
CLUSTER_DISPARITY_THRESHOLD = 0.35   # Same as ToneAnalyst threshold (sync'd)

# =============================================================================
# Ingestion Sources
# =============================================================================
# Finnhub wire API key
WIRE_API_KEY = os.environ.get(
    "FINNHUB_API_KEY",
    "d8g80l9r01qlgcuhr95gd8g80l9r01qlgcuhr960",
)

# Macro blog scraping targets
BLOG_TARGETS = [
    "https://www.bespokepremium.com/interactive/blog/",
    "https://macrocompass.substack.com/",
]

# Regional RSS feeds
REGIONAL_FEEDS = {
    "Nikkei Asia": "https://services.nikkei.com/core/v1/rss/asia/news.xml",
    "Al Jazeera":   "https://www.aljazeera.com/xml/rss/all.xml",
}

# =============================================================================
# Ingestor Limits — max articles fetched per source per run
# =============================================================================
WIRE_MAX_ARTICLES = 20       # Finnhub wire headlines
RSS_MAX_PER_FEED = 10        # Regional RSS feed entries per source

# =============================================================================
# Regime Analyst – Capital Rotation & Macro Regime Detection
# =============================================================================
# LLM temperatures
REGIME_LLM_TEMPERATURE = 0.15
REGIME_LLM_MAX_TOKENS = 600

# Scoring weights (LLM scores each factor 1-10, code computes weighted S)
REGIME_WEIGHT_MACRO = 0.35      # α — Macroeconomic Impact
REGIME_WEIGHT_ROTATION = 0.40   # β — Capital Rotation Intensity (highest weight)
REGIME_WEIGHT_EMOTIONAL = 0.25  # γ — Emotional Arbitrage Gap

# Significance threshold: score > N triggers Agent 3 (Portfolio Manager)
REGIME_SIGNIFICANCE_THRESHOLD = 70  # integer 0-100

# Default market state (used as baseline; override with JSON file or env var)
REGIME_DEFAULT_MARKET_STATE = {
    "timestamp": "2026-05-31",
    "macro_baseline": {
        "interest_rate_trend": "Holding",
        "inflation_trend": "Cooling",
        "market_regime": "Low Volatility Expansion",
    },
    "portfolio_allocations": {
        "total_value": 19091.87,
        "cash_reserves_percent": 32.3,
        "sectors": {
            "Technology": {
                "weight_percent": 35.2,
                "sub_sector_bias": [
                    "Semiconductors",
                    "AI Infrastructure",
                    "Communication Services",
                    "Quantum Computing",
                ],
            },
            "Healthcare": {
                "weight_percent": 13.5,
                "sub_sector_bias": ["Telehealth", "Biotechnology"],
            },
            "Broad_Market": {
                "weight_percent": 7.3,
                "sub_sector_bias": ["S&P 500 Index"],
            },
            "Consumer_Discretionary": {
                "weight_percent": 6.3,
                "sub_sector_bias": ["Apparel", "Automotive", "Sports Technology"],
            },
            "Energy": {
                "weight_percent": 4.3,
                "sub_sector_bias": ["Uranium", "Nuclear Power"],
            },
            "Industrials": {
                "weight_percent": 1.1,
                "sub_sector_bias": ["Marine Robotics"],
            },
        },
    },
}

# =============================================================================
# Pipeline Thresholds (used in run_pipeline.py summary reporting)
# =============================================================================
IMPORTANCE_HIGH_THRESHOLD = 0.7      # Articles with score ≥ this are "high importance"
DISPARITY_HIGH_THRESHOLD = 0.35      # Articles with disparity ≥ this are "high disparity"
