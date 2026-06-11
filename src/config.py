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
# LLM / Gemini Configuration
# =============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")  # Global fallback

# Per-agent model selection (explicit environment variable names)
SCOUT_GEMINI_MODEL = os.environ.get("SCOUT_GEMINI_MODEL", "gemini-2.5-flash-lite") #Doer
TONALITY_GEMINI_MODEL = os.environ.get("TONALITY_GEMINI_MODEL", "gemini-2.5-flash") # Slightly thinker model or maybe just light one
REGIME_GEMINI_MODEL = os.environ.get("REGIME_GEMINI_MODEL", "gemini-3.1-pro-preview") #Thinker model
PORTFOLIO_GEMINI_MODEL = os.environ.get("PORTFOLIO_GEMINI_MODEL", "gemini-3.1-pro-preview") #Super Thinker model
RISK_REVIEWER_GEMINI_MODEL = os.environ.get("RISK_REVIEWER_GEMINI_MODEL", "gemini-3.1-pro-preview") #Critic model
PORTFOLIO_REVISE_GEMINI_MODEL = os.environ.get("PORTFOLIO_REVISE_GEMINI_MODEL", "gemini-3.1-pro-preview") #Revise model

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
REGIME_LLM_TEMPERATURE = 0.15 #How creative or "Conservative" the model is. 0-1. Higher is less conservative
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
        "cash_holdings": [
            {"ticker": "SPAXX", "concentration_percent": 32.30}
        ],
        "sectors": {
            "Technology": {
                "weight_percent": 35.2,
                "sub_sector_bias": [
                    "Semiconductors",
                    "AI Infrastructure",
                    "Communication Services",
                    "Quantum Computing",
                ],
                "holdings": [
                    {"ticker": "NVDL", "concentration_percent": 6.99},
                    {"ticker": "LITE", "concentration_percent": 6.72},
                    {"ticker": "GOOGL", "concentration_percent": 3.98},
                    {"ticker": "META", "concentration_percent": 3.31},
                    {"ticker": "QBTS", "concentration_percent": 2.53},
                    {"ticker": "NBIS", "concentration_percent": 2.42},
                    {"ticker": "MSFT", "concentration_percent": 2.36},
                    {"ticker": "ARTY", "concentration_percent": 1.96},
                    {"ticker": "STM", "concentration_percent": 1.45},
                    {"ticker": "ABTC", "concentration_percent": 1.18},
                    {"ticker": "AXTI", "concentration_percent": 1.08},
                    {"ticker": "NVTS", "concentration_percent": 0.70},
                    {"ticker": "SMCI", "concentration_percent": 0.48}
                ]
            },
            "Healthcare": {
                "weight_percent": 13.5,
                "sub_sector_bias": ["Telehealth", "Biotechnology"],
                "holdings": [
                    {"ticker": "HIMS", "concentration_percent": 11.64},
                    {"ticker": "MGNX", "concentration_percent": 1.53},
                    {"ticker": "HIMZ", "concentration_percent": 0.32}
                ]
            },
            "Broad_Market": {
                "weight_percent": 7.3,
                "sub_sector_bias": ["S&P 500 Index"],
                "holdings": [
                    {"ticker": "VOO", "concentration_percent": 7.32}
                ]
            },
            "Consumer_Discretionary": {
                "weight_percent": 6.3,
                "sub_sector_bias": ["Apparel", "Automotive", "Sports Technology"],
                "holdings": [
                    {"ticker": "SRAD", "concentration_percent": 3.39},
                    {"ticker": "NKE", "concentration_percent": 1.69},
                    {"ticker": "TSLA", "concentration_percent": 1.14},
                    {"ticker": "TSLQ", "concentration_percent": 0.09}
                ]
            },
            "Energy": {
                "weight_percent": 4.3,
                "sub_sector_bias": ["Uranium", "Nuclear Power"],
                "holdings": [
                    {"ticker": "UUUU", "concentration_percent": 2.96},
                    {"ticker": "SMR", "concentration_percent": 1.33}
                ]
            },
            "Industrials": {
                "weight_percent": 1.1,
                "sub_sector_bias": ["Marine Robotics"],
                "holdings": [
                    {"ticker": "KRKNF", "concentration_percent": 1.13}
                ]
            },
        },
    },
}

# =============================================================================
# Pipeline Thresholds (used in run_pipeline.py summary reporting)
# =============================================================================
IMPORTANCE_HIGH_THRESHOLD = 0.7      # Articles with score ≥ this are "high importance"
DISPARITY_HIGH_THRESHOLD = 0.35      # Articles with disparity ≥ this are "high disparity"

# =============================================================================
# Portfolio Manager – Agent 3 (Optimizer & Execution Strategist)
# =============================================================================
# Researcher chain: needs creative proxy discovery → higher temperature.
# Allocator chain:  needs deterministic portfolio math → lower temperature.
PORTFOLIO_RESEARCHER_TEMPERATURE = 0.4
PORTFOLIO_RESEARCHER_MAX_TOKENS = 600
PORTFOLIO_ALLOCATOR_TEMPERATURE = 0.15
PORTFOLIO_ALLOCATOR_MAX_TOKENS = 1000

# DuckDuckGo (ddgs) execution parameters for the Researcher
PORTFOLIO_DDG_MAX_RESULTS = 5        # Snippets per query
PORTFOLIO_DDG_QUERIES = 4            # Number of web queries the LLM is asked to formulate

# Action-validity constraints enforced by the Allocator validator
PORTFOLIO_MAX_PROPOSED_ACTIONS = 8   # Cap to prevent runaway trade lists
PORTFOLIO_VALID_ACTIONS = {"EXPAND", "DILUTE", "ADD"}
PORTFOLIO_VALID_HORIZONS = {"SHORT_TERM_MOMENTUM", "LONG_TERM_HOLD"}

# Finnhub enrichment base URL (reuses WIRE_API_KEY above)
PORTFOLIO_FINNHUB_BASE = "https://finnhub.io/api/v1"
PORTFOLIO_FINNHUB_TIMEOUT = 10       # seconds per request

# =============================================================================
# Risk Reviewer – Agent 4 (The Critic / Quantitative Gatekeeper)
# =============================================================================
# Critic: low temperature, deterministic veto power. Be ruthless.
RISK_REVIEWER_TEMPERATURE = 0.1
RISK_REVIEWER_MAX_TOKENS = 800

# PM revise chain (Option B): in-place revision of the existing recommendation
# given the critic's feedback. Low temperature too — apply the suggested fix.
PORTFOLIO_REVISE_TEMPERATURE = 0.15
PORTFOLIO_REVISE_MAX_TOKENS = 1000

# Hard cap on how many times Agent 3 ↔ Agent 4 can disagree before the loop
# terminates (and the graph ends with the most recent feedback attached).
RISK_REVIEW_MAX_ITERATIONS = 3

# Hard-rule thresholds the heuristic critic (no-LLM path) uses to auto-veto
RISK_MAX_SINGLE_POSITION_PCT = 25.0   # any single position > this → veto
RISK_MAX_SECTOR_PCT = 45.0            # any sector post-trade > this → veto
