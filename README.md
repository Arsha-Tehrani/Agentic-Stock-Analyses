# Agentic Workflow

Multi-bucket news ingestion pipeline with Scout enrichment, emotional tonality analysis, related-article clustering, and Regime Analyst for detecting capital rotation and macro regime changes.

## Pipeline Overview

```
Finnhub ──> Wires ──────────┐
Macro_Blogs ──> excerpts ───┼──> ScoutNode ──> importance_score + search_query
Regional ──> RSS ───────────┘       │
                                    ▼
                              DuckDuckGo (top-7 results)
                                    │
                                    ▼
                              aggregated_content (1,500+ chars)
                                    │
                                    ▼
                              ToneAnalystNode ──> emotional_score + factual_score
                                    │              └──> disparity_score
                                    ▼
                              ClusterFinder ──> finds related articles for
                                    │            high-disparity pieces (±7 days)
                                    ▼
                              data/news.db (all articles persisted)
                                    │
                                    ▼
                              RegimeAnalystNode ──> Significance Score (0-100)
                                    │              ├── M: Macro Impact (1-10)
                                    │              ├── R: Rotation Intensity (1-10)
                                    │              └── E: Emotional Arbitrage (1-10)
                                    ▼
                        ┌─── Score > 70? ───┐
                        │                   │
                        ▼                   ▼
              significant_articles    ✅ No regime change
              table (full metrics)    (graph ends)
                        │
                        ▼
              Portfolio Manager (Agent 3 — coming soon)
```

## Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Set API keys (optional — heuristic fallbacks work without them)
export GEMINI_API_KEY="your_gemini_key_here"    # LLM-powered scoring & tonality
export FINNHUB_API_KEY="your_finnhub_key_here"  # Wire/news feed

# 3. Run the pipeline
python3 run_pipeline.py
```

Without API keys, the pipeline uses keyword-based heuristics — fully functional, just less nuanced than LLM-driven analysis.

## Agents

### ScoutNode (`src/agents/ScoutNode.py`)
First-stage enrichment engine (Agent 1) that:
- **Scores importance** (0.0–1.0) via Gemini LLM or keyword heuristic
- **Generates search queries** optimized for DuckDuckGo
- **Aggregates snippets** from top-N DuckDuckGo results for rich context

### ToneAnalystNode (`src/agents/ToneAnalystNode.py`)
Emotional tonality analysis (Agent 1b) that separates language from substance:
- **Emotional Score**: -1.0 (fear/panic) to +1.0 (euphoria)
- **Factual Score**: 0.0 (pure opinion) to 1.0 (dense data/numbers)
- **Disparity Score**: `|emotional| - factual` — high disparity signals potential market overreaction
- **Tonality Labels**: `alarmist | measured | euphoric | clinical | balanced | sensationalist`

### ClusterFinder (`src/agents/ClusterFinder.py`)
For articles with high emotional disparity (Agent 1c):
- Extracts keywords from factual claims, ticker tags, and title
- Searches the database for related coverage within ±N days
- Runs tonality analysis on found articles to compare emotional reactions

### RegimeAnalystNode (`src/agents/RegimeAnalystNode.py`)
Capital rotation and macro regime detector (Agent 2) — a strict quantitative gatekeeper:
- **Ingests** all articles with their Scout and Tonality analysis, plus the current `MarketState` baseline
- **LLM evaluates** three factors (1-10 each):
  - **M (Macro Impact)**: Shifts in rates, inflation, growth vs baseline
  - **R (Rotation Intensity)**: Evidence of sector/intra-sector capital migration
  - **E (Emotional Arbitrage)**: Narrative over/under-reaction using disparity scores
- **Python computes** weighted Significance Score: `S = αM + βR + γE` (where α=0.35, β=0.40, γ=0.25, configurable)
- **Conditional routing**: Score > 70 → saves to `significant_articles` table → routes to Portfolio Manager (Agent 3). Score ≤ 70 → graph ends.

## Configuration

All tunable constants live in **[`src/config.py`](src/config.py)** — the single source of truth:

| Category | Constant | Default | Description |
|----------|----------|---------|-------------|
| **LLM** | `GEMINI_API_KEY` | `""` (from env) | Gemini API key |
| | `GEMINI_MODEL` | `gemini-2.0-flash` | Model used for all agents |
| **ScoutNode** | `SCOUT_DDG_MAX_RESULTS` | `7` | DuckDuckGo results per query |
| | `SCOUT_IMPORTANCE_TEMPERATURE` | `0.1` | LLM temperature for scoring |
| | `SCOUT_QUERY_TEMPERATURE` | `0.2` | LLM temperature for queries |
| **ToneAnalystNode** | `DISPARITY_THRESHOLD` | `0.35` | Triggers cluster search |
| | `TONALITY_TEMPERATURE` | `0.15` | LLM temperature for tonality |
| | `TONALITY_MAX_TOKENS` | `400` | Max response tokens |
| | `TONALITY_ANALYSIS_MAX_CHARS` | `3000` | Text chars fed to LLM prompt |
| **ClusterFinder** | `CLUSTER_DAYS_WINDOW` | `7` | ±N day range for related articles |
| | `CLUSTER_MAX_RELATED` | `10` | Max related articles per cluster |
| **Ingestor Limits** | `WIRE_MAX_ARTICLES` | `20` | Max Finnhub wire headlines |
| | `RSS_MAX_PER_FEED` | `10` | Max RSS entries per feed |
| **Regime Analyst** | `REGIME_WEIGHT_MACRO` | `0.35` | α — Macro impact weight |
| | `REGIME_WEIGHT_ROTATION` | `0.40` | β — Rotation intensity weight (highest) |
| | `REGIME_WEIGHT_EMOTIONAL` | `0.25` | γ — Emotional arbitrage weight |
| | `REGIME_SIGNIFICANCE_THRESHOLD` | `70` | Score > this triggers Agent 3 |
| | `REGIME_LLM_TEMPERATURE` | `0.15` | LLM temperature for regime analysis |
| | `REGIME_LLM_MAX_TOKENS` | `600` | Max response tokens |
| **Pipeline** | `IMPORTANCE_HIGH_THRESHOLD` | `0.7` | Flags "high importance" |
| | `DISPARITY_HIGH_THRESHOLD` | `0.35` | Flags "high emotional disparity" |

### API Keys & Environment Variables

| Variable | Required? | Fallback |
|----------|-----------|----------|
| `GEMINI_API_KEY` | No | Keyword heuristic scoring & tonality |
| `GEMINI_MODEL` | No | Defaults to `gemini-2.0-flash` |
| `FINNHUB_API_KEY` | No | Default Finnhub key in config |

Set them before running:
```bash
export GEMINI_API_KEY="your_key"
export FINNHUB_API_KEY="your_key"
python3 run_pipeline.py
```

### Ingestion Sources

Configured in `src/config.py`:

```python
BLOG_TARGETS = [
    "https://www.bespokepremium.com/interactive/blog/",
    "https://macrocompass.substack.com/",
]
REGIONAL_FEEDS = {
    "Nikkei Asia": "https://services.nikkei.com/core/v1/rss/asia/news.xml",
    "Al Jazeera":   "https://www.aljazeera.com/xml/rss/all.xml",
}
```

Add or remove sources by editing these lists — the pipeline picks them up automatically.

## Directory Structure

```
Agentic_Workflow/
├── .venv/                       # Python virtual environment
├── data/
│   ├── news.db                  # SQLite database
│   └── news.sql                 # Schema definition
├── requirements.txt             # Dependencies
├── run_pipeline.py              # Main entry point
└── src/
    ├── __init__.py
    ├── config.py                # ★ Centralized configuration (all constants)
    ├── NewsArticle.py           # Pydantic validation schema
    ├── state.py                 # Data classes (ScoutArticle, EmotionalAnalysis, RegimeAnalysis, GraphState)
    ├── agents/
    │   ├── ScoutNode.py         # Agent 1a: importance scoring + search + aggregation
    │   ├── ToneAnalystNode.py   # Agent 1b: emotional tonality analysis
    │   ├── ClusterFinder.py     # Agent 1c: related-article clustering
    │   └── RegimeAnalystNode.py # Agent 2: macro regime & capital rotation detection
    ├── db/
    │   └── DatabaseSink.py      # SQLite persistence + related-article queries + significant_articles persistence
    └── ingestors/
        ├── WireIngestor.py      # Finnhub API → headlines
        ├── MacroBlogs.py        # Crawl4AI macro blog scraping
        └── GlobalOutlets.py     # Regional RSS feeds
```

## Database Schema

### `articles` — All ingested news

| Column | Type | Description |
|--------|------|-------------|
| `importance_score` | REAL | Scout importance 0.0–1.0 |
| `emotional_score` | REAL | Tonality emotional -1.0 to +1.0 |
| `factual_score` | REAL | Tonality factual density 0.0–1.0 |
| `disparity_score` | REAL | `\|emotional\| - factual` |
| `tonality_label` | TEXT | alarmist/measured/euphoric/clinical/balanced/sensationalist |
| `emotional_reasoning` | TEXT | LLM explanation of disparity |
| `emotional_phrases` | TEXT | JSON array of emotional phrases |
| `factual_claims` | TEXT | JSON array of factual claims |

### `significant_articles` — Articles that triggered a regime change (score > 70)

| Column | Type | Description |
|--------|------|-------------|
| `article_id` | INTEGER | FK to `articles.id` |
| `importance_score` | REAL | Scout importance score |
| `emotional_score` | REAL | Tonality emotional score |
| `factual_score` | REAL | Tonality factual density |
| `disparity_score` | REAL | Emotional-factual gap |
| `tonality_label` | TEXT | Tonality classification |
| `emotional_reasoning` | TEXT | LLM explanation |
| `emotional_phrases` | TEXT | JSON array |
| `factual_claims` | TEXT | JSON array |
| `macro_analysis` | TEXT | Regime Analyst M-factor explanation |
| `rotation_analysis` | TEXT | Regime Analyst R-factor explanation |
| `emotional_arbitrage_analysis` | TEXT | Regime Analyst E-factor explanation |
| `macro_score` | INTEGER | 1-10 macro impact score |
| `rotation_score` | INTEGER | 1-10 rotation intensity score |
| `emotional_arbitrage_score` | INTEGER | 1-10 arbitrage gap score |
| `significance_score` | INTEGER | 0-100 weighted composite |
| `proceed_to_portfolio_manager` | BOOLEAN | True = route to Agent 3 |
| `related_article_count` | INTEGER | Number of related articles in cluster |
| `analyzed_at` | TEXT | Timestamp of regime analysis |

Query for significant articles ranked by score:
```sql
SELECT title, source_name, tonality_label,
       macro_score, rotation_score, emotional_arbitrage_score,
       significance_score
FROM significant_articles
ORDER BY significance_score DESC;
```

Query for high-disparity articles:
```sql
SELECT title, emotional_score, factual_score, disparity_score, tonality_label
FROM articles
WHERE disparity_score >= 0.35
ORDER BY disparity_score DESC;
```

## Scoring Formula (Regime Analyst)

```
S = α × (M/10 × 100) + β × (R/10 × 100) + γ × (E/10 × 100)

Where:
  M, R, E = LLM scores (1-10 each)
  α = 0.35 (Macro weight)
  β = 0.40 (Rotation weight — highest, capital rotation is key)
  γ = 0.25 (Emotional Arbitrage weight)

Max possible: 100  |  Min possible: 10  |  Threshold for Agent 3: > 70
```

## Dependencies

- `ddgs` — DuckDuckGo search (snippet aggregation)
- `google-genai` — Gemini LLM (importance, tonality, queries, regime analysis)
- `feedparser` — RSS feed parsing
- `aiohttp` — Async HTTP client (Finnhub API)
- `pydantic` — Data validation
- `Crawl4AI` — Macro blog content scraping

Install with: `pip install -r requirements.txt`