# Agentic Workflow

Multi-bucket news ingestion pipeline with Scout enrichment, emotional tonality analysis, related-article clustering, and Regime Analyst for detecting capital rotation and macro regime changes.

**Async concurrency** across all agents (`asyncio.gather` + `asyncio.Semaphore`), **native Gemini structured output** (Pydantic schemas + `response_mime_type="application/json"`), and **exponential-backoff retry** for transient API errors.

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
                         ┌── DIMMEDIATE PERSISTENCE ──┐
                         │   (raw articles saved to     │
                         │    data/news.db BEFORE any    │
                         │    enrichment, ensuring zero  │
                         │    data loss on crash)        │
                         └──────────┬──────────────────┘
                                    ▼
ToneAnalystNode ──> emotional_score + factual_score
                    └──> disparity_score
                              │
                              ▼
ClusterFinder ──> finds related articles for
                    high-disparity pieces (±7 days)
                    Uses keyword + word-level fallback
                              │
                              ▼
     ┌── Backfill enrichment data into DB rows ──┐
     │  (emotional_analysis / cluster metadata    │
     │   written back to existing article rows)   │
     └──────────────────┬────────────────────────┘
                        ▼
                  data/news.db (all articles persisted)
                        │
                        ▼
                RegimeAnalystNode ──> Significance Score (0-100)
                        │              ├── M: Macro Impact (1-10)
                        │              ├── R: Rotation Intensity (1-10)
                        │              └── E: Emotional Arbitrage (1-10)
                        │
              ┌── composite > 70? ─┬── single factor ≥ 7? ──┐
              │ (weighted score)    │ (individual trigger)   │
              └─────────┬───────────┘                       │
                        │  (OR gate — either triggers routing)
                        ▼
              significant_articles    ⚠️ No regime change
              table (full metrics)    (graph ends)
                        │
                         ▼
              Portfolio Manager (Agent 3)
              ├── Step 1: Quant Researcher (DDG search + LLM synthesis)
              │   └── Target_Research_List (HEADLINE / PROXY / COMPETITOR / SUPPLIER)
              └── Step 2: Book Runner (LLM allocation + Python validator)
                  └── PortfolioRecommendation (strict JSON) → Agent 4 (Risk Reviewer)
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
- **Scores importance** (0.0–1.0) via Gemini LLM native structured output (`ImportanceSchema`) or keyword heuristic
- **Generates search queries** optimized for DuckDuckGo
- **Aggregates snippets** from top-N DuckDuckGo results for rich context
- **Async concurrency**: enriches articles with `asyncio.gather` capped by `asyncio.Semaphore(SCOUT_CONCURRENCY_LIMIT)` to avoid overwhelming the Gemini API
- **Retry logic**: exponential-backoff retry for transient errors (429/503/5xx) via `tenacity`
- **Safety-filter guard**: catches Gemini safety-filtered responses (`text=None`) and falls back to heuristics
- **System instruction separated** from user content to prevent instruction-echo responses

### ToneAnalystNode (`src/agents/ToneAnalystNode.py`)
Emotional tonality analysis (Agent 1b) that separates language from substance:
- **Emotional Score**: -1.0 (fear/panic) to +1.0 (euphoria)
- **Factual Score**: 0.0 (pure opinion) to 1.0 (dense data/numbers)
- **Disparity Score**: `|emotional| - factual` — high disparity signals potential market overreaction
- **Tonality Labels**: `alarmist | measured | euphoric | clinical | balanced | sensationalist`
- **Native structured output**: uses `TonalitySchema` Pydantic model with Gemini's `response_schema` to eliminate JSON parse failures
- **Async concurrency**: same pattern as ScoutNode with `TONE_CONCURRENCY_LIMIT` semaphore
- **Retry + safety-filter guard**: identical to ScoutNode

### ClusterFinder (`src/agents/ClusterFinder.py`)
For articles with high emotional disparity (Agent 1c):
- Extracts keywords from factual claims, ticker tags, and title
- Searches the database for related coverage within ±N days
- Runs tonality analysis on found articles to compare emotional reactions
- **Word-level fallback**: when primary keyword search returns 0 results, falls back to ticker symbols + long words for broader matching
- **Async**: `find_clusters` and `_tone_analyst.analyze_batch` are now async

### RegimeAnalystNode (`src/agents/RegimeAnalystNode.py`)
Capital rotation and macro regime detector (Agent 2) — a strict quantitative gatekeeper:
- **Ingests** all articles with their Scout and Tonality analysis, plus the current `MarketState` baseline
- **LLM evaluates** three factors (1-10 each) via **native JSON mode** (Pydantic `RegimeResponse` schema with `response_mime_type="application/json"`)
- **M (Macro Impact)**: Shifts in rates, inflation, growth vs baseline
- **R (Rotation Intensity)**: Evidence of sector/intra-sector capital migration
- **E (Emotional Arbitrage)**: Narrative over/under-reaction using disparity scores
- **Python computes** weighted Significance Score: `S = αM + βR + γE` (where α=0.35, β=0.40, γ=0.25, configurable)
- **Dual gate routing**: Score > 70 (composite) **OR** any single factor ≥ 7.0 (individual trigger) → saves to `significant_articles` table → routes to Portfolio Manager (Agent 3). Only if BOTH gates fail does the graph end.
- **Safety-filter guard**: catches `text=None` responses and falls back to heuristic analysis
- **Retry via `tenacity`**: exponential-backoff for transient failures

### PortfolioManagerNode (`src/agents/PortfolioManagerNode.py`)
The Optimizer & Execution Strategist (Agent 3) — invoked whenever the Regime Analyst's `Significance_Score > 70`. Implemented as **two sequential LLM chains** within a single node, because combining abstract proxy research with portfolio mathematics exceeds the cognitive load of a single LLM call.

#### Step 1 — The Quant Researcher (Abstract Discovery & Proxy Hunting)
Handles "second-order thinking": looks beyond the headline ticker to the supply chain, competitors, and historical precedents.
1. Receives the **Significant_Regime_Payload** from Agent 2 and the **Current_Market_State** JSON.
2. Calls the LLM to formulate **3-4 targeted DuckDuckGo queries** (suppliers, competitors, historical patterns, proxy stocks).
3. Executes those queries via the `ddgs` library and aggregates snippets.
4. **Optionally enriches** every target with live market cap + price from Finnhub (`aiohttp`, parallel fetches via `asyncio.gather`).
5. Calls the LLM a second time to synthesize the snippets into a structured `Target_Research_List`.

Each target is classified as:
- `HEADLINE` — the ticker named in the news
- `PROXY` — a substitute offering cheaper / safer exposure
- `COMPETITOR` — a direct peer
- `SUPPLIER` — an upstream beneficiary in the value chain

Each target also gets a `momentum_flag` indicating whether it's a tactical momentum ride vs. a pure value play, plus per-target `valuation_thesis`, `momentum_thesis`, and `evidence` snippets.

**Output**: `TargetResearchList { targets[], queries_used[], research_summary }`

#### Step 2 — The Book Runner (Allocation & Risk Management)
Takes the `Target_Research_List` and the `Current_Market_State` and decides what actually gets bought, sold, or trimmed.
1. Single LLM call emits the **strict JSON trade plan** with the schema:
   ```json
   {
     "Portfolio_Impact_Assessment": "...",
     "Abstract_Proxy_Discoveries": "...",
     "Momentum_vs_Valuation_Analysis": "...",
     "Proposed_Actions": [
       {
         "Ticker": "AAPL",
         "Action": "EXPAND | DILUTE | ADD",
         "Time_Horizon": "SHORT_TERM_MOMENTUM | LONG_TERM_HOLD",
         "Reasoning": "Highly detailed justification..."
       }
     ]
   }
   ```
2. **Python validator** enforces the schema:
   - `Action` ∈ {`EXPAND`, `DILUTE`, `ADD`} (invalid → dropped)
   - `Time_Horizon` ∈ {`SHORT_TERM_MOMENTUM`, `LONG_TERM_HOLD`} (invalid → `LONG_TERM_HOLD`)
   - Ticker must exist in either the research list or the current portfolio
   - Caps the action list at `PORTFOLIO_MAX_PROPOSED_ACTIONS` (default 8)
   - Deduplicates `(ticker, action)` tuples
   - Pads short reasoning to 30+ words

**Output**: `PortfolioRecommendation` — routes to Agent 4 (The Risk Reviewer) via the `route_after_portfolio_manager` conditional edge.

#### Retry / Resilience
- **Allocator retry loop**: up to 3 retries with jittered backoff for the Book Runner JSON parse — the most critical parse point in the pipeline
- **`json_repair` utility**: uses `parse_json_with_repair` instead of `json.loads` to recover from common LLM JSON errors (missing commas, unterminated strings, trailing commas)
- If all retries are exhausted, falls back to a conservative no-trade recommendation

#### Routing
```python
def route_after_portfolio_manager(state) -> str:
    # returns "risk_reviewer" if recommendation has actions
    # returns "__end__" otherwise (no-trade signal)
```

#### Failure Modes (no LLM / no API)
- **No Gemini key**: falls back to a heuristic Researcher that surfaces only the `ticker_tags` already present in articles, marked as `HEADLINE` role. Allocator emits a conservative no-trade recommendation.
- **No Finnhub key**: skips live enrichment; `market_cap` / `current_price` stay `None` on each target.
- **LLM call failure**: caught and converted to a `_fallback_recommendation` that leaves `Proposed_Actions` empty and routes to `__end__`.

### RiskReviewerNode (`src/agents/RiskReviewerNode.py`)
The Critic (Agent 4) — invoked by `route_after_portfolio_manager` whenever Agent 3 emits a `PortfolioRecommendation` with `Proposed_Actions`. Performs a **Dual-Check Evaluation** with **VETO POWER**:

**1. OPTIMIZATION CHECK**
- Did the PM capitalize on the regime shift via SMART PROXIES (cheaper, purer, less obvious exposure) — or did it just chase the obvious overvalued headline?
- Are the proposed actions internally consistent (no EXPAND + DILUTE on the same ticker; every ADD funded by a DILUTE)?
- Is the funding logic sound? Are the time horizons appropriate for the catalyst?

**2. FLAW DETECTION**
- Scans the original news payload for **contradictory articles** that invalidate the PM's thesis (macro headwinds, rate fears, negative earnings, regulatory action).
- Looks for **liquidity traps** (low-float names, momentum rides that will reverse on a 5% down-day).
- Checks for **concentration risk** (sector overweight above tolerance).
- Verifies TIME_HORIZON matches the catalyst.

**Output schema** (strict):
```json
{
  "optimization_verdict": "1-3 sentence verdict on proxy selection and consistency.",
  "risk_flaw_analysis":  "1-3 sentence identification of missed risks / contradictions / concentration.",
  "approval_status":     true | false,
  "critic_feedback":      "EMPTY if approved; specific, actionable, named-ticker fix instructions if rejected (≥30 chars)."
}
```

**Routing**
```python
def route_after_risk_reviewer(state) -> str:
    # approved                → "output_reporter"  (Agent 5)
    # rejected & iter < MAX   → "portfolio_manager" (back to PM for revise)
    # rejected & iter == MAX  → "__end__"           (hard cap)
```

**Revision loop (Option B — in-place revise)**
When the critic rejects, the `previous_critic_feedback` is piped back to Agent 3 and `PortfolioManagerNode.revise_recommendation(existing, critic_feedback, ...)` is called. This re-prompts the Book Runner with the prior recommendation + the critic's specific feedback (no fresh DDG / Finnhub work — the existing research is reused). The revised recommendation is then re-validated against the same universe rules.

**Hard cap** — `RISK_REVIEW_MAX_ITERATIONS` (default 3). After the critic and PM disagree 3 times in a row, the loop terminates with `__end__` and a warning is logged. The graph can never get stuck in an infinite disagreement loop.

**Failure Modes (no LLM / no API)**
- **No Gemini key**: heuristic auto-approves with an explanatory verdict. The human reviewer at the terminal step remains the real Critic in that scenario.
- **LLM call failure**: caught and converted to a fallback `CriticFeedback(approval_status=True)` so the graph continues.
- **Max iterations hit**: forces `__end__`; Agent 5 (when built) will surface the loop-exhaustion in its report.

## Configuration

All tunable constants live in **[`src/config.py`](src/config.py)** — the single source of truth:

### LLM / Gemini

| Constant | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | `""` (from env) | Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Global fallback model |
| `SCOUT_GEMINI_MODEL` | `gemini-2.5-flash` | Scout importance & query generation |
| `TONALITY_GEMINI_MODEL` | `gemini-2.5-flash` | Tonality analysis |
| `REGIME_GEMINI_MODEL` | `gemini-3.1-pro-preview` | Regime analysis (thinker model) |
| `PORTFOLIO_GEMINI_MODEL` | `gemini-3.1-pro-preview` | Portfolio Manager (super thinker) |
| `RISK_REVIEWER_GEMINI_MODEL` | `gemini-3.1-pro-preview` | Risk Reviewer (critic) |
| `PORTFOLIO_REVISE_GEMINI_MODEL` | `gemini-3.1-pro-preview` | PM revision chain |

### ScoutNode

| Constant | Default | Description |
|----------|---------|-------------|
| `SCOUT_DDG_MAX_RESULTS` | `7` | DuckDuckGo results per query |
| `SCOUT_SUMMARY_MAX_CHARS` | `300` | Fallback chars from content when summary empty |
| `SCOUT_IMPORTANCE_TEMPERATURE` | `0.1` | LLM temperature for scoring |
| `SCOUT_IMPORTANCE_MAX_TOKENS` | `800` | Max response tokens for importance |
| `SCOUT_IMPORTANCE_PROMPT_CHARS` | `1500` | Max summary chars fed to importance prompt |
| `SCOUT_QUERY_TEMPERATURE` | `0.2` | LLM temperature for queries |
| `SCOUT_QUERY_MAX_TOKENS` | `80` | Max response tokens for query generation |
| `SCOUT_CONCURRENCY_LIMIT` | `5` | Max concurrent Gemini/DDG calls |

### ToneAnalystNode

| Constant | Default | Description |
|----------|---------|-------------|
| `DISPARITY_THRESHOLD` | `0.35` | Triggers cluster search |
| `TONE_CONCURRENCY_LIMIT` | `5` | Max concurrent Gemini calls |
| `TONALITY_TEMPERATURE` | `0.15` | LLM temperature for tonality |
| `TONALITY_MAX_TOKENS` | `3000` | Max response tokens |
| `TONALITY_ANALYSIS_MAX_CHARS` | `3000` | Text chars fed to LLM prompt |

### ClusterFinder

| Constant | Default | Description |
|----------|---------|-------------|
| `CLUSTER_DAYS_WINDOW` | `7` | ±N day range for related articles |
| `CLUSTER_MAX_RELATED` | `10` | Max related articles per cluster |
| `CLUSTER_DISPARITY_THRESHOLD` | `0.35` | Same as ToneAnalyst (sync'd) |

### Ingestor Limits

| Constant | Default | Description |
|----------|---------|-------------|
| `WIRE_MAX_ARTICLES` | `30` | Max Finnhub wire headlines |
| `RSS_MAX_PER_FEED` | `20` | Max RSS entries per feed |

### Regime Analyst

| Constant | Default | Description |
|----------|---------|-------------|
| `REGIME_LLM_TEMPERATURE` | `0.15` | LLM temperature for regime analysis |
| `REGIME_LLM_MAX_TOKENS` | `2500` | Max response tokens |
| `REGIME_INDIVIDUAL_TRIGGER_THRESHOLD` | `7.0` | Single factor ≥ this triggers regime flag |
| `REGIME_WEIGHT_MACRO` | `0.35` | α — Macro impact weight |
| `REGIME_WEIGHT_ROTATION` | `0.40` | β — Rotation intensity weight (highest) |
| `REGIME_WEIGHT_EMOTIONAL` | `0.25` | γ — Emotional arbitrage weight |
| `REGIME_SIGNIFICANCE_THRESHOLD` | `70` | Score > this triggers Agent 3 |

### Portfolio Manager

| Constant | Default | Description |
|----------|---------|-------------|
| `PORTFOLIO_RESEARCHER_TEMPERATURE` | `0.4` | Researcher LLM temp (creative proxy hunt) |
| `PORTFOLIO_RESEARCHER_MAX_TOKENS` | `6000` | Researcher max output tokens |
| `PORTFOLIO_ALLOCATOR_TEMPERATURE` | `0.15` | Allocator LLM temp (deterministic math) |
| `PORTFOLIO_ALLOCATOR_MAX_TOKENS` | `6000` | Allocator max output tokens |
| `PORTFOLIO_DDG_MAX_RESULTS` | `5` | DDG snippets per query |
| `PORTFOLIO_DDG_QUERIES` | `4` | Number of search queries the LLM formulates |
| `PORTFOLIO_MAX_PROPOSED_ACTIONS` | `8` | Cap on validated trade list |
| `PORTFOLIO_VALID_ACTIONS` | `{EXPAND, DILUTE, ADD}` | Action vocabulary |
| `PORTFOLIO_VALID_HORIZONS` | `{SHORT_TERM_MOMENTUM, LONG_TERM_HOLD}` | Time-horizon vocabulary |
| `PORTFOLIO_FINNHUB_TIMEOUT` | `10` | Seconds per Finnhub request |

### Risk Reviewer

| Constant | Default | Description |
|----------|---------|-------------|
| `RISK_REVIEWER_TEMPERATURE` | `0.1` | Critic LLM temp (deterministic) |
| `RISK_REVIEWER_MAX_TOKENS` | `3000` | Critic max output tokens |
| `PORTFOLIO_REVISE_TEMPERATURE` | `0.15` | PM-revise LLM temp |
| `PORTFOLIO_REVISE_MAX_TOKENS` | `3500` | PM-revise max output tokens |
| `RISK_REVIEW_MAX_ITERATIONS` | `3` | Hard cap on PM ↔ Critic disagreements |
| `RISK_MAX_SINGLE_POSITION_PCT` | `25.0` | Single position > this → heuristic veto |
| `RISK_MAX_SECTOR_PCT` | `45.0` | Sector post-trade > this → heuristic veto |

### Pipeline Thresholds

| Constant | Default | Description |
|----------|---------|-------------|
| `IMPORTANCE_HIGH_THRESHOLD` | `0.7` | Flags "high importance" in summary |
| `DISPARITY_HIGH_THRESHOLD` | `0.35` | Flags "high emotional disparity" in summary |

### API Keys & Environment Variables

| Variable | Required? | Fallback |
|----------|-----------|----------|
| `GEMINI_API_KEY` | No | Keyword heuristic scoring & tonality |
| `FINNHUB_API_KEY` | No | Default Finnhub key in config |
| `SCOUT_GEMINI_MODEL` | No | Defaults to `gemini-2.5-flash` |
| `TONALITY_GEMINI_MODEL` | No | Defaults to `gemini-2.5-flash` |
| `REGIME_GEMINI_MODEL` | No | Defaults to `gemini-3.1-pro-preview` |
| `PORTFOLIO_GEMINI_MODEL` | No | Defaults to `gemini-3.1-pro-preview` |
| `RISK_REVIEWER_GEMINI_MODEL` | No | Defaults to `gemini-3.1-pro-preview` |

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
├── debug_pipeline.py            # Pipeline debug harness
├── debug_portfolio_manager.py   # Portfolio Manager debug harness
└── src/
    ├── __init__.py
    ├── config.py                # ★ Centralized configuration (all constants)
    ├── NewsArticle.py           # Pydantic validation schema
    ├── state.py                 # Data classes (ScoutArticle, EmotionalAnalysis, RegimeAnalysis, TargetStock, ProposedAction, PortfolioRecommendation, CriticFeedback, GraphState)
    ├── agents/
    │   ├── ScoutNode.py            # Agent 1a: importance scoring + search + aggregation
    │   ├── ToneAnalystNode.py      # Agent 1b: emotional tonality analysis
    │   ├── ClusterFinder.py        # Agent 1c: related-article clustering
    │   ├── RegimeAnalystNode.py    # Agent 2: macro regime & capital rotation detection
    │   ├── PortfolioManagerNode.py # Agent 3: Researcher + Book Runner (target discovery + allocation)
    │   └── RiskReviewerNode.py    # Agent 4: Critic (dual-check + revise loop)
    ├── db/
    │   └── DatabaseSink.py      # SQLite persistence + related-article queries + significant_articles persistence
    ├── ingestors/
    │   ├── WireIngestor.py      # Finnhub API → headlines
    │   ├── MacroBlogs.py        # Crawl4AI macro blog scraping
    │   └── GlobalOutlets.py     # Regional RSS feeds
    └── utils/
        └── json_repair.py       # Robust JSON recovery (falls back from common LLM errors)
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

### `significant_articles` — Articles that triggered a regime change (score > 70 or any single factor ≥ 7)

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

Max possible: 100  |  Min possible: 10
```

### Dual Gate Routing

The Regime Analyst uses an **OR gate** for routing to the Portfolio Manager:

```
proceed = (significance_score > 70) OR (M ≥ 7) OR (R ≥ 7) OR (E ≥ 7)

If true → save to significant_articles → route to Agent 3
If false → graph ends (no regime change detected)
```

The individual trigger threshold (`REGIME_INDIVIDUAL_TRIGGER_THRESHOLD = 7.0`) catches
regime signals where one factor is extreme but the weighted composite hasn't crossed 70.

## Dependencies

- `ddgs` — DuckDuckGo search (snippet aggregation)
- `google-genai` — Gemini LLM (importance, tonality, queries, regime analysis)
- `feedparser` — RSS feed parsing
- `aiohttp` — Async HTTP client (Finnhub API)
- `pydantic` — Data validation + structured output schemas
- `Crawl4AI` — Macro blog content scraping
- `tenacity` — Exponential backoff retry for LLM API calls

Install with: `pip install -r requirements.txt`