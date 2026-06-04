# Agentic Workflow

Multi-bucket news ingestion pipeline with event-centric Scout enrichment.

## Directory Structure

```
Agentic_Workflow/
├── .venv/                    # Python virtual environment
├── data/
│   ├── news.db               # SQLite database (auto-created on first run)
│   └── news.sql              # Schema definition
├── requirements.txt          # Dependencies
├── run_pipeline.py           # Main entry point
└── src/
    ├── __init__.py
    ├── NewsArticle.py        # Pydantic validation schema
    ├── state.py              # ScoutArticle dataclass
    ├── db/
    │   ├── __init__.py
    │   └── DatabaseSink.py   # SQLite persistence layer
    ├── ingestors/
    │   ├── __init__.py
    │   ├── WireIngestor.py   # Finnhub API → headlines + summaries
    │   ├── MacroBlogs.py     # Crawl4AI macro blog scraping
    │   └── GlobalOutlets.py  # Regional RSS feeds (Nikkei, Al Jazeera)
    └── agents/
        ├── __init__.py
        ├── ScoutNode.py      # Agent 1: importance scoring + search + aggregation
        └── AnalystNode.py    # Agent 2: (to be built)
```

## Pipeline

```
Finnhub ──> summaries ──┐
Macro_Blogs ──> excerpts ──┼──> ScoutNode ──> importance_score (Gemini / heuristic)
Regional ──> excerpts ──┘                  │
                                            ├──> search_query (Gemini / heuristic)
                                            │
                                            ▼
                                      DuckDuckGo (top-5 results)
                                            │
                                            ▼
                                      aggregated_content (1,000+ chars)
                                            │
                                            ▼
                                      data/news.db ──> Regime Analyst
```

## Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. (Optional) Set Gemini API key for LLM-powered scoring & query generation
export GEMINI_API_KEY="your_key_here"
export GEMINI_MODEL="gemini-2.0-flash"  # optional, default value shown

# 3. Run the pipeline
python3 run_pipeline.py
```

Without a Gemini key, the pipeline uses a keyword-based heuristic for importance scoring and query extraction — still functional, just less intelligent.

## Dependencies

- `ddgs` — DuckDuckGo search (snippet aggregation)
- `google-genai` — Gemini LLM (importance scoring, query generation)
- `feedparser` — RSS feed parsing
- `aiohttp` — Async HTTP client (Finnhub API)
- `pydantic` — Data validation
- `Crawl4AI` — Macro blog content scraping

Install with: `pip install -r requirements.txt`