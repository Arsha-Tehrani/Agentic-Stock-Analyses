-- news.sql
-- Schema for news article persistence and downstream filtering.
-- This SQLite database stores articles from all three ingestion buckets:
--   Wires, Macro_Blogs, and Regional.
--
-- NOTE: DROP TABLE statements have been moved to news_reset.sql.
-- This file only runs CREATE TABLE IF NOT EXISTS so data persists
-- across pipeline runs. To wipe and re-initialize, run news_reset.sql.

CREATE TABLE IF NOT EXISTS articles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_bucket   TEXT    NOT NULL CHECK(source_bucket IN ('Wires', 'Macro_Blogs', 'Regional')),
    source_name     TEXT    NOT NULL,
    title           TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    summary         TEXT    DEFAULT '', -- Original snippet before Scout enrichment
    url             TEXT,
    timestamp       TEXT    NOT NULL,  -- ISO-8601 datetime string
    ticker_tags     TEXT    DEFAULT '', -- JSON array stored as text, e.g. '["AAPL","TSLA"]'
    importance_score REAL   DEFAULT NULL,
    emotional_score REAL   DEFAULT NULL,
    factual_score REAL     DEFAULT NULL,
    disparity_score REAL   DEFAULT NULL,
    tonality_label TEXT    DEFAULT NULL,
    emotional_reasoning TEXT DEFAULT NULL,
    emotional_phrases TEXT DEFAULT NULL,
    factual_claims TEXT    DEFAULT NULL,
    related_article_ids TEXT DEFAULT NULL,
    ingested_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Index for fast bucket-based queries
CREATE INDEX IF NOT EXISTS idx_articles_bucket ON articles(source_bucket);

-- Index for deduplication lookups
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);

-- Index for time-range filtering
CREATE INDEX IF NOT EXISTS idx_articles_timestamp ON articles(timestamp);


-- ---------------------------------------------------------------
-- Significant Articles — articles that triggered a regime change
-- (Regime Analyst significance score > threshold)
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS significant_articles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id      INTEGER NOT NULL,             -- FK to articles.id
    title           TEXT    NOT NULL,
    source_bucket   TEXT    NOT NULL,
    source_name     TEXT    NOT NULL,
    url             TEXT,
    timestamp       TEXT    NOT NULL,

    -- Scout metrics
    importance_score     REAL,

    -- Tonality metrics
    emotional_score      REAL,
    factual_score        REAL,
    disparity_score      REAL,
    tonality_label       TEXT,
    emotional_reasoning  TEXT,
    emotional_phrases    TEXT,   -- JSON array
    factual_claims       TEXT,   -- JSON array

    -- Regime Analyst metrics
    macro_analysis                TEXT,
    rotation_analysis             TEXT,
    emotional_arbitrage_analysis  TEXT,
    macro_score                   INTEGER,
    rotation_score                INTEGER,
    emotional_arbitrage_score     INTEGER,
    significance_score            INTEGER,
    proceed_to_portfolio_manager  BOOLEAN,

    -- Cluster info (related articles found)
    related_article_count  INTEGER DEFAULT 0,

    -- Metadata
    analyzed_at        TEXT    NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_significant_score ON significant_articles(significance_score DESC);
CREATE INDEX IF NOT EXISTS idx_significant_timestamp ON significant_articles(timestamp DESC);


-- -------------------------------------------------------------------
-- Portfolio State — persistent portfolio allocation state
-- This table stores the current portfolio state that can be updated
-- by the Portfolio Manager → Reviewer → Human approval workflow.
-- It's a singleton table (only one row with id=1).
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_state (
    id                      INTEGER PRIMARY KEY CHECK (id = 1),  -- Singleton row
    timestamp               TEXT    NOT NULL,                     -- ISO-8601 datetime of state
    macro_baseline          TEXT    NOT NULL,                     -- JSON object
    portfolio_allocations   TEXT    NOT NULL,                     -- JSON object (full structure)
    updated_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_by              TEXT    NOT NULL CHECK (updated_by IN ('human', 'portfolio_manager', 'reviewer', 'system', 'slack_gateway', 'slack_debug')),
    version                 INTEGER NOT NULL DEFAULT 1            -- Version number for tracking changes
);

CREATE INDEX IF NOT EXISTS idx_portfolio_updated_at ON portfolio_state(updated_at DESC);


-- -------------------------------------------------------------------
-- Portfolio State History — tracks all changes to portfolio state
-- for audit and rollback purposes
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_state_history (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_state_id      INTEGER NOT NULL,                       -- FK to portfolio_state.id
    timestamp               TEXT    NOT NULL,                       -- ISO-8601 datetime of state snapshot
    macro_baseline          TEXT    NOT NULL,                       -- JSON object
    portfolio_allocations   TEXT    NOT NULL,                       -- JSON object (full structure)
    updated_by              TEXT    NOT NULL,                       -- Who made the change
    reason                  TEXT    DEFAULT '',                     -- Reason for change (e.g., regime change detected)
    created_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    
    FOREIGN KEY (portfolio_state_id) REFERENCES portfolio_state(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_portfolio_history_created_at ON portfolio_state_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_history_state_id ON portfolio_state_history(portfolio_state_id);


-- -------------------------------------------------------------------
-- Pending User Theses — user-submitted research theses from Slack
-- The pipeline reads-and-deletes from this staging table atomically
-- so no thesis is processed twice.
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pending_user_theses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    core_argument   TEXT    NOT NULL,
    time_horizon    TEXT    NOT NULL CHECK(time_horizon IN ('SHORT_TERM_MOMENTUM', 'LONG_TERM_HOLD')),
    raw_message     TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    ingested_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pending_theses_timestamp ON pending_user_theses(timestamp DESC);
