-- news.sql
-- Schema for news article persistence and downstream filtering.
-- This SQLite database stores articles from all three ingestion buckets:
--   Wires, Macro_Blogs, and Regional.

-- Drop if re-running
DROP TABLE IF EXISTS articles;

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
DROP TABLE IF EXISTS significant_articles;

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
-- INSERT statement (used by DatabaseSink.py via parameterised query)
-- -------------------------------------------------------------------
-- INSERT INTO articles
--     (source_bucket, source_name, title, content, url, timestamp, ticker_tags)
