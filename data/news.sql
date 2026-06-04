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
    ingested_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Index for fast bucket-based queries
CREATE INDEX IF NOT EXISTS idx_articles_bucket ON articles(source_bucket);

-- Index for deduplication lookups
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);

-- Index for time-range filtering
CREATE INDEX IF NOT EXISTS idx_articles_timestamp ON articles(timestamp);


-- -------------------------------------------------------------------
-- INSERT statement (used by DatabaseSink.py via parameterised query)
-- -------------------------------------------------------------------
-- INSERT INTO articles
--     (source_bucket, source_name, title, content, url, timestamp, ticker_tags)
-- VALUES (?, ?, ?, ?, ?, ?, ?);