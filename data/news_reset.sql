-- news_reset.sql
-- Destructive reset — DROPs all tables and re-creates them.
-- Run this manually when you want to wipe the database:
--   sqlite3 data/news.db < data/news_reset.sql
-- Or from Python:
--   DatabaseSink.reset_schema()

DROP TABLE IF EXISTS portfolio_state_history;
DROP TABLE IF EXISTS portfolio_state;
DROP TABLE IF EXISTS significant_articles;
DROP TABLE IF EXISTS articles;

-- Re-create everything (re-uses the same CREATE statements from news.sql)
.read data/news.sql