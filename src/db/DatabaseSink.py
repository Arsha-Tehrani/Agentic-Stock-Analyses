"""
DatabaseSink – Persists validated NewsArticle objects into an SQLite database.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional
from src.NewsArticle import NewsArticle

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_DIR = os.path.join(_PROJECT_ROOT, "data")
DB_PATH = os.path.join(_DB_DIR, "news.db")
_SQL_PATH = os.path.join(_DB_DIR, "news.sql")


class DatabaseSink:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        with open(_SQL_PATH, "r") as f:
            raw_sql = f.read()

        clean_lines = []
        for line in raw_sql.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("--"):
                clean_lines.append(line)
        clean_sql = "\n".join(clean_lines)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(clean_sql)
            conn.commit()
        finally:
            conn.close()

    def insert_articles(self, articles: List[NewsArticle]) -> int:
        conn = sqlite3.connect(self.db_path)
        inserted = 0
        try:
            for article in articles:
                if self._url_exists(conn, article.url):
                    print(f"  ⏭️  Duplicate skipped: {str(article.url)[:60]}...")
                    continue
                self._insert_one(conn, article)
                inserted += 1
            conn.commit()
        finally:
            conn.close()
        print(f"  💾 {inserted} new article(s) written to {self.db_path}")
        return inserted

    def insert_articles_batch(self, articles: List[NewsArticle]) -> int:
        conn = sqlite3.connect(self.db_path)
        total_inserted = 0
        try:
            new_articles = [a for a in articles if not self._url_exists(conn, a.url)]
            if not new_articles:
                print("  ⏭️  All duplicates — nothing to insert.")
                return 0
            rows = [
                (a.source_bucket, a.source_name, a.title, a.content, a.summary,
                 a.url, a.timestamp.isoformat(), json.dumps(a.ticker_tags), a.importance_score)
                for a in new_articles
            ]
            cursor = conn.executemany(
                "INSERT INTO articles (source_bucket, source_name, title, content, summary, url, timestamp, ticker_tags, importance_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            total_inserted = cursor.rowcount
        finally:
            conn.close()
        print(f"  💾 {total_inserted} new article(s) written (batch) to {self.db_path}")
        return total_inserted

    @staticmethod
    def _url_exists(conn: sqlite3.Connection, url: Optional[str]) -> bool:
        if not url:
            return False
        cursor = conn.execute("SELECT 1 FROM articles WHERE url = ? LIMIT 1", (url,))
        return cursor.fetchone() is not None

    @staticmethod
    def _insert_one(conn: sqlite3.Connection, article: NewsArticle):
        conn.execute(
            "INSERT INTO articles (source_bucket, source_name, title, content, summary, url, timestamp, ticker_tags, importance_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (article.source_bucket, article.source_name, article.title, article.content, article.summary,
             article.url, article.timestamp.isoformat(), json.dumps(article.ticker_tags), article.importance_score),
        )

    def fetch_recent(self, limit: int = 20) -> List[NewsArticle]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM articles ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            articles = []
            for row in rows:
                articles.append(NewsArticle(
                    source_bucket=row["source_bucket"],
                    source_name=row["source_name"],
                    title=row["title"],
                    content=row["content"],
                    summary=row["summary"] if row["summary"] else "",
                    url=row["url"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    ticker_tags=json.loads(row["ticker_tags"]) if row["ticker_tags"] else [],
                    importance_score=row["importance_score"],
                ))
            return articles
        finally:
            conn.close()

    def article_count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        finally:
            conn.close()