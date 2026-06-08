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
                 a.url, a.timestamp.isoformat(), json.dumps(a.ticker_tags), a.importance_score,
                 getattr(a, 'emotional_score', None),
                 getattr(a, 'factual_score', None),
                 getattr(a, 'disparity_score', None),
                 getattr(a, 'tonality_label', None),
                 getattr(a, 'emotional_reasoning', None),
                 getattr(a, 'emotional_phrases', None) if getattr(a, 'emotional_phrases', None) is None else json.dumps(getattr(a, 'emotional_phrases', [])),
                 getattr(a, 'factual_claims', None) if getattr(a, 'factual_claims', None) is None else json.dumps(getattr(a, 'factual_claims', [])),
                 )
                for a in new_articles
            ]
            cursor = conn.executemany(
                "INSERT INTO articles (source_bucket, source_name, title, content, summary, url, timestamp, ticker_tags, importance_score,"
                " emotional_score, factual_score, disparity_score, tonality_label, emotional_reasoning, emotional_phrases, factual_claims)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            "INSERT INTO articles (source_bucket, source_name, title, content, summary, url, timestamp, ticker_tags, importance_score,"
            " emotional_score, factual_score, disparity_score, tonality_label, emotional_reasoning, emotional_phrases, factual_claims)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (article.source_bucket, article.source_name, article.title, article.content, article.summary,
             article.url, article.timestamp.isoformat(), json.dumps(article.ticker_tags), article.importance_score,
             getattr(article, 'emotional_score', None),
             getattr(article, 'factual_score', None),
             getattr(article, 'disparity_score', None),
             getattr(article, 'tonality_label', None),
             getattr(article, 'emotional_reasoning', None),
             getattr(article, 'emotional_phrases', None) if getattr(article, 'emotional_phrases', None) is None else json.dumps(getattr(article, 'emotional_phrases', [])),
             getattr(article, 'factual_claims', None) if getattr(article, 'factual_claims', None) is None else json.dumps(getattr(article, 'factual_claims', [])),
             ),
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

    def find_related_by_keywords(
        self,
        keywords: List[str],
        exclude_url: Optional[str] = None,
        days_window: int = 7,
        limit: int = 10,
    ) -> List[NewsArticle]:
        """
        Find articles whose title or content contains any of the given keywords,
        within a time window (±days_window from now), excluding the given URL.
        """
        if not keywords:
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Build a LIKE-based query for each keyword
            where_clauses = []
            params: List[str] = []
            for kw in keywords:
                like_kw = f"%{kw}%"
                where_clauses.append("(title LIKE ? OR content LIKE ?)")
                params.extend([like_kw, like_kw])

            where_sql = " OR ".join(where_clauses)

            query = (
                f"SELECT * FROM articles WHERE ({where_sql})"
                f" AND timestamp >= datetime('now', '-{days_window} days')"
            )
            query_params = params

            if exclude_url:
                query += " AND url != ?"
                query_params.append(exclude_url)

            query += " ORDER BY timestamp DESC LIMIT ?"
            query_params.append(str(limit))

            rows = conn.execute(query, query_params).fetchall()
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

    def find_articles_with_emotional_analysis(
        self, limit: int = 20
    ) -> List[dict]:
        """Fetch articles that have emotional_analysis data, returning raw rows."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM articles WHERE emotional_score IS NOT NULL "
                "ORDER BY disparity_score DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def insert_significant_articles(
        self,
        articles: List["NewsArticle"],
        signed_articles: List["ScoutArticle"],
        regime_analysis: "RegimeAnalysis",
    ) -> int:
        """
        Insert articles that triggered a regime change into the
        significant_articles table with all Scout, Tonality, and
        Regime metrics for later manual analysis.

        Uses the article_id FK from the main articles table.
        """
        from src.state import ScoutArticle, RegimeAnalysis  # noqa: F811

        conn = sqlite3.connect(self.db_path)
        inserted = 0
        try:
            # Map URLs to their article IDs
            url_to_id = {}
            cursor = conn.execute("SELECT id, url FROM articles WHERE url IS NOT NULL")
            for row in cursor:
                url_to_id[row[1]] = row[0]

            for i, article in enumerate(articles):
                article_id = url_to_id.get(article.url)
                if not article_id:
                    continue

                # Get corresponding ScoutArticle for tonality/cluster data
                sa = signed_articles[i] if i < len(signed_articles) else None

                emotional_score = getattr(article, 'emotional_score', None)
                factual_score = getattr(article, 'factual_score', None)
                disparity_score = getattr(article, 'disparity_score', None)
                tonality_label = getattr(article, 'tonality_label', None)
                emotional_reasoning = getattr(article, 'emotional_reasoning', None)
                emotional_phrases = (
                    json.dumps(getattr(article, 'emotional_phrases', []))
                    if getattr(article, 'emotional_phrases', None) else None
                )
                factual_claims = (
                    json.dumps(getattr(article, 'factual_claims', []))
                    if getattr(article, 'factual_claims', None) else None
                )
                related_count = len(sa.related_articles) if sa else 0

                conn.execute(
                    "INSERT INTO significant_articles ("
                    " article_id, title, source_bucket, source_name, url, timestamp,"
                    " importance_score,"
                    " emotional_score, factual_score, disparity_score,"
                    " tonality_label, emotional_reasoning, emotional_phrases, factual_claims,"
                    " macro_analysis, rotation_analysis, emotional_arbitrage_analysis,"
                    " macro_score, rotation_score, emotional_arbitrage_score,"
                    " significance_score, proceed_to_portfolio_manager,"
                    " related_article_count"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        article_id,
                        article.title,
                        article.source_bucket,
                        article.source_name,
                        article.url,
                        article.timestamp.isoformat(),
                        article.importance_score,
                        emotional_score,
                        factual_score,
                        disparity_score,
                        tonality_label,
                        emotional_reasoning,
                        emotional_phrases,
                        factual_claims,
                        regime_analysis.Macro_Analysis,
                        regime_analysis.Rotation_Analysis,
                        regime_analysis.Emotional_Arbitrage_Analysis,
                        regime_analysis.macro_score,
                        regime_analysis.rotation_score,
                        regime_analysis.emotional_arbitrage_score,
                        regime_analysis.Significance_Score,
                        regime_analysis.proceed_to_portfolio_manager,
                        related_count,
                    ),
                )
                inserted += 1

            conn.commit()
        finally:
            conn.close()

        print(f"  📋 {inserted} article(s) saved to significant_articles table for later analysis")
        return inserted

    def fetch_significant_articles(
        self, limit: int = 50
    ) -> List[dict]:
        """Fetch all significant articles ordered by significance score descending."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM significant_articles ORDER BY significance_score DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
