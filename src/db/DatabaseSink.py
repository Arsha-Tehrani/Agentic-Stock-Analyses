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
    # Class-level flag so schema is only initialized once per process
    _schema_initialized = False

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        if not DatabaseSink._schema_initialized:
            self._init_schema()
            DatabaseSink._schema_initialized = True

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

    @staticmethod
    def reset_schema(db_path: str = DB_PATH):
        """
        Destructive reset — DROPs all tables and re-creates them.
        Useful for testing. Re-initializes the class-level flag so
        the next DatabaseSink() call re-runs _init_schema.
        """
        reset_sql_path = os.path.join(os.path.dirname(_SQL_PATH), "news_reset.sql")
        if not os.path.exists(reset_sql_path):
            raise FileNotFoundError(
                f"Reset SQL file not found at {reset_sql_path}. "
                "Run 'sqlite3 data/news.db < data/news_reset.sql' manually."
            )
        with open(reset_sql_path, "r") as f:
            raw_sql = f.read()

        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(raw_sql)
            conn.commit()
        finally:
            conn.close()
        DatabaseSink._schema_initialized = False
        print(f"  🔄 Database schema reset at {db_path}")

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

    def is_article_enriched(self, url: Optional[str]) -> bool:
        """
        Check if an article URL already exists in the database with
        emotional analysis data populated. Used to skip re-enrichment.

        Returns True if the article exists AND has emotional_score set.
        """
        if not url:
            return False
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT 1 FROM articles WHERE url = ? AND emotional_score IS NOT NULL LIMIT 1",
                (url,),
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def fetch_enriched_article(self, url: str) -> Optional[dict]:
        """
        Load a fully enriched article from the database by URL.
        Returns a dict with all columns, or None if not found.
        The dict can be used to reconstruct a ScoutArticle.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM articles WHERE url = ? LIMIT 1",
                (url,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

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
        Uses URL-based lookup for both the article ID and the
        ScoutArticle (instead of fragile index-based mapping).
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

            # Build URL-to-ScoutArticle lookup (instead of index-based mapping)
            url_to_scout: dict = {}
            for sa in signed_articles:
                if sa.url:
                    url_to_scout[sa.url] = sa

            for i, article in enumerate(articles):
                article_id = url_to_id.get(article.url)
                if not article_id:
                    print(f"    ⚠️  No article ID found for URL: {str(article.url)[:70]}... skipping")
                    continue

                # Use URL-based lookup for ScoutArticle (fixes index mismatch bugs)
                sa = url_to_scout.get(article.url)
                if not sa:
                    print(f"    ⚠️  No ScoutArticle found for URL: {str(article.url)[:70]}... skipping")
                    continue

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

    def update_article_enrichment(
        self, signed_articles: List["ScoutArticle"]
    ) -> int:
        """
        Backfill emotional analysis data into the articles table after
        ToneAnalyst and ClusterFinder have completed.

        Matches rows by URL and UPDATEs emotional_score, factual_score,
        disparity_score, tonality_label, emotional_reasoning,
        emotional_phrases, and factual_claims.

        Args:
            signed_articles: List of ScoutArticle objects with emotional_analysis set.

        Returns:
            Number of rows updated.
        """
        conn = sqlite3.connect(self.db_path)
        updated = 0
        try:
            for sa in signed_articles:
                if not sa.url or not sa.emotional_analysis:
                    continue
                ea = sa.emotional_analysis
                cursor = conn.execute(
                    "UPDATE articles SET "
                    "  emotional_score = ?, factual_score = ?, disparity_score = ?,"
                    "  tonality_label = ?, emotional_reasoning = ?,"
                    "  emotional_phrases = ?, factual_claims = ?"
                    " WHERE url = ?",
                    (
                        ea.emotional_score,
                        ea.factual_score,
                        ea.disparity_score,
                        ea.tonality_label,
                        ea.reasoning,
                        json.dumps(ea.key_emotional_phrases) if ea.key_emotional_phrases else None,
                        json.dumps(ea.key_factual_claims) if ea.key_factual_claims else None,
                        sa.url,
                    ),
                )
                updated += cursor.rowcount
            conn.commit()
        finally:
            conn.close()
        print(f"  💾 {updated} article enrichment(s) updated in {self.db_path}")
        return updated

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

    # ========================================================================
    # Portfolio State Persistence
    # ========================================================================

    def save_portfolio_state(
        self,
        state_dict: dict,
        updated_by: str,
        reason: str = "",
    ) -> int:
        """
        Save the current portfolio state to the database.
        This updates the singleton row in portfolio_state and archives
        the previous state to portfolio_state_history.

        Args:
            state_dict: Dictionary representation of PortfolioState
            updated_by: Who made the change ('human', 'portfolio_manager', 'reviewer', 'system')
            reason: Optional reason for the change (for audit trail)

        Returns:
            Number of rows affected (should be 1)
        """
        import json
        from datetime import datetime

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Get current version for history archival
            current = conn.execute(
                "SELECT id, timestamp, macro_baseline, portfolio_allocations, updated_by, version "
                "FROM portfolio_state WHERE id = 1"
            ).fetchone()

            if current:
                # Archive current state to history before updating
                conn.execute(
                    "INSERT INTO portfolio_state_history "
                    "(portfolio_state_id, timestamp, macro_baseline, portfolio_allocations, updated_by, reason) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        1,
                        current["timestamp"],
                        current["macro_baseline"],
                        current["portfolio_allocations"],
                        current["updated_by"],
                        reason or "Scheduled update",
                    ),
                )
                new_version = current["version"] + 1
            else:
                new_version = 1

            # Update the singleton portfolio_state row
            now = datetime.now().isoformat()
            conn.execute(
                "INSERT OR REPLACE INTO portfolio_state "
                "(id, timestamp, macro_baseline, portfolio_allocations, updated_by, version) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    1,
                    state_dict["timestamp"],
                    json.dumps(state_dict["macro_baseline"]),
                    json.dumps(state_dict["portfolio_allocations"]),
                    updated_by,
                    new_version,
                ),
            )
            conn.commit()
            print(f"  💾 Portfolio state saved (v{new_version}) by {updated_by}")
            return new_version
        finally:
            conn.close()

    def load_portfolio_state(self) -> Optional[dict]:
        """
        Load the current portfolio state from the database.

        Returns:
            Dictionary with portfolio state data, or None if not found.
            The dict has keys: timestamp, macro_baseline, portfolio_allocations, version, updated_by
        """
        import json

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM portfolio_state WHERE id = 1"
            ).fetchone()

            if not row:
                return None

            return {
                "timestamp": row["timestamp"],
                "macro_baseline": json.loads(row["macro_baseline"]),
                "portfolio_allocations": json.loads(row["portfolio_allocations"]),
                "version": row["version"],
                "updated_by": row["updated_by"],
            }
        finally:
            conn.close()

    def get_portfolio_state_history(
        self, limit: int = 20
    ) -> List[dict]:
        """
        Fetch portfolio state history for audit purposes.

        Args:
            limit: Maximum number of historical records to return

        Returns:
            List of historical portfolio state records, ordered by most recent first
        """
        import json

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM portfolio_state_history "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "portfolio_state_id": row["portfolio_state_id"],
                    "timestamp": row["timestamp"],
                    "macro_baseline": json.loads(row["macro_baseline"]),
                    "portfolio_allocations": json.loads(row["portfolio_allocations"]),
                    "updated_by": row["updated_by"],
                    "reason": row["reason"],
                    "created_at": row["created_at"],
                })
            return results
        finally:
            conn.close()

    def initialize_portfolio_state(self, default_state: dict) -> bool:
        """
        Initialize the portfolio state with a default if it doesn't exist.
        This is useful for first-time setup.

        Args:
            default_state: Default portfolio state dictionary from config

        Returns:
            True if initialized, False if already existed
        """
        existing = self.load_portfolio_state()
        if existing:
            return False

        from datetime import datetime
        import json

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio_state "
                "(id, timestamp, macro_baseline, portfolio_allocations, updated_by, version) "
                "VALUES (1, ?, ?, ?, ?, ?)",
                (
                    default_state["timestamp"],
                    json.dumps(default_state["macro_baseline"]),
                    json.dumps(default_state["portfolio_allocations"]),
                    "system",
                    1,
                ),
            )
            conn.commit()
            print(f"  📦 Portfolio state initialized from config defaults")
            return True
        finally:
            conn.close()
