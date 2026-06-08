import feedparser
import asyncio
from datetime import datetime
import time

from src.config import RSS_MAX_PER_FEED

class RegionalRSSIngestor:
    def __init__(self, rss_feeds: dict):
        self.rss_feeds = rss_feeds

    async def fetch_feed(self, source_name: str, url: str) -> list:
        loop = asyncio.get_running_loop()
        feed = await loop.run_in_executor(None, feedparser.parse, url)
        articles = []
        for entry in feed.entries[:RSS_MAX_PER_FEED]:
            ts = datetime.now()
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                ts = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            summary = entry.get("summary", "") or entry.get("description", "")
            articles.append({
                "source_bucket": "Regional",
                "source_name": source_name,
                "title": entry.get("title", ""),
                "summary": summary,
                "content": summary,
                "url": entry.get("link", ""),
                "timestamp": ts
            })
        return articles

    async def fetch_all_regional(self) -> list:
        tasks = [self.fetch_feed(name, url) for name, url in self.rss_feeds.items()]
        results = await asyncio.gather(*tasks)
        return [article for sublist in results for article in sublist]