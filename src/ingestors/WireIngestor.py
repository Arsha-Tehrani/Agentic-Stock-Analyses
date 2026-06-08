"""WireIngestor - Fetches news summaries from Finnhub API."""
import aiohttp
from datetime import datetime

from src.config import WIRE_MAX_ARTICLES


class WireIngestor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = f"https://finnhub.io/api/v1/news?category=general&token={self.api_key}"

    async def fetch_latest_wires(self, max_articles: int | None = None) -> list:
        limit = max_articles if max_articles is not None else WIRE_MAX_ARTICLES
        raw_data = await self._fetch_headlines()
        if not raw_data:
            return []
        return self._normalize(raw_data[:limit])

    async def _fetch_headlines(self) -> list:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        return await response.json()
                    print(f"  ⚠️  Finnhub returned status {response.status}")
                    return []
            except Exception as e:
                print(f"  ⚠️  Finnhub request failed: {e}")
                return []

    def _normalize(self, raw_items: list) -> list:
        normalized = []
        for item in raw_items:
            normalized.append({
                "source_bucket": "Wires",
                "source_name": item.get("source", "Financial Wire"),
                "title": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "content": item.get("summary", ""),
                "url": item.get("url", ""),
                "timestamp": self._parse_timestamp(item.get("datetime")),
            })
        return normalized

    @staticmethod
    def _parse_timestamp(ts: int) -> datetime:
        try:
            return datetime.fromtimestamp(ts)
        except (TypeError, OSError):
            return datetime.now()