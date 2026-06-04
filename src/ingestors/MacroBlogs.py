import asyncio
from crawl4ai import AsyncWebCrawler
from datetime import datetime

class MacroBlogIngestor:
    def __init__(self, target_urls: list):
        self.target_urls = target_urls

    async def fetch_blog_posts(self) -> list:
        normalized_articles = []
        async with AsyncWebCrawler() as crawler:
            for url in self.target_urls:
                result = await crawler.arun(url=url)
                if result.success:
                    markdown_content = result.markdown
                    normalized_articles.append({
                        "source_bucket": "Macro_Blogs",
                        "source_name": url.split("//")[-1].split(".")[0],
                        "title": result.metadata.get("title", "Macro Analysis Update"),
                        "summary": markdown_content[:300],
                        "content": markdown_content,
                        "url": url,
                        "timestamp": datetime.now()
                    })
        return normalized_articles