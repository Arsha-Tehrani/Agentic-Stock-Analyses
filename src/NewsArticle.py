from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class NewsArticle(BaseModel):
    source_bucket: str = Field(..., description="Wires, Macro_Blogs, or Regional")
    source_name: str
    title: str
    content: str
    summary: str = Field(default="", description="Original snippet/excerpt before Scout enrichment")
    url: Optional[str] = None
    timestamp: datetime
    ticker_tags: List[str] = []
    importance_score: Optional[float] = None  # Populated during synthesis