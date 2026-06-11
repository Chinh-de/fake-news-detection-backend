from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class HistoryListItem(BaseModel):
    id: int
    text_snippet: str
    fb_post_id: Optional[str]
    slm_label: int
    slm_confidence: float
    llm_label: int
    is_trained: bool
    created_at: datetime
    xgboost_label: Optional[int] = None
    xgboost_confidence: Optional[float] = None

class HistoryPaginated(BaseModel):
    items: List[HistoryListItem]
    total_records: int
    page: int
    limit: int
    total_pages: int
