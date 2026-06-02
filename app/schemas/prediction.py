from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class PredictRequest(BaseModel):
    text: str = Field(..., description="Nội dung bài viết cần dự đoán")
    fb_post_id: Optional[str] = Field(None, description="ID bài viết trên Facebook (nếu có)")
    fb_post_created_at: Optional[datetime] = Field(None, description="Thời gian đăng bài viết (nếu có)")

class PredictResponse(BaseModel):
    record_id: int = Field(..., description="ID bản ghi trong hệ thống")
    fb_post_id: Optional[str] = Field(None, description="ID bài viết trên Facebook")
    slm_label: int = Field(..., description="Dự đoán của SLM: 0=Thật, 1=Giả")
    slm_confidence: float = Field(..., description="Độ tự tin của SLM (0.0 -> 1.0)")
    status: str = Field(..., description="Trạng thái phân tích chuyên sâu (e.g. 'pending_analysis' hoặc 'completed')")

class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Nội dung bài viết cần phân tích")
    fb_post_id: Optional[str] = Field(None, description="ID bài viết trên Facebook (nếu có)")
    fb_post_created_at: Optional[datetime] = Field(None, description="Thời gian đăng bài viết (nếu có)")
    record_id: Optional[int] = Field(None, description="ID bản ghi đã tạo từ bước dự đoán nhanh (nếu có)")

class ChunkEvidence(BaseModel):
    score: float
    chunk_text: str
    title: str
    url: str
    source: str

class FewshotDemo(BaseModel):
    text: str
    label: str
    source: str

class AnalyzeResponse(BaseModel):
    record_id: int
    fb_post_id: Optional[str]
    post_text: str
    normalized_text: str
    slm_label: int
    slm_confidence: float
    llm_label: int
    llm_explanation: Optional[str]
    wiki_evidence: Optional[Dict[str, str]] = None
    rag_evidence: Optional[List[ChunkEvidence]] = None
    fewshot_examples: Optional[List[FewshotDemo]] = None
    final_prompt: Optional[str] = None
    created_at: datetime
