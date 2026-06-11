from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas.prediction import AnalyzeRequest, AnalyzeResponse
from app.services.prediction_service import prediction_service
import logging

router = APIRouter(prefix="/analyze", tags=["analysis"])
logger = logging.getLogger("analyze_router")

@router.post("", response_model=AnalyzeResponse, status_code=status.HTTP_200_OK)
async def analyze_news(request: AnalyzeRequest, db: AsyncSession = Depends(get_db)):
    """
    Phân tích chuyên sâu tin tức dùng RAG (Wikipedia, Google News) + LLM Qwen.
    Nếu fb_post_id đã được phân tích trước đó, tái sử dụng kết quả phân tích LLM cũ,
    chỉ dự đoán lại bằng SLM theo quy tắc chống trùng lặp.
    """
    try:
        record = await prediction_service.run_deep_analysis(
            db=db,
            text_input=request.text,
            fb_post_id=request.fb_post_id,
            fb_post_created_at=request.fb_post_created_at,
            record_id=request.record_id
        )
        
        # Format the RAG evidence chunks and few-shot examples to match the response schema
        rag_evidence_formatted = []
        if record.rag_evidence:
            for item in record.rag_evidence:
                rag_evidence_formatted.append({
                    "score": item.get("score", 0.0),
                    "chunk_text": item.get("chunk_text", ""),
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", "")
                })
                
        fewshot_formatted = []
        if record.fewshot_examples:
            for item in record.fewshot_examples:
                fewshot_formatted.append({
                    "text": item.get("text", ""),
                    "label": item.get("label", ""),
                    "source": item.get("source", "")
                })
                
        return AnalyzeResponse(
            record_id=record.id,
            fb_post_id=record.fb_post_id,
            post_text=record.post_text,
            normalized_text=record.normalized_text or "",
            slm_label=record.slm_label,
            slm_confidence=record.slm_confidence,
            llm_label=record.llm_label,
            llm_explanation=record.llm_explanation,
            wiki_evidence=record.wiki_evidence,
            rag_evidence=rag_evidence_formatted,
            fewshot_examples=fewshot_formatted,
            final_prompt=record.final_prompt,
            created_at=record.created_at,
            xgboost_label=getattr(record, "xgboost_label", None),
            xgboost_confidence=getattr(record, "xgboost_confidence", None)
        )
    except Exception as e:
        logger.exception("Lỗi hệ thống khi chạy phân tích RAG+LLM: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi chạy phân tích RAG+LLM: {str(e)}"
        )
