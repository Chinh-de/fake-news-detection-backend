from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.prediction_service import prediction_service

router = APIRouter(prefix="/predict", tags=["prediction"])

@router.post("", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_news(request: PredictRequest, db: AsyncSession = Depends(get_db)):
    """
    Dự đoán nhanh nhãn tin tức (Thật/Giả) bằng mô hình SLM PhoBERT.
    Nếu fb_post_id đã tồn tại, thực hiện cập nhật dự đoán SLM và trả về kết quả.
    """
    try:
        record = await prediction_service.get_or_create_predict(
            db=db,
            text_input=request.text,
            fb_post_id=request.fb_post_id,
            fb_post_created_at=request.fb_post_created_at
        )
        
        status_msg = "completed" if record.llm_label != -1 else "pending_analysis"
        
        return PredictResponse(
            record_id=record.id,
            fb_post_id=record.fb_post_id,
            slm_label=record.slm_label,
            slm_confidence=record.slm_confidence,
            status=status_msg
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi dự đoán SLM: {str(e)}"
        )
