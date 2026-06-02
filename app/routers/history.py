from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import Optional
from app.database import get_db
from app.models.prediction import PredictionRecord
from app.schemas.history import HistoryPaginated, HistoryListItem
from app.schemas.prediction import AnalyzeResponse
from app.core.security import get_current_admin

router = APIRouter(prefix="/history", tags=["history"])

@router.get("", response_model=HistoryPaginated, status_code=status.HTTP_200_OK)
async def get_history_list(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Page size"),
    is_trained: Optional[str] = Query(None, alias="isTrained", description="Filter by is_trained: true, false, all"),
    current_user: str = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Lấy lịch sử dự đoán tin tức có phân trang và bộ lọc isTrained.
    Yêu cầu quyền Admin.
    """
    offset = (page - 1) * limit
    
    # Base query
    query = select(PredictionRecord)
    
    # Filter by is_trained
    if is_trained == "true":
        query = query.where(PredictionRecord.is_trained == True)
    elif is_trained == "false":
        query = query.where(PredictionRecord.is_trained == False)
        
    # Total count query
    count_stmt = select(func.count()).select_from(query.subquery())
    total_records = (await db.execute(count_stmt)).scalar() or 0
    
    # Get items sorted by latest
    query = query.order_by(desc(PredictionRecord.created_at)).offset(offset).limit(limit)
    res = await db.execute(query)
    records = res.scalars().all()
    
    items = []
    for r in records:
        # Create text snippet (first 120 chars)
        snippet = r.post_text[:120] + "..." if len(r.post_text) > 120 else r.post_text
        
        items.append(
            HistoryListItem(
                id=r.id,
                text_snippet=snippet,
                fb_post_id=r.fb_post_id,
                slm_label=r.slm_label,
                slm_confidence=r.slm_confidence,
                llm_label=r.llm_label,
                is_trained=r.is_trained,
                created_at=r.created_at
            )
        )
        
    total_pages = (total_records + limit - 1) // limit if total_records > 0 else 0
    
    return HistoryPaginated(
        items=items,
        total_records=total_records,
        page=page,
        limit=limit,
        total_pages=total_pages
    )


@router.get("/{record_id}", response_model=AnalyzeResponse, status_code=status.HTTP_200_OK)
async def get_history_detail(
    record_id: int,
    current_user: str = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Lấy chi tiết một kết quả dự đoán cụ thể theo ID.
    Hiển thị đầy đủ thông tin bao gồm nội dung, nguồn, kết quả SLM, RAG Wikipedia, Internet, prompt và LLM.
    Yêu cầu quyền Admin.
    """
    stmt = select(PredictionRecord).where(PredictionRecord.id == record_id)
    res = await db.execute(stmt)
    record = res.scalars().first()
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy bản ghi lịch sử với ID {record_id}"
        )
        
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
        created_at=record.created_at
    )


@router.delete("/{record_id}", status_code=status.HTTP_200_OK)
async def delete_history_record(
    record_id: int,
    current_user: str = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Xóa một bản ghi lịch sử dự đoán theo ID.
    Yêu cầu quyền Admin.
    """
    stmt = select(PredictionRecord).where(PredictionRecord.id == record_id)
    res = await db.execute(stmt)
    record = res.scalars().first()
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy bản ghi lịch sử với ID {record_id}"
        )
        
    try:
        await db.delete(record)
        await db.commit()
        return {"message": f"Đã xóa thành công bản ghi {record_id}"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xóa bản ghi: {str(e)}"
        )
