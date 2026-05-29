from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import Optional
from app.database import get_db
from app.schemas.dashboard import DashboardResponse
from app.services.dashboard_service import dashboard_service
from app.core.security import get_current_admin

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

@router.get("", response_model=DashboardResponse, status_code=status.HTTP_200_OK)
async def get_dashboard(
    time_range: str = Query("all", alias="range", description="Time range: today, week, month, all"),
    start_date: Optional[datetime] = Query(None, description="Custom start date (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="Custom end date (ISO format)"),
    current_user: str = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Lấy thông tin và biểu đồ thống kê cho admin dashboard.
    Hỗ trợ lọc theo hôm nay, tuần này, tháng này hoặc khoảng thời gian tùy chỉnh.
    Yêu cầu quyền Admin.
    """
    now = datetime.now()
    
    # Process time range shortcut if custom dates are not provided
    if not start_date and not end_date:
        if time_range == "today":
            start_date = datetime(now.year, now.month, now.day)
            end_date = now
        elif time_range == "week":
            start_date = now - timedelta(days=7)
            end_date = now
        elif time_range == "month":
            # 30 days
            start_date = now - timedelta(days=30)
            end_date = now
            
    try:
        data = await dashboard_service.get_dashboard_data(
            db=db,
            start_date=start_date,
            end_date=end_date
        )
        return data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi tải dữ liệu dashboard: {str(e)}"
        )
