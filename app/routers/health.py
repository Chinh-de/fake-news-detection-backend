from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.database import get_db
from app.services.slm_service import slm_service

router = APIRouter(prefix="/health", tags=["health"])

@router.get("", status_code=status.HTTP_200_OK)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check for API, PostgreSQL connection, and SLM model load state."""
    db_status = "healthy"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
        
    slm_status = "loaded" if slm_service.model is not None else "not_loaded"
    
    return {
        "status": "healthy",
        "database": db_status,
        "slm_model": slm_status
    }
