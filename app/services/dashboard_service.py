from datetime import datetime, timedelta
from sqlalchemy import select, func, and_, or_
from sqlalchemy.future import select
from app.database import AsyncSession
from app.models.prediction import PredictionRecord
from app.schemas.dashboard import (
    DashboardOverview, DashboardCharts, DashboardResponse,
    TimeSeriesPoint, ConfidenceSeriesPoint, AgreementSeriesPoint
)

class DashboardService:
    async def get_dashboard_data(
        self, db: AsyncSession, start_date: datetime = None, end_date: datetime = None
    ) -> DashboardResponse:
        # Build date filters
        filters = []
        if start_date:
            filters.append(PredictionRecord.created_at >= start_date)
        if end_date:
            filters.append(PredictionRecord.created_at <= end_date)
            
        # 1. Calculate Overview Metrics
        # Total Checked
        total_stmt = select(func.count(PredictionRecord.id)).where(and_(*filters) if filters else True)
        total_checked = (await db.execute(total_stmt)).scalar() or 0
        
        # Real: slm == 0 and llm == 0
        real_stmt = select(func.count(PredictionRecord.id)).where(
            and_(
                PredictionRecord.slm_label == 0,
                PredictionRecord.llm_label == 0,
                *filters
            ) if filters else and_(PredictionRecord.slm_label == 0, PredictionRecord.llm_label == 0)
        )
        total_real = (await db.execute(real_stmt)).scalar() or 0
        
        # Fake: slm == 1 and llm == 1
        fake_stmt = select(func.count(PredictionRecord.id)).where(
            and_(
                PredictionRecord.slm_label == 1,
                PredictionRecord.llm_label == 1,
                *filters
            ) if filters else and_(PredictionRecord.slm_label == 1, PredictionRecord.llm_label == 1)
        )
        total_fake = (await db.execute(fake_stmt)).scalar() or 0
        
        # Conflict: slm != llm and llm != -1 (ignoring pending)
        conflict_stmt = select(func.count(PredictionRecord.id)).where(
            and_(
                PredictionRecord.slm_label != PredictionRecord.llm_label,
                PredictionRecord.llm_label != -1,
                *filters
            ) if filters else and_(PredictionRecord.slm_label != PredictionRecord.llm_label, PredictionRecord.llm_label != -1)
        )
        total_conflict = (await db.execute(conflict_stmt)).scalar() or 0
        
        # 2. Charts aggregation grouped by hour
        # Format for hourly: 'YYYY-MM-DD HH:00'
        # In PostgreSQL we can use date_trunc('hour', created_at)
        time_col = func.date_trunc('hour', PredictionRecord.created_at)
        
        # Chart 1: Time distribution (number of predictions by hour)
        dist_stmt = select(
            time_col,
            func.count(PredictionRecord.id)
        ).where(
            and_(*filters) if filters else True
        ).group_by(
            time_col
        ).order_by(
            time_col
        )
        dist_res = await db.execute(dist_stmt)
        time_distribution = [
            TimeSeriesPoint(time_label=row[0].strftime("%Y-%m-%d %H:00"), count=row[1])
            for row in dist_res.fetchall() if row[0] is not None
        ]
        
        # Chart 2: SLM confidence distribution over time (avg confidence by hour)
        conf_stmt = select(
            time_col,
            func.avg(PredictionRecord.slm_confidence)
        ).where(
            and_(*filters) if filters else True
        ).group_by(
            time_col
        ).order_by(
            time_col
        )
        conf_res = await db.execute(conf_stmt)
        slm_confidence_distribution = [
            ConfidenceSeriesPoint(time_label=row[0].strftime("%Y-%m-%d %H:00"), avg_confidence=float(row[1] or 0.0))
            for row in conf_res.fetchall() if row[0] is not None
        ]
        
        # Chart 3: SLM-LLM agreement distribution where slm_confidence > 0.8 and slm == llm by hour
        agree_stmt = select(
            time_col,
            func.count(PredictionRecord.id)
        ).where(
            and_(
                PredictionRecord.slm_confidence > 0.8,
                PredictionRecord.slm_label == PredictionRecord.llm_label,
                *filters
            ) if filters else and_(
                PredictionRecord.slm_confidence > 0.8,
                PredictionRecord.slm_label == PredictionRecord.llm_label
            )
        ).group_by(
            time_col
        ).order_by(
            time_col
        )
        agree_res = await db.execute(agree_stmt)
        slm_llm_agreement_distribution = [
            AgreementSeriesPoint(time_label=row[0].strftime("%Y-%m-%d %H:00"), count=row[1])
            for row in agree_res.fetchall() if row[0] is not None
        ]
        
        # Return response
        return DashboardResponse(
            overview=DashboardOverview(
                total_checked=total_checked,
                total_fake=total_fake,
                total_real=total_real,
                total_conflict=total_conflict
            ),
            charts=DashboardCharts(
                time_distribution=time_distribution,
                pie_fake=total_fake,
                pie_real=total_real,
                pie_conflict=total_conflict,
                slm_confidence_distribution=slm_confidence_distribution,
                slm_llm_agreement_distribution=slm_llm_agreement_distribution
            )
        )

# Global singleton instance
dashboard_service = DashboardService()
