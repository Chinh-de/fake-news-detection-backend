from pydantic import BaseModel
from typing import List, Dict, Any

class DashboardOverview(BaseModel):
    total_checked: int
    total_fake: int
    total_real: int
    total_conflict: int

class TimeSeriesPoint(BaseModel):
    time_label: str  # Format: "YYYY-MM-DD HH:00" or similar
    count: int

class ConfidenceSeriesPoint(BaseModel):
    time_label: str
    avg_confidence: float

class AgreementSeriesPoint(BaseModel):
    time_label: str
    count: int

class DashboardCharts(BaseModel):
    # Distribution over time (hourly)
    time_distribution: List[TimeSeriesPoint]
    
    # Pie chart data
    pie_fake: int
    pie_real: int
    pie_conflict: int
    
    # SLM confidence over time
    slm_confidence_distribution: List[ConfidenceSeriesPoint]
    
    # SLM > 80% confidence & SLM == LLM count over time
    slm_llm_agreement_distribution: List[AgreementSeriesPoint]

class DashboardResponse(BaseModel):
    overview: DashboardOverview
    charts: DashboardCharts
