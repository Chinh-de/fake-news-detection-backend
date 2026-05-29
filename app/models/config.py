from sqlalchemy import Column, String, Text, Integer, DateTime
from datetime import datetime
from app.database import Base

class SystemConfig(Base):
    __tablename__ = "system_config"

    key = Column(String, primary_key=True, index=True)
    value = Column(Text, nullable=False)

class SLMUpdateHistory(Base):
    __tablename__ = "slm_update_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    version_sha = Column(String, nullable=False)
    status = Column(String, nullable=False)  # "SUCCESS" or "FAILED"
    error_message = Column(Text, nullable=True)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, default=datetime.utcnow)
