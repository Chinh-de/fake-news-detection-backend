from sqlalchemy import Column, String, Text, Integer, DateTime, Float, JSON
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

class MRCDRunHistory(Base):
    __tablename__ = "mrcd_run_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    hf_commit_sha = Column(String, nullable=True)
    total_samples = Column(Integer, nullable=False)
    clean_count_r1 = Column(Integer, nullable=False, default=0)
    clean_count_r2 = Column(Integer, nullable=False, default=0)
    clean_count_r3 = Column(Integer, nullable=False, default=0)
    noisy_count_final = Column(Integer, nullable=False, default=0)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, default=datetime.utcnow)

class MRCDSampleRoundLog(Base):
    __tablename__ = "mrcd_sample_round_log"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(Integer, nullable=False, index=True)
    prediction_record_id = Column(Integer, nullable=False, index=True)
    round_id = Column(Integer, nullable=False)
    y_llm = Column(Integer, nullable=True)
    y_slm = Column(Integer, nullable=False)
    conf_slm = Column(Float, nullable=False)
    status = Column(String, nullable=False)  # "clean" or "noisy"
    
    # Round-specific RAG/Wiki/Fewshot details
    fewshot_examples = Column(JSON, nullable=True)
    rag_evidence = Column(JSON, nullable=True)
    wiki_evidence = Column(JSON, nullable=True)

