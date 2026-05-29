from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime, Index, JSON
from sqlalchemy.sql import func
from app.database import Base

class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True)
    fb_post_id = Column(String, nullable=True, index=True)
    post_text = Column(Text, nullable=False)
    normalized_text = Column(Text, nullable=True)
    
    slm_label = Column(Integer, nullable=False)        # 0 = Thật (Real), 1 = Giả (Fake)
    slm_confidence = Column(Float, nullable=False)
    
    llm_label = Column(Integer, nullable=False)        # 0 = Thật, 1 = Giả
    llm_raw = Column(Text, nullable=True)
    llm_explanation = Column(Text, nullable=True)
    
    wiki_evidence = Column(JSON, nullable=True)        # JSON containing Wikipedia definition parts
    rag_evidence = Column(JSON, nullable=True)         # JSON containing RAG text chunks and source URLs
    fewshot_examples = Column(JSON, nullable=True)     # JSON containing the few-shot examples
    final_prompt = Column(Text, nullable=True)         # The final prompt sent to the LLM
    
    is_trained = Column(Boolean, default=False, nullable=False)
    
    fb_post_created_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


class NewsCorpus(Base):
    __tablename__ = "news_corpus"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    label = Column(String, nullable=True)              # e.g., "True" or "False"
    label_id = Column(Integer, nullable=True)          # e.g., 0 or 1
    timestamp = Column(DateTime, nullable=True)
    source_dataset = Column(String, nullable=True)

# GIN Index for pg_trgm lexical search
# We specify postgresql_using='gin' and postgresql_ops={'text': 'gin_trgm_ops'}
# This allows quick similarity search using pg_trgm
Index(
    "idx_news_corpus_text_trgm",
    NewsCorpus.text,
    postgresql_using="gin",
    postgresql_ops={"text": "gin_trgm_ops"}
)
