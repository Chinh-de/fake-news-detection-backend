import os
import pandas as pd
from datetime import datetime
from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.models.prediction import NewsCorpus

class CorpusService:
    async def init_db_extensions(self, db: AsyncSession):
        """Create pg_trgm extension if not exists."""
        print("Ensuring pg_trgm extension is loaded...")
        try:
            await db.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
            await db.commit()
            print("pg_trgm extension loaded successfully.")
        except Exception as e:
            print(f"Error loading pg_trgm extension: {e}")
            await db.rollback()

    async def seed_corpus_if_empty(self, db: AsyncSession):
        """Seed news_corpus table from CSV if it is empty."""
        # 1. Check if empty
        stmt = select(func.count(NewsCorpus.id))
        res = await db.execute(stmt)
        count = res.scalar() or 0
        
        if count > 0:
            print(f"News corpus already seeded. Found {count} records. Skipping seeding.")
            return
            
        csv_path = settings.SEED_CORPUS_CSV
        print(f"News corpus database is empty. Seeding from {csv_path}...")
        
        if not os.path.exists(csv_path):
            print(f"Seed CSV file not found at {csv_path}. Skipping seed. Please configure SEED_CORPUS_CSV in .env")
            return
            
        try:
            # Load CSV using pandas
            df = pd.read_csv(csv_path)
            
            records_to_insert = []
            
            for _, row in df.iterrows():
                text_val = str(row['text']) if ('text' in row and pd.notna(row['text'])) else ""
                
                if not text_val.strip():
                    continue
                    
                records_to_insert.append(
                    NewsCorpus(
                        text=text_val,
                        source_dataset="news_corpus"
                    )
                )
                
            # Bulk insert in batches of 1000
            batch_size = 1000
            print(f"Preparing to insert {len(records_to_insert)} records into news_corpus...")
            for i in range(0, len(records_to_insert), batch_size):
                batch = records_to_insert[i:i+batch_size]
                db.add_all(batch)
                await db.commit()
                
            print(f"Successfully seeded {len(records_to_insert)} records into news_corpus.")
        except Exception as e:
            print(f"Error seeding news corpus: {e}")
            await db.rollback()

# Global singleton instance
corpus_service = CorpusService()
