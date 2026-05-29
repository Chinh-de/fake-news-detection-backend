from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.database import engine, Base, SessionLocal
from app.routers import (
    predict_router, analyze_router, dashboard_router,
    history_router, admin_router, health_router
)
from app.services.slm_service import slm_service
from app.services.retrieval_service import retrieval_service
from app.services.corpus_service import corpus_service

import asyncio
from datetime import datetime, timezone
import logging
from sqlalchemy import select
from app.models.config import SystemConfig

logger = logging.getLogger("main_lifespan")

async def perform_scheduled_update(sha: str):
    async with SessionLocal() as db:
        try:
            await slm_service.hot_swap_model(db, sha)
        except Exception as e:
            logger.error(f"Scheduled SLM hot-swap update failed: {e}")

async def run_scheduler_loop():
    logger.info("Starting SLM update scheduler loop...")
    while True:
        try:
            await asyncio.sleep(30)
            async with SessionLocal() as db:
                # Fetch schedule configs
                res_time = await db.execute(select(SystemConfig).where(SystemConfig.key == "slm_update_schedule_time"))
                cfg_time = res_time.scalars().first()
                res_sha = await db.execute(select(SystemConfig).where(SystemConfig.key == "slm_update_schedule_sha"))
                cfg_sha = res_sha.scalars().first()
                
                if cfg_time and cfg_time.value and cfg_sha and cfg_sha.value:
                    try:
                        target_dt = datetime.fromisoformat(cfg_time.value.replace("Z", "+00:00"))
                        now = datetime.now(timezone.utc) if target_dt.tzinfo else datetime.now()
                        
                        if now >= target_dt:
                            logger.info(f"Scheduled update time reached! Swapping model to revision: {cfg_sha.value}")
                            
                            # Clear schedule first to prevent double runs
                            cfg_time.value = ""
                            cfg_sha.value = ""
                            await db.commit()
                            
                            # Trigger hot-swap (run in background)
                            asyncio.create_task(perform_scheduled_update(cfg_sha.value))
                    except Exception as parse_err:
                        logger.error(f"Failed to parse schedule time or execute update: {parse_err}")
        except asyncio.CancelledError:
            logger.info("SLM update scheduler loop cancelled.")
            break
        except Exception as loop_err:
            logger.error(f"Error in SLM scheduler loop: {loop_err}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to initialize database and load AI models on startup."""
    print("==================================================")
    print("Starting Fake News Detection Backend...")
    
    from sqlalchemy import text
    # 1. Create DB tables and enable pg_trgm extension
    async with engine.begin() as conn:
        print("Ensuring pg_trgm extension and database tables are created...")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
        await conn.run_sync(Base.metadata.create_all)
        
    async with SessionLocal() as session:
        # Load trigram extension
        await corpus_service.init_db_extensions(session)
        # Seed the database corpus if it is empty
        await corpus_service.seed_corpus_if_empty(session)
        
        # Load custom model configuration if present in database
        try:
            res_path = await session.execute(select(SystemConfig).where(SystemConfig.key == "slm_active_model_path"))
            cfg_path = res_path.scalars().first()
            res_sha = await session.execute(select(SystemConfig).where(SystemConfig.key == "slm_active_model_sha"))
            cfg_sha = res_sha.scalars().first()
            
            active_path = cfg_path.value if cfg_path else None
            active_sha = cfg_sha.value if cfg_sha else None
            
            repo_id = settings.SLM_REPO_ID
        except Exception as db_err:
            print(f"Error fetching active model config: {db_err}")
            active_path = None
            active_sha = None
            repo_id = settings.SLM_REPO_ID

        # Global Hugging Face authentication
        if settings.HF_TOKEN:
            try:
                from huggingface_hub import login
                login(token=settings.HF_TOKEN)
                print("Successfully logged in to Hugging Face globally.")
            except Exception as hf_login_err:
                print(f"Warning: Failed to log in to Hugging Face globally: {hf_login_err}")

        # Check if the active path exists, or if fallback local path exists.
        # If not, download from HF and treat it as a version.
        import os
        local_dev_path = settings.SLM_MODEL_PATH
        local_dev_config = os.path.join(local_dev_path, "config.json")
        
        need_download = False
        
        if active_path and active_sha:
            active_config = os.path.join(active_path, "config.json")
            if not os.path.exists(active_path) or not os.path.exists(active_config):
                print(f"Active model path '{active_path}' does not exist or is invalid.")
                # Active path doesn't exist, check if local dev path exists
                if not os.path.exists(local_dev_path) or not os.path.exists(local_dev_config):
                    need_download = True
                else:
                    # Fallback to local dev path
                    active_path = local_dev_path
                    active_sha = "local"
        else:
            # No active version set in DB. Check local dev path
            if not os.path.exists(local_dev_path) or not os.path.exists(local_dev_config):
                need_download = True
            else:
                active_path = local_dev_path
                active_sha = "local"
                
        if need_download:
            print(f"Local model not found. Fetching latest version info from Hugging Face for repo '{repo_id}'...")
            from huggingface_hub import HfApi, snapshot_download
            try:
                # 1. Fetch latest SHA
                api = HfApi()
                info = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: api.model_info(repo_id)
                )
                latest_sha = info.sha
                
                # 2. Define download directory under downloads/<latest_sha>
                target_dir = os.path.join(local_dev_path, "downloads", latest_sha)
                os.makedirs(target_dir, exist_ok=True)
                
                # Check if already downloaded
                target_config = os.path.join(target_dir, "config.json")
                if not os.path.exists(target_config):
                    print(f"Downloading model from Hugging Face for version {latest_sha}...")
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: snapshot_download(
                            repo_id=repo_id,
                            revision=latest_sha,
                            local_dir=target_dir,
                            local_dir_use_symlinks=False
                        )
                    )
                
                # 3. Update DB configuration to point to this downloaded version
                active_path = target_dir
                active_sha = latest_sha
                
                for key, val in [
                    ("slm_active_model_path", target_dir),
                    ("slm_active_model_sha", latest_sha)
                ]:
                    stmt = select(SystemConfig).where(SystemConfig.key == key)
                    res = await session.execute(stmt)
                    cfg = res.scalars().first()
                    if cfg:
                        cfg.value = val
                    else:
                        session.add(SystemConfig(key=key, value=val))
                await session.commit()
                print(f"Successfully initialized and saved default HF model version: {latest_sha}")
            except Exception as download_err:
                print(f"Failed to auto-download default model from Hugging Face: {download_err}")
                # Fall back to local_dev_path anyway
                active_path = local_dev_path
                active_sha = "local"

    # 2. Load the SLM model (restoring database state if set)
    try:
        if active_path and active_sha:
            print(f"Restoring active model from database configuration: {active_path} (SHA: {active_sha})")
            slm_service.load_model(model_path=active_path, sha=active_sha)
        else:
            slm_service.load_model()
    except Exception as e:
        print(f"CRITICAL ERROR loading SLM model: {e}")
        print("Backend starting without pre-loaded SLM. Will try to load on first prediction request.")
        
    # 3. Load embedding model (multilingual-e5-small) to verify download & cache
    try:
        retrieval_service.get_encoder()
    except Exception as e:
        print(f"WARNING: Error downloading/loading embedding model: {e}")
        
    # 4. Start the background scheduler task
    scheduler_task = asyncio.create_task(run_scheduler_loop())
    
    print("Fake News Detection Backend is ready!")
    print("==================================================")
    
    yield
    
    # Clean up and close DB connections
    print("Shutting down backend...")
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass
    await engine.dispose()
    print("Backend shut down.")

app = FastAPI(
    title="Fake News Detection System API",
    description="FastAPI Backend for Fake News Detection System using PhoBERT (SLM), RAG and Qwen (LLM)",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers under /api prefix
app.include_router(predict_router, prefix="/api")
app.include_router(analyze_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(health_router, prefix="/api")

@app.get("/")
async def root():
    return {
        "message": "Fake News Detection API is running. Go to /docs for Swagger documentation.",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
