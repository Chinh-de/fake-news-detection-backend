import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.config import settings
from app.services.text_processor import preprocess_text
import logging

# Logger configuration
logger = logging.getLogger("slm_service")
level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logger.setLevel(level)

class SLMService:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.SLM_MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.current_sha = "local"
        self.is_updating = False
        self.last_update_error = None
        
    def load_model(self, model_path: str = None, sha: str = None):
        """Load the fine-tuned PhoBERT model from disk."""
        if self.model is not None:
            return
            
        load_path = model_path or self.model_path
        self.current_sha = sha or "local"
        
        logger.info("Loading SLM model from %s (SHA: %s) on device %s...", load_path, self.current_sha, self.device)
        if not os.path.exists(load_path):
            logger.warning("Configured path '%s' does not exist, falling back to default SLM_MODEL_PATH.", load_path)
            load_path = self.model_path
            self.current_sha = "local"
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"SLM model path '{load_path}' does not exist.")
            
        # Load tokenizer and model locally and offline only
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            use_fast=False,
            local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            num_labels=2,
            local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("SLM model loaded successfully.")

    async def hot_swap_model(self, db, latest_sha: str):
        """
        Download a model from Hugging Face, load it in memory,
        and atomically hot-swap the loaded model and tokenizer.
        """
        if self.is_updating:
            logger.warning("SLM update is already in progress.")
            return
            
        self.is_updating = True
        self.last_update_error = None
        
        from datetime import datetime
        triggered_at = datetime.utcnow()
        
        try:
            logger.info("Starting SLM model hot-swap update to Hugging Face SHA: %s", latest_sha)
            
            # Use SLM_REPO_ID directly from settings/environment
            repo_id = settings.SLM_REPO_ID
            
            # 1. Download model from Hugging Face to downloads directory
            target_dir = os.path.join(self.model_path, "downloads", latest_sha)
            os.makedirs(target_dir, exist_ok=True)
            
            from huggingface_hub import snapshot_download
            import asyncio
            loop = asyncio.get_event_loop()
            
            logger.info("Downloading model files from %s for revision %s...", repo_id, latest_sha)
            # Run snapshot_download in executor
            await loop.run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=repo_id,
                    revision=latest_sha,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False
                )
            )
            logger.info("Model download complete. Loading new model into RAM/VRAM...")
            
            # 2. Load tokenizer and model in memory (in executor)
            def _load_new_model():
                new_tok = AutoTokenizer.from_pretrained(
                    target_dir,
                    use_fast=False,
                    local_files_only=True
                )
                new_mod = AutoModelForSequenceClassification.from_pretrained(
                    target_dir,
                    num_labels=2,
                    local_files_only=True
                )
                new_mod.to(self.device)
                new_mod.eval()
                return new_tok, new_mod
                
            new_tokenizer, new_model = await loop.run_in_executor(None, _load_new_model)
            
            # 3. Swap references atomically
            old_tokenizer = self.tokenizer
            old_model = self.model
            
            self.tokenizer = new_tokenizer
            self.model = new_model
            self.current_sha = latest_sha
            
            # 4. Persist configuration to Database
            from app.models.config import SystemConfig, SLMUpdateHistory
            from sqlalchemy import select
            for key, val in [
                ("slm_active_model_path", target_dir),
                ("slm_active_model_sha", latest_sha)
            ]:
                stmt = select(SystemConfig).where(SystemConfig.key == key)
                res = await db.execute(stmt)
                cfg = res.scalars().first()
                if cfg:
                    cfg.value = val
                else:
                    db.add(SystemConfig(key=key, value=val))
            
            # 5. Log update history
            db.add(SLMUpdateHistory(
                version_sha=latest_sha,
                status="SUCCESS",
                error_message=None,
                triggered_at=triggered_at,
                completed_at=datetime.utcnow()
            ))
                     
            await db.commit()
            
            logger.info("SLM model hot-swap update to SHA %s completed successfully!", latest_sha)

            # Keep at most 2 newest version directories in downloads/
            downloads_dir = os.path.join(self.model_path, "downloads")
            if os.path.exists(downloads_dir):
                try:
                    import shutil
                    subdirs = [os.path.join(downloads_dir, d) for d in os.listdir(downloads_dir)]
                    subdirs = [d for d in subdirs if os.path.isdir(d)]
                    # Sort by modification time (oldest first)
                    subdirs.sort(key=os.path.getmtime)
                    if len(subdirs) > 2:
                        logger.info("Found %d model version directories on disk. Keeping 2 newest, cleaning up old versions...", len(subdirs))
                        while len(subdirs) > 2:
                            oldest_dir = subdirs.pop(0)
                            logger.info("Deleting old model version directory to save space: %s", oldest_dir)
                            shutil.rmtree(oldest_dir)
                except Exception as cleanup_err:
                    logger.error("Error during model directory cleanup: %s", cleanup_err)
            
            # Explicitly clean up old model/VRAM if CUDA is used
            if torch.cuda.is_available():
                import gc
                del old_model
                del old_tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error("SLM model hot-swap failed: %s. Rolling back to old model.", e)
            self.last_update_error = str(e)
            await db.rollback()

            # Clean up the failed download directory if it exists
            if os.path.exists(target_dir):
                logger.info("Cleaning up failed download directory: %s", target_dir)
                import shutil
                try:
                    shutil.rmtree(target_dir)
                except Exception as cleanup_err:
                    logger.error("Failed to delete failed download directory %s: %s", target_dir, cleanup_err)
            
            # Persist failure details using a new database session
            try:
                from app.database import SessionLocal
                from app.models.config import SLMUpdateHistory
                async with SessionLocal() as error_db:
                    error_db.add(SLMUpdateHistory(
                        version_sha=latest_sha,
                        status="FAILED",
                        error_message=str(e),
                        triggered_at=triggered_at,
                        completed_at=datetime.utcnow()
                    ))
                    await error_db.commit()
            except Exception as history_err:
                logger.error("Failed to write failure history: %s", history_err)
                
            raise e
        finally:
            self.is_updating = False

    def predict(self, text: str) -> tuple[int, float]:
        """
        Run inference using the SLM.
        Returns:
            pred_label: 0 for Real (Thật), 1 for Fake (Giả)
            confidence: float between 0.0 and 1.0
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        cleaned_text = preprocess_text(text)
        
        # Word segmentation using underthesea
        try:
            from underthesea import word_tokenize
            cleaneed_text = word_tokenize(cleaned_text, format="text")
        except Exception as e:
            logger.error("Failed to tokenize text using underthesea: %s", e)
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length"
        )
        
        # Run model
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device)
            )
            probs = F.softmax(outputs.logits, dim=1)
            
        confidence, prediction = torch.max(probs, dim=1)
        return int(prediction.item()), float(confidence.item())

# Global singleton instance
slm_service = SLMService()
