import os
import re
import logging
import joblib
import torch
import numpy as np
from underthesea import word_tokenize
from app.config import settings
from app.services.slm_service import slm_service

logger = logging.getLogger("xgboost_service")
level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logger.setLevel(level)

class XGBoostService:
    def __init__(self):
        self.xgb_model = None
        self.scaler = None
        self.vectorizer = None
        self.is_loaded = False

    def load_model(self):
        if self.is_loaded:
            return

        xgb_dir = os.path.join(settings.SLM_MODEL_PATH, "xgboost_model")
        
        # Check if the files are present, download if any are missing
        required_files = ["xgboost_model.joblib", "phobert_scaler.joblib", "tfidf_vectorizer.joblib"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(xgb_dir, f))]
        
        if missing_files:
            logger.info("XGBoost model components %s are missing. Downloading from Hugging Face: chinhde/xgboost_model...", missing_files)
            os.makedirs(xgb_dir, exist_ok=True)
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="chinhde/xgboost_model",
                    local_dir=xgb_dir,
                    local_dir_use_symlinks=False
                )
                logger.info("XGBoost model components downloaded successfully.")
            except Exception as e:
                logger.error("Failed to download XGBoost model from Hugging Face: %s", e)
                if any(not os.path.exists(os.path.join(xgb_dir, f)) for f in required_files):
                    raise FileNotFoundError(f"XGBoost model files not found locally and download failed: {e}")

        logger.info("Loading XGBoost components from: %s", xgb_dir)

        # Load models
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.xgb_model = joblib.load(os.path.join(xgb_dir, "xgboost_model.joblib"))
                self.scaler = joblib.load(os.path.join(xgb_dir, "phobert_scaler.joblib"))
                self.vectorizer = joblib.load(os.path.join(xgb_dir, "tfidf_vectorizer.joblib"))
            self.is_loaded = True
            logger.info("XGBoost service loaded components successfully.")
        except Exception as e:
            logger.error("Failed to load XGBoost components: %s", e)
            raise e

    def predict(self, text: str) -> tuple[int, float]:
        """
        Predict Vietnamese news authenticity using XGBoost + TF-IDF + PhoBERT features.
        Returns:
            label: 0 for Real (Thật), 1 for Fake (Giả)
            confidence: float between 0.5 and 1.0 (representing probability percentage)
        """
        if not self.is_loaded:
            self.load_model()

        # Step 1: Preprocess text (simple clean + underthesea word tokenize)
        cleaned = text.lower()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        tokenized = word_tokenize(cleaned, format="text")

        # Step 2: TF-IDF feature extraction
        tfidf_feats = self.vectorizer.transform([tokenized]).toarray()

        # Step 3: PhoBERT CLS token embedding extraction
        # Ensure SLM service model is loaded
        if slm_service.model is None or slm_service.tokenizer is None:
            slm_service.load_model()

        inputs = slm_service.tokenizer(
            tokenized,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length"
        )

        with torch.no_grad():
            outputs = slm_service.model.base_model(
                input_ids=inputs["input_ids"].to(slm_service.device),
                attention_mask=inputs["attention_mask"].to(slm_service.device)
            )
            # CLS token embedding (index 0)
            phobert_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Step 4: Scale PhoBERT embeddings
        phobert_emb_scaled = self.scaler.transform(phobert_emb)

        # Step 5: Stack features
        X = np.hstack([tfidf_feats, phobert_emb_scaled])

        # Step 6: Prediction probabilities
        probs = self.xgb_model.predict_proba(X)[:, 1]
        prob = float(probs[0])
        label = int(prob >= 0.5)

        # Confidence: 0.5 -> 1.0
        confidence = prob if label == 1 else (1.0 - prob)

        return label, confidence

# Global singleton instance
xgboost_service = XGBoostService()
