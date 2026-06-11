import json
import re
from datetime import datetime
from sqlalchemy.future import select
from app.database import AsyncSession
from app.models.prediction import PredictionRecord, XGBoostPrediction
from app.services.slm_service import slm_service
from app.services.llm_service import llm_service
from app.services.xgboost_service import xgboost_service
from app.services.retrieval_service import retrieval_service, ALL_SYNONYM_LABELS
from app.services.text_processor import clean_text_transformer, preprocess_text
from app.core.prompts import (
    build_entity_extraction_prompt,
    build_final_classification_prompt
)
import logging
from app.config import settings

# Logger for analysis pipeline
logger = logging.getLogger("prediction_service")
level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logger.setLevel(level)

class PredictionService:
    @staticmethod
    def _normalize_fb_post_id(fb_post_id: str | None) -> str | None:
        """Normalize placeholder/invalid post ids so they are not reused as real keys."""
        if fb_post_id is None:
            return None
        value = str(fb_post_id).strip()
        if not value:
            return None
        if value.lower() in {"n/a", "na", "none", "null", "undefined", "unknown"}:
            return None
        return value

    async def run_deep_analysis(
        self, db: AsyncSession, text_input: str, fb_post_id: str = None, fb_post_created_at: datetime = None, record_id: int = None
    ) -> PredictionRecord:
        """
        Logic for POST /api/analyze:
        Check if record_id or fb_post_id already exists.
        If yes -> rerun SLM, update DB, but KEEP original LLM result from DB, and return.
        If no -> run full pipeline (SLM + RAG + LLM), save new record to DB, and return.
        """
        normalized = clean_text_transformer(text_input)
        fb_post_id = self._normalize_fb_post_id(fb_post_id)
        logger.debug("Normalized input length=%d", len(normalized) if normalized else 0)
        slm_label, slm_confidence = slm_service.predict(text_input)
        logger.debug("SLM prediction=%s confidence=%s", slm_label, slm_confidence)

        # Predict with XGBoost
        try:
            xgb_label, xgb_confidence = xgboost_service.predict(text_input)
        except Exception as e:
            logger.error("Failed to predict with XGBoost: %s", e)
            xgb_label, xgb_confidence = None, None
        
        record = None
        if record_id:
            stmt = select(PredictionRecord).where(PredictionRecord.id == record_id)
            res = await db.execute(stmt)
            record = res.scalars().first()

        if not record and fb_post_id:
            stmt = select(PredictionRecord).where(PredictionRecord.fb_post_id == fb_post_id)
            res = await db.execute(stmt)
            record = res.scalars().first()
            
        if record and record.llm_label != -1:
            # Record already exists and has been analyzed by LLM.
            # Keep original LLM results, only update SLM prediction.
            record.post_text = text_input
            record.normalized_text = normalized
            record.slm_label = slm_label
            record.slm_confidence = slm_confidence
            if fb_post_created_at:
                record.fb_post_created_at = fb_post_created_at
            await db.commit()
            await db.refresh(record)
            logger.info("Re-used LLM analysis for existing record ID %d for fb_post_id %s", record.id, fb_post_id)

            # Update or create XGBoost prediction
            if xgb_label is not None and xgb_confidence is not None:
                stmt_xgb = select(XGBoostPrediction).where(XGBoostPrediction.prediction_record_id == record.id)
                res_xgb = await db.execute(stmt_xgb)
                xgb_pred = res_xgb.scalars().first()
                
                if xgb_pred:
                    xgb_pred.xgboost_label = xgb_label
                    xgb_pred.xgboost_confidence = xgb_confidence
                else:
                    xgb_pred = XGBoostPrediction(
                        prediction_record_id=record.id,
                        xgboost_label=xgb_label,
                        xgboost_confidence=xgb_confidence
                    )
                    db.add(xgb_pred)
                await db.commit()
                record.xgboost_label = xgb_label
                record.xgboost_confidence = xgb_confidence
            else:
                record.xgboost_label = None
                record.xgboost_confidence = None

            return record
            
        # Run full pipeline (either new record or existing record with pending LLM)
        
        # Step 1: Entity & Search Query extraction using LLM
        entity_sys_prompt, entity_user_prompt = build_entity_extraction_prompt(normalized)
        
        # Initialize defaults
        entities = []
        search_query = normalized[:80] # Default fallback query
        
        # Log the entity extraction prompts when analysis logging enabled
        try:
            if settings.ENABLE_ANALYSIS_LOG:
                logger.debug("Entity extraction system prompt:\n%s", entity_sys_prompt)
                logger.debug("Entity extraction user prompt:\n%s", entity_user_prompt)
        except Exception:
            pass

        raw_entities = await llm_service.call_llm(
            db,
            entity_sys_prompt,
            entity_user_prompt,
            max_tokens=1024
        )
        clean_resp = raw_entities.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", clean_resp, re.DOTALL)
        if match:
            clean_resp = match.group(0)
        
        logger.debug("Raw entity extraction response:\n%s", raw_entities)

        data = json.loads(clean_resp)
        entities = data.get("entities", [])
        search_query = data.get("query", search_query)
        logger.debug("Extracted entities=%s search_query=%s", entities, search_query)

        # Also save/log raw extraction response when enabled
        try:
            if settings.ENABLE_ANALYSIS_LOG:
                logger.debug("Entity extraction raw response:\n%s", raw_entities)
        except Exception:
            pass
            
        # Step 2: Wikipedia lookup
        wiki_definitions = {}
        try:
            wiki_definitions = await retrieval_service.get_wiki_definitions(entities)
            logger.debug("wiki_definitions count=%d", len(wiki_definitions))
        except Exception as e:
            logger.error("Error getting Wikipedia definitions: %s", e)
            logger.exception("Error getting Wikipedia definitions")
            
        # Step 3: Fewshot retrieval from News Corpus DB & Internet news (DDG) reranked by BM25
        fewshot_demos = []
        try:
            fewshot_demos = await retrieval_service.retrieve_fewshot_examples(db, text_input, search_query)
            logger.debug("fewshot_demos count=%d", len(fewshot_demos))
        except Exception as e:
            logger.error("Error retrieving fewshot examples: %s", e)
            logger.exception("Error retrieving fewshot examples")
            
        # Step 4: RAG article chunk retrieval & Semantic Search
        rag_evidence = []
        try:
            rag_evidence = await retrieval_service.retrieve_rag_evidence(search_query, normalized)
            logger.debug("rag_evidence count=%d", len(rag_evidence))
        except Exception as e:
            logger.error("Error retrieving RAG evidence: %s", e)
            logger.exception("Error retrieving RAG evidence")
            
        # Step 5: Final Prompt construction
        transformer_input = preprocess_text(text_input)
        logger.debug("Transformer input length=%d", len(transformer_input) if transformer_input else 0)
        final_system_prompt, final_user_prompt = build_final_classification_prompt(
            text_input=transformer_input,
            wiki_definitions=wiki_definitions,
            rag_evidence=rag_evidence,
            fewshot_demos=fewshot_demos
        )

        # Log the final prompt details if analysis logging is enabled
        logger.debug("Final prompt length=%d", len(final_user_prompt))
        logger.debug("Final prompt:\n%s", final_user_prompt)

        llm_raw = await llm_service.call_llm(db, final_system_prompt, final_user_prompt, max_tokens=4096)
        llm_label, llm_explanation = llm_service.parse_llm_json_response(llm_raw)
            
        if record:
            # We are completing a pending analysis record
            record.post_text = text_input
            record.normalized_text = normalized
            record.slm_label = slm_label
            record.slm_confidence = slm_confidence
            record.llm_label = llm_label
            record.llm_raw = llm_raw
            record.llm_explanation = llm_explanation
            record.wiki_evidence = wiki_definitions
            record.rag_evidence = rag_evidence
            record.fewshot_examples = fewshot_demos
            record.final_prompt = final_user_prompt
            if fb_post_created_at:
                record.fb_post_created_at = fb_post_created_at
            await db.commit()
            await db.refresh(record)
            logger.info("Completed pending LLM analysis for record ID %d", record.id)
        else:
            # Create a completely new record with full analysis
            record = PredictionRecord(
                fb_post_id=fb_post_id,
                post_text=text_input,
                normalized_text=normalized,
                slm_label=slm_label,
                slm_confidence=slm_confidence,
                llm_label=llm_label,
                llm_raw=llm_raw,
                llm_explanation=llm_explanation,
                wiki_evidence=wiki_definitions,
                rag_evidence=rag_evidence,
                fewshot_examples=fewshot_demos,
                final_prompt=final_user_prompt,
                fb_post_created_at=fb_post_created_at
            )
            db.add(record)
            await db.commit()
            await db.refresh(record)
            logger.info("Created new complete prediction record ID %d", record.id)

        # Update or create XGBoost prediction
        if xgb_label is not None and xgb_confidence is not None:
            stmt_xgb = select(XGBoostPrediction).where(XGBoostPrediction.prediction_record_id == record.id)
            res_xgb = await db.execute(stmt_xgb)
            xgb_pred = res_xgb.scalars().first()
            
            if xgb_pred:
                xgb_pred.xgboost_label = xgb_label
                xgb_pred.xgboost_confidence = xgb_confidence
            else:
                xgb_pred = XGBoostPrediction(
                    prediction_record_id=record.id,
                    xgboost_label=xgb_label,
                    xgboost_confidence=xgb_confidence
                )
                db.add(xgb_pred)
            await db.commit()
            record.xgboost_label = xgb_label
            record.xgboost_confidence = xgb_confidence
        else:
            record.xgboost_label = None
            record.xgboost_confidence = None
            
        return record

    async def get_or_create_predict(
        self, db: AsyncSession, text_input: str, fb_post_id: str = None, fb_post_created_at: datetime = None
    ) -> PredictionRecord:
        """
        Logic for POST /api/predict:
        Check if fb_post_id already exists.
        If yes -> rerun SLM prediction, update the existing record with new SLM pred, and return it.
        If no -> run SLM prediction, create a new record (LLM details set to default/null), and return it.
        """
        # Clean text
        normalized = clean_text_transformer(text_input)
        fb_post_id = self._normalize_fb_post_id(fb_post_id)
        
        # Predict with SLM
        slm_label, slm_confidence = slm_service.predict(text_input)
        
        # Predict with XGBoost
        try:
            xgb_label, xgb_confidence = xgboost_service.predict(text_input)
        except Exception as e:
            logger.error("Failed to predict with XGBoost: %s", e)
            xgb_label, xgb_confidence = None, None

        record = None
        if fb_post_id:
            # Check existing record by fb_post_id
            stmt = select(PredictionRecord).where(PredictionRecord.fb_post_id == fb_post_id)
            res = await db.execute(stmt)
            record = res.scalars().first()
            
        if record:
            # Update existing record's SLM prediction
            record.post_text = text_input
            record.normalized_text = normalized
            record.slm_label = slm_label
            record.slm_confidence = slm_confidence
            if fb_post_created_at:
                record.fb_post_created_at = fb_post_created_at
            await db.commit()
            await db.refresh(record)
            logger.info("Updated existing prediction record ID %d for fb_post_id %s", record.id, fb_post_id)
        else:
            # Create a new record
            record = PredictionRecord(
                fb_post_id=fb_post_id,
                post_text=text_input,
                normalized_text=normalized,
                slm_label=slm_label,
                slm_confidence=slm_confidence,
                llm_label=-1,  # Default -1 denotes pending/not analyzed
                llm_raw=None,
                llm_explanation="Chưa được phân tích chuyên sâu bởi LLM. Hãy yêu cầu Phân tích chuyên sâu.",
                wiki_evidence=None,
                rag_evidence=None,
                fewshot_examples=None,
                final_prompt=None,
                fb_post_created_at=fb_post_created_at
            )
            db.add(record)
            await db.commit()
            await db.refresh(record)
            logger.info("Created new prediction record ID %d", record.id)
            
        # Update or create XGBoost prediction
        if xgb_label is not None and xgb_confidence is not None:
            stmt_xgb = select(XGBoostPrediction).where(XGBoostPrediction.prediction_record_id == record.id)
            res_xgb = await db.execute(stmt_xgb)
            xgb_pred = res_xgb.scalars().first()
            
            if xgb_pred:
                xgb_pred.xgboost_label = xgb_label
                xgb_pred.xgboost_confidence = xgb_confidence
            else:
                xgb_pred = XGBoostPrediction(
                    prediction_record_id=record.id,
                    xgboost_label=xgb_label,
                    xgboost_confidence=xgb_confidence
                )
                db.add(xgb_pred)
            await db.commit()
            record.xgboost_label = xgb_label
            record.xgboost_confidence = xgb_confidence
        else:
            record.xgboost_label = None
            record.xgboost_confidence = None
            
        return record



# Global singleton instance
prediction_service = PredictionService()
