import os
import json
import httpx
from openai import OpenAI
from app.config import settings
from app.models.config import SystemConfig
from sqlalchemy import select
import logging

# Logger configuration
logger = logging.getLogger("llm_service")
if settings.ENABLE_ANALYSIS_LOG:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level)
    logger.setLevel(level)

class LLMService:
    async def get_client_config(self, db) -> tuple[str, str, str, str]:
        """
        Get LLM configuration from database or fallback to settings.
        Returns:
            (provider, base_url, api_key, model_name)
        """
        provider = settings.LLM_PROVIDER
        base_url = settings.LLM_ENDPOINT
        api_key = settings.LLM_API_KEY
        model_name = settings.LLM_MODEL
        
        try:
            # Query from DB system config
            stmt = select(SystemConfig).where(SystemConfig.key.in_(["llm_provider", "llm_endpoint", "llm_api_key", "llm_model"]))
            result = await db.execute(stmt)
            configs = {cfg.key: cfg.value for cfg in result.scalars().all()}
            
            if "llm_provider" in configs:
                provider = configs["llm_provider"]
            if "llm_endpoint" in configs:
                base_url = configs["llm_endpoint"]
            if "llm_api_key" in configs:
                api_key = configs["llm_api_key"]
            if "llm_model" in configs:
                model_name = configs["llm_model"]
        except Exception as e:
            logger.warning("Error querying LLM config from DB: %s. Using defaults from .env", e)
            
        # Fallback for GOOGLE_API_KEY if provider is genai and api_key is default/empty
        if provider == "genai" and (not api_key or api_key == "EMPTY" or api_key == "default_api_key"):
            api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
            
        return provider, base_url, api_key, model_name

    async def call_llm(self, db, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """
        Call the dynamic LLM provider (openai or genai) configured in DB.
        """
        provider, base_url, api_key, model_name = await self.get_client_config(db)
        
        # Analysis logging: record prompts sent when enabled
        try:
            if settings.ENABLE_ANALYSIS_LOG:
                logger.debug("LLM system prompt:\n%s", system_prompt)
                logger.debug("LLM user prompt:\n%s", user_prompt)
        except Exception:
            pass

        import asyncio
        loop = asyncio.get_event_loop()

        if provider == "genai":
            logger.info("Calling GenAI (Google) using model %s...", model_name)
            # Import google genai dynamically to avoid startup dependency requirements if not used
            from google import genai
            from google.genai import types
            
            # If api_key is still empty, fallback to environment variable
            if not api_key:
                api_key = os.getenv("GOOGLE_API_KEY")
                
            client = genai.Client(api_key=api_key)
            
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                top_p=1.0,
                max_output_tokens=max_tokens,
                top_k=40,
                response_mime_type="application/json", 
                # === THÊM ĐOẠN NÀY ĐỂ TẮT THINKING (TIẾT KIỆM TOKENS) ===
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0  # Set budget bằng 0 để tắt hẳn chế độ suy luận ngầm
                ),

                # === QUAN TRỌNG: TẮT AFC ===
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,          # Tắt AFC
                    # max_remote_calls=0   # hoặc set = 0
                ),
                
                # Tùy chọn thêm
                safety_settings = [
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, 
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, 
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, 
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                # QUAN TRỌNG NHẤT: Chặn đứng việc chặt cụt tin đồn y tế/khoa học/5G
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, 
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
            ]

            )
            
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])]

            def _summarize_genai_response(response) -> str:
                candidate_summaries = []
                try:
                    for index, candidate in enumerate(getattr(response, "candidates", []) or [], start=1):
                        finish_reason = getattr(candidate, "finish_reason", None)
                        candidate_summaries.append(
                            f"candidate[{index}].finish_reason={finish_reason}"
                        )
                except Exception as summary_err:
                    candidate_summaries.append(f"candidate_summary_error={summary_err}")

                summary_text = "; ".join(candidate_summaries) if candidate_summaries else "no_candidates_returned"
                return (
                    f"model={model_name}, max_output_tokens={max_tokens}, "
                    f"system_prompt_chars={len(system_prompt)}, user_prompt_chars={len(user_prompt)}, "
                    f"response_has_text={response.text is not None}, {summary_text}"
                )
            
            def _sync_call_genai():
                response = client.models.generate_content(
                    model=model_name,
                    config=config,
                    contents=contents
                )
                logger.info("GenAI response details: %s", response)
                if response.text is None:
                    err_msg = "Google GenAI trả về phản hồi rỗng (None)."
                    try:
                        if response.candidates and len(response.candidates) > 0:
                            finish_reasons = []
                            for candidate in response.candidates:
                                finish_reasons.append(str(getattr(candidate, "finish_reason", None)))
                            err_msg += f" Lý do dừng: {', '.join(finish_reasons)}."
                    except Exception:
                        pass
                    logger.error(
                        "GenAI empty response details: %s",
                        _summarize_genai_response(response)
                    )
                    raise ValueError(err_msg)
                return response.text.strip()
                
            try:
                content = await loop.run_in_executor(None, _sync_call_genai)
                
                # Log raw response for analysis when enabled
                try:
                    if settings.ENABLE_ANALYSIS_LOG:
                        preview = content if len(content) <= 2000 else content[:2000] + "...[truncated]"
                        logger.debug("LLM raw response (truncated):\n%s", preview)
                except Exception:
                    pass
                
                return content
            except Exception as e:
                logger.exception("Error calling GenAI LLM: %s", e)
                raise e
        else:
            logger.info("Calling OpenAI LLM at %s using model %s...", base_url, model_name)
            client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": max_tokens,
                "extra_body": {
                    "top_k": 1,
                    "presence_penalty": 0.0,
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                }
            }
            
            def _sync_call_openai():
                response = client.chat.completions.create(**payload)
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI/Tương thích trả về phản hồi rỗng (None).")
                return content.strip()
                
            try:
                content = await loop.run_in_executor(None, _sync_call_openai)
                
                # Log raw response for analysis when enabled
                try:
                    if settings.ENABLE_ANALYSIS_LOG:
                        preview = content if len(content) <= 2000 else content[:2000] + "...[truncated]"
                        logger.debug("LLM raw response (truncated):\n%s", preview)
                except Exception:
                    pass
                
                return content
            except Exception as e:
                logger.exception("Error calling OpenAI LLM: %s", e)
                raise e

    def parse_llm_json_response(self, raw_response: str) -> tuple[int, str]:
        """
        Parse JSON response from LLM: {"label": "Thật/Giả", "explanation": "..."}
        Returns:
            label: 0 for Real (Thật), 1 for Fake (Giả)
            explanation: str
        """
        clean_text = raw_response.replace("```json", "").replace("```", "").strip()
        # Find JSON boundaries
        import re
        match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(0)
            
        try:
            data = json.loads(clean_text)
            label_text = str(data.get("label", "")).strip().lower()
            explanation = str(data.get("explanation", "")).strip()
            
            # Match label text
            if "thật" in label_text or "co" in label_text or "có" in label_text:
                return 0, explanation
            elif "giả" in label_text or "khong" in label_text or "không" in label_text:
                return 1, explanation
            else:
                # Fallback matching
                if 0 == parse_simple_label(label_text):
                    return 0, explanation
                return 1, explanation
        except Exception as e:
            logger.error("Error parsing JSON response: %s. Raw response was: %s", e, raw_response)
            # Regex fallback
            label = 1
            if "thật" in clean_text.lower() and "giả" not in clean_text.lower():
                label = 0
            elif "giả" in clean_text.lower() and "thật" not in clean_text.lower():
                label = 1
                
            return label, f"Không thể phân tích phản hồi JSON hợp lệ từ LLM. Phản hồi thô: {raw_response}"

def parse_simple_label(text: str) -> int:
    text = text.lower()
    if "thật" in text or "chính xác" in text or "đúng" in text or "xác thực" in text:
        return 0
    return 1

# Global singleton instance
llm_service = LLMService()
