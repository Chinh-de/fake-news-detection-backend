from pydantic import BaseModel, Field

from typing import Optional

class AdminLogin(BaseModel):
    username: str = Field(..., example="admin")
    password: str = Field(..., example="adminpassword")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class LLMConfigUpdate(BaseModel):
    llm_provider: Optional[str] = Field(default="openai", example="openai")
    llm_endpoint: Optional[str] = Field(default=None, example="https://chinh-de--qwen3-4b-t4-production-v2-serve.modal.run/v1")
    llm_api_key: Optional[str] = Field(default=None, example="EMPTY")
    llm_model: Optional[str] = Field(default=None, example="Qwen/Qwen3-4B-AWQ")


class SystemConfigResponse(BaseModel):
    llm_provider: str
    llm_endpoint: str
    llm_model: str
    slm_model_path: str
    embedding_model_cache_dir: str
    seed_corpus_csv: str
    slm_repo_id: str


class ModelStatusResponse(BaseModel):
    model_name: str
    current_sha: str
    scheduled_time: Optional[str] = None
    scheduled_sha: Optional[str] = None
    is_updating: bool
    last_update_error: Optional[str] = None


class ModelUpdateRequest(BaseModel):
    target_time: Optional[str] = None # ISO string or null

