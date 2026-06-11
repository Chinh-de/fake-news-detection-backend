from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import get_db, SessionLocal
from app.models.config import SystemConfig
from app.schemas.admin import (
    AdminLogin, TokenResponse, LLMConfigUpdate, SystemConfigResponse,
    ModelStatusResponse, ModelUpdateRequest
)
from app.services.auth_service import auth_service
from app.core.security import get_current_admin
from app.config import settings
from app.services.slm_service import slm_service
from huggingface_hub import HfApi
import asyncio
from datetime import datetime, timezone
import logging

logger = logging.getLogger("admin_slm_router")

router = APIRouter(prefix="/admin", tags=["admin"])


def _default_llm_model(provider: str) -> str:
    if provider == "genai":
        return "gemini-3.1-flash-lite"
    return "Qwen/Qwen3-4B-AWQ"

@router.post("/login", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def login_admin(request: AdminLogin):
    """
    Đăng nhập tài khoản admin.
    Trả về token truy cập JWT.
    """
    # Check username and password against .env values
    if request.username != settings.ADMIN_USERNAME or not auth_service.verify_password(request.password, settings.ADMIN_PASSWORD):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tên đăng nhập hoặc mật khẩu không chính xác",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = auth_service.create_access_token(data={"sub": request.username})
    return TokenResponse(access_token=access_token)


@router.get("/config", response_model=SystemConfigResponse, status_code=status.HTTP_200_OK)
async def get_system_configuration(
    current_user: str = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Lấy thông tin cấu hình hệ thống hiện tại.
    Yêu cầu quyền Admin.
    """
    stmt = select(SystemConfig)
    res = await db.execute(stmt)
    configs = {cfg.key: cfg.value for cfg in res.scalars().all()}
    
    return SystemConfigResponse(
        llm_provider=configs.get("llm_provider", settings.LLM_PROVIDER),
        llm_endpoint=configs.get("llm_endpoint", settings.LLM_ENDPOINT),
        llm_model=configs.get("llm_model") or _default_llm_model(configs.get("llm_provider", settings.LLM_PROVIDER)),
        slm_model_path=settings.SLM_MODEL_PATH,
        embedding_model_cache_dir=settings.EMBEDDING_MODEL_CACHE_DIR,
        seed_corpus_csv=settings.SEED_CORPUS_CSV,
        slm_repo_id=settings.SLM_REPO_ID
    )


@router.post("/config", status_code=status.HTTP_200_OK)
async def update_llm_configuration(
    request: LLMConfigUpdate,
    current_user: str = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Cập nhật cấu hình LLM (provider, endpoint, model, api key).
    Lưu trực tiếp vào database để có hiệu lực ngay lập tức.
    Yêu cầu quyền Admin.
    """
    try:
        # Upsert configurations
        for key, value in [
            ("llm_provider", request.llm_provider),
            ("llm_endpoint", request.llm_endpoint),
            ("llm_api_key", request.llm_api_key),
            ("llm_model", request.llm_model)
        ]:
            if value is not None:
                stmt = select(SystemConfig).where(SystemConfig.key == key)
                res = await db.execute(stmt)
                cfg = res.scalars().first()
                if cfg:
                    cfg.value = value
                else:
                    db.add(SystemConfig(key=key, value=value))
                    
        await db.commit()
        return {"message": "Cấu hình hệ thống đã được cập nhật thành công"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi cập nhật cấu hình: {str(e)}"
        )


async def run_immediate_update(sha: str):
    async with SessionLocal() as db:
        try:
            await slm_service.hot_swap_model(db, sha)
        except Exception as e:
            logger.error("Immediate SLM hot-swap update failed: %s", e)


@router.get("/model-status", response_model=ModelStatusResponse, status_code=status.HTTP_200_OK)
async def get_model_status(
    current_user: str = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Lấy thông tin trạng thái mô hình SLM hiện tại, bao gồm cả lịch trình cập nhật.
    Yêu cầu quyền Admin.
    """
    res_time = await db.execute(select(SystemConfig).where(SystemConfig.key == "slm_update_schedule_time"))
    cfg_time = res_time.scalars().first()
    res_sha = await db.execute(select(SystemConfig).where(SystemConfig.key == "slm_update_schedule_sha"))
    cfg_sha = res_sha.scalars().first()
    
    return ModelStatusResponse(
        model_name=settings.SLM_REPO_ID,
        current_sha=slm_service.current_sha,
        scheduled_time=cfg_time.value if (cfg_time and cfg_time.value) else None,
        scheduled_sha=cfg_sha.value if (cfg_sha and cfg_sha.value) else None,
        is_updating=slm_service.is_updating,
        last_update_error=slm_service.last_update_error
    )


@router.post("/update-model", status_code=status.HTTP_200_OK)
async def update_model(
    request: ModelUpdateRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Cập nhật mô hình SLM.
    Nếu target_time là null hoặc trống, cập nhật ngay lập tức (hot-swap).
    Nếu target_time được thiết lập, lên lịch cập nhật (scheduled update).
    Yêu cầu quyền Admin.
    """
    # 1. Fetch latest SHA from Hugging Face
    try:
        loop = asyncio.get_event_loop()
        api = HfApi()
        info = await loop.run_in_executor(None, lambda: api.model_info(settings.SLM_REPO_ID))
        latest_sha = info.sha
    except Exception as e:
        logger.error("Failed to fetch model info from Hugging Face: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Không thể lấy thông tin mô hình từ Hugging Face: {str(e)}"
        )

    # 2. Check if immediate or scheduled
    if not request.target_time:
        if slm_service.is_updating:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mô hình đang được cập nhật ở chế độ nền, vui lòng đợi."
            )
        background_tasks.add_task(run_immediate_update, latest_sha)
        return {"message": "Đã bắt đầu cập nhật mô hình ngay lập tức ở chế độ nền.", "sha": latest_sha}
    
    else:
        # Validate time is in the future
        try:
            target_dt = datetime.fromisoformat(request.target_time.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc) if target_dt.tzinfo else datetime.now()
            if target_dt < now:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Thời gian hẹn giờ cập nhật phải ở tương lai."
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Định dạng thời gian không hợp lệ. Vui lòng sử dụng chuẩn ISO."
            )
            
        # Save scheduled info in database
        try:
            for key, val in [
                ("slm_update_schedule_time", request.target_time),
                ("slm_update_schedule_sha", latest_sha)
            ]:
                stmt = select(SystemConfig).where(SystemConfig.key == key)
                res = await db.execute(stmt)
                cfg = res.scalars().first()
                if cfg:
                    cfg.value = val
                else:
                    db.add(SystemConfig(key=key, value=val))
            await db.commit()
            return {"message": f"Đã đặt lịch cập nhật thành công vào {request.target_time}.", "sha": latest_sha}
        except Exception as e:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi khi đặt lịch cập nhật mô hình: {str(e)}"
            )


@router.post("/cancel-update", status_code=status.HTTP_200_OK)
async def cancel_update(
    current_user: str = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Hủy lịch cập nhật mô hình SLM đã hẹn giờ.
    Yêu cầu quyền Admin.
    """
    try:
        for key in ["slm_update_schedule_time", "slm_update_schedule_sha"]:
            stmt = select(SystemConfig).where(SystemConfig.key == key)
            res = await db.execute(stmt)
            cfg = res.scalars().first()
            if cfg:
                cfg.value = ""
        await db.commit()
        return {"message": "Đã hủy lịch hẹn cập nhật mô hình thành công."}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi hủy lịch cập nhật mô hình: {str(e)}"
        )


@router.get("/model-update-history", status_code=status.HTTP_200_OK)
async def get_model_update_history(
    current_user: str = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Lấy lịch sử cập nhật mô hình SLM (Thành công và Thất bại).
    Yêu cầu quyền Admin.
    """
    from app.models.config import SLMUpdateHistory
    try:
        stmt = select(SLMUpdateHistory).order_by(SLMUpdateHistory.triggered_at.desc())
        res = await db.execute(stmt)
        history_items = res.scalars().all()
        
        return [
            {
                "id": item.id,
                "version_sha": item.version_sha,
                "status": item.status,
                "error_message": item.error_message,
                "triggered_at": item.triggered_at.isoformat() if item.triggered_at else None,
                "completed_at": item.completed_at.isoformat() if item.completed_at else None
            }
            for item in history_items
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy lịch sử cập nhật mô hình: {str(e)}"
        )


@router.get("/mrcd-runs", status_code=status.HTTP_200_OK)
async def get_mrcd_runs_history(
    current_user: str = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Lấy lịch sử tất cả các lượt chạy MRCD.
    Yêu cầu quyền Admin.
    """
    from app.models.config import MRCDRunHistory
    try:
        stmt = select(MRCDRunHistory).order_by(MRCDRunHistory.triggered_at.desc())
        res = await db.execute(stmt)
        runs = res.scalars().all()
        return [
            {
                "id": run.id,
                "hf_commit_sha": run.hf_commit_sha,
                "total_samples": run.total_samples,
                "clean_count_r1": run.clean_count_r1,
                "clean_count_r2": run.clean_count_r2,
                "clean_count_r3": run.clean_count_r3,
                "noisy_count_final": run.noisy_count_final,
                "triggered_at": run.triggered_at.isoformat() if run.triggered_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None
            }
            for run in runs
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy lịch sử chạy MRCD: {str(e)}"
        )


@router.get("/mrcd-runs/{run_id}", status_code=status.HTTP_200_OK)
async def get_mrcd_run_detail(
    run_id: int,
    round: int = None,
    pool: str = None,
    page: int = 1,
    limit: int = 10,
    current_user: str = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Lấy chi tiết một lượt chạy MRCD bao gồm cả quá trình dịch chuyển các mẫu qua các vòng.
    Hỗ trợ lọc theo vòng, phân loại tập (clean/noisy) và phân trang dưới database.
    Yêu cầu quyền Admin.
    """
    from app.models.config import MRCDRunHistory, MRCDSampleRoundLog
    from app.models.prediction import PredictionRecord
    try:
        # 1. Fetch run info
        stmt_run = select(MRCDRunHistory).where(MRCDRunHistory.id == run_id)
        res_run = await db.execute(stmt_run)
        run = res_run.scalars().first()
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Không tìm thấy lượt chạy MRCD này"
            )

        # 2. Build subquery for filtered prediction_record_ids
        if round is not None:
            if pool == "clean":
                query_ids = (
                    select(MRCDSampleRoundLog.prediction_record_id)
                    .where(MRCDSampleRoundLog.run_id == run_id)
                    .where(MRCDSampleRoundLog.round_id == round)
                    .where(MRCDSampleRoundLog.status.like("clean%"))
                )
            elif pool == "noisy":
                query_ids = (
                    select(MRCDSampleRoundLog.prediction_record_id)
                    .where(MRCDSampleRoundLog.run_id == run_id)
                    .where(MRCDSampleRoundLog.round_id == round)
                    .where(MRCDSampleRoundLog.status.like("noisy%"))
                )
            else:
                query_ids = (
                    select(MRCDSampleRoundLog.prediction_record_id)
                    .where(MRCDSampleRoundLog.run_id == run_id)
                    .where(MRCDSampleRoundLog.round_id == round)
                )
        else:
            if pool == "clean":
                query_ids = (
                    select(MRCDSampleRoundLog.prediction_record_id)
                    .where(MRCDSampleRoundLog.run_id == run_id)
                    .where(MRCDSampleRoundLog.status.like("clean%"))
                    .distinct()
                )
            elif pool == "noisy":
                clean_ids_subquery = (
                    select(MRCDSampleRoundLog.prediction_record_id)
                    .where(MRCDSampleRoundLog.run_id == run_id)
                    .where(MRCDSampleRoundLog.status.like("clean%"))
                )
                query_ids = (
                    select(MRCDSampleRoundLog.prediction_record_id)
                    .where(MRCDSampleRoundLog.run_id == run_id)
                    .where(MRCDSampleRoundLog.prediction_record_id.notin_(clean_ids_subquery))
                    .distinct()
                )
            else:
                query_ids = (
                    select(MRCDSampleRoundLog.prediction_record_id)
                    .where(MRCDSampleRoundLog.run_id == run_id)
                    .distinct()
                )

        # Get total count of matched record IDs
        res_count = await db.execute(select(func.count()).select_from(query_ids.subquery()))
        total_count = res_count.scalar() or 0

        # Paginate the matched record IDs
        offset = (page - 1) * limit
        paginated_query = query_ids.order_by(MRCDSampleRoundLog.prediction_record_id).offset(offset).limit(limit)
        res_ids = await db.execute(paginated_query)
        matched_ids = [r[0] for r in res_ids.all()]

        # 3. Fetch related prediction records
        records = {}
        if matched_ids:
            stmt_rec = select(PredictionRecord).where(PredictionRecord.id.in_(matched_ids))
            res_rec = await db.execute(stmt_rec)
            for rec in res_rec.scalars().all():
                records[rec.id] = {
                    "id": rec.id,
                    "post_text": rec.post_text,
                    "final_slm_label": rec.slm_label,
                    "final_slm_conf": rec.slm_confidence,
                    "final_llm_label": rec.llm_label,
                    "wiki_evidence": rec.wiki_evidence,
                    "rag_evidence": rec.rag_evidence,
                    "fewshot_examples": rec.fewshot_examples
                }

        # 4. Fetch all round logs for matched record IDs
        logs = []
        if matched_ids:
            stmt_logs = (
                select(MRCDSampleRoundLog)
                .where(MRCDSampleRoundLog.run_id == run_id)
                .where(MRCDSampleRoundLog.prediction_record_id.in_(matched_ids))
                .order_by(MRCDSampleRoundLog.prediction_record_id, MRCDSampleRoundLog.round_id)
            )
            res_logs = await db.execute(stmt_logs)
            logs = res_logs.scalars().all()

        # 5. Group round logs by prediction record preserving matched_ids order
        sample_transitions = {}
        for rec_id in matched_ids:
            if rec_id in records:
                sample_transitions[rec_id] = {
                    "record": records[rec_id],
                    "rounds": []
                }

        for log in logs:
            rec_id = log.prediction_record_id
            if rec_id in sample_transitions:
                sample_transitions[rec_id]["rounds"].append({
                    "round_id": log.round_id,
                    "y_llm": log.y_llm,
                    "y_slm": log.y_slm,
                    "conf_slm": log.conf_slm,
                    "status": log.status,
                    "fewshot_examples": log.fewshot_examples,
                    "rag_evidence": log.rag_evidence,
                    "wiki_evidence": log.wiki_evidence
                })

        return {
            "run_info": {
                "id": run.id,
                "hf_commit_sha": run.hf_commit_sha,
                "total_samples": run.total_samples,
                "clean_count_r1": run.clean_count_r1,
                "clean_count_r2": run.clean_count_r2,
                "clean_count_r3": run.clean_count_r3,
                "noisy_count_final": run.noisy_count_final,
                "triggered_at": run.triggered_at.isoformat() if run.triggered_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None
            },
            "samples": list(sample_transitions.values()),
            "total_count": total_count,
            "page": page,
            "limit": limit
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy chi tiết lượt chạy MRCD: {str(e)}"
        )
