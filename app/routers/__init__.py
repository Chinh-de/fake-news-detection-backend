from app.routers.predict import router as predict_router
from app.routers.analyze import router as analyze_router
from app.routers.dashboard import router as dashboard_router
from app.routers.history import router as history_router
from app.routers.admin import router as admin_router
from app.routers.health import router as health_router

__all__ = [
    "predict_router",
    "analyze_router",
    "dashboard_router",
    "history_router",
    "admin_router",
    "health_router"
]
