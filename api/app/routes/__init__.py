from fastapi import APIRouter

from app.routes import experiments, health, models, projects, settings

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(projects.router)
api_router.include_router(models.router)
api_router.include_router(experiments.router)
api_router.include_router(settings.router)
