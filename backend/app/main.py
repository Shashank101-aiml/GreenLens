"""FastAPI backend for the ESG Performance Analytics platform."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from api.routes import analytics, companies, model
from core.config import settings
from db.database import engine
from services.model_service import ModelNotTrained, get_bundle

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"]
)
app.include_router(companies.router, prefix="/companies", tags=["Companies"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
app.include_router(model.router, prefix="/model", tags=["Model"])


@app.exception_handler(OperationalError)
def database_unavailable(_: Request, exc: OperationalError) -> JSONResponse:
    return JSONResponse(
        status_code=503, content={"detail": "Database unavailable. Is PostgreSQL running (docker compose up -d db)?"}
    )


@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        database = "ok"
    except OperationalError:
        database = "unavailable"
    try:
        model_version = get_bundle()["version"]
    except ModelNotTrained:
        model_version = None
    status = "ok" if database == "ok" and model_version else "degraded"
    return {"status": status, "database": database, "model_version": model_version}
