from fastapi import APIRouter, HTTPException

from schemas.api import PredictRequest
from services import model_service

router = APIRouter()


@router.get("/summary")
def summary():
    try:
        return model_service.model_summary()
    except model_service.ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/predict")
def predict(req: PredictRequest):
    try:
        return model_service.predict(req.description, req.sector, req.employees, req.controversy_level)
    except model_service.ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
