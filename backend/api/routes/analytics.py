from fastapi import APIRouter

from services import company_service

router = APIRouter()


@router.get("/overview")
def overview():
    return company_service.overview()


@router.get("/sectors")
def sectors():
    return company_service.sector_summary()
