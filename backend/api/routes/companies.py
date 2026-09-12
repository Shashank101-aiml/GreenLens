from fastapi import APIRouter, HTTPException, Query

from services import company_service, report_service

router = APIRouter()


@router.get("")
def list_companies(sector: str | None = None, q: str | None = Query(default=None, max_length=50)):
    return company_service.list_companies(sector=sector, q=q)


@router.get("/{symbol}")
def get_company(symbol: str):
    company = company_service.get_company(symbol)
    if company is None:
        raise HTTPException(status_code=404, detail=f"Unknown symbol '{symbol}'.")
    return company


@router.post("/{symbol}/report")
def company_report(symbol: str, refresh: bool = False):
    try:
        result = report_service.get_report(symbol, refresh=refresh)
    except report_service.ReportUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown symbol '{symbol}'.")
    return result
