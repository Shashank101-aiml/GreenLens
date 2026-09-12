import hashlib
import json
from datetime import datetime, timezone

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.config import settings
from db.database import ai_reports, engine
from services.company_service import get_company, sector_summary

SYSTEM_INSTRUCTION = """You are an ESG analyst writing a short briefing for an investment committee.
Use ONLY the facts in the JSON you are given. Never invent numbers, events, controversies or sources.
If a fact is missing (null), say it is not available instead of guessing.
Sustainalytics ESG risk scores measure unmanaged ESG risk: LOWER is better.
Model estimates come from an ML model trained on company descriptions; call them estimates and compare
them with the rated scores when both exist. Keep each field concise and specific to this company."""


class ESGReport(BaseModel):
    summary: str = Field(description="Two or three sentences on the company's overall ESG risk profile.")
    environment: str = Field(description="Environmental risk commentary, citing the given scores.")
    social: str = Field(description="Social risk commentary, citing the given scores.")
    governance: str = Field(description="Governance risk commentary, citing the given scores.")
    key_risks: list[str] = Field(description="Up to four specific risks supported by the facts.")
    strengths: list[str] = Field(description="Up to four strengths supported by the facts.")
    data_caveats: list[str] = Field(description="Limitations of the data behind this report.")


class ReportUnavailable(RuntimeError):
    pass


def build_facts(company: dict, sector: dict | None) -> dict:
    themes = company.get("themes") or {}
    note = None
    if company.get("out_of_fold") is True:
        note = "Out-of-fold estimate: the model never saw this company during training."
    elif company.get("out_of_fold") is False:
        note = "Estimate for a company without a published rating."
    return {
        "company": {k: company.get(k) for k in ("symbol", "name", "sector", "industry", "employees")},
        "rated_esg_risk_sustainalytics": {
            "total": company.get("esg_total"),
            "environment": company.get("esg_environment"),
            "social": company.get("esg_social"),
            "governance": company.get("esg_governance"),
            "risk_level": company.get("risk_level"),
            "percentile": company.get("risk_percentile"),
            "controversy_level": company.get("controversy_level"),
        },
        "model_estimate": {
            "total": company.get("pred_total"),
            "environment": company.get("pred_environment"),
            "social": company.get("pred_social"),
            "governance": company.get("pred_governance"),
            "risk_level": company.get("pred_level"),
            "note": note,
        },
        "sector_average_esg_risk": sector,
        "stock_performance_1y": {k: company.get(k) for k in ("return_1y", "volatility_1y", "max_drawdown_1y", "sharpe_1y", "as_of")},
        "description_themes_per_100_tokens": {k: v for k, v in sorted(themes.items(), key=lambda kv: -kv[1]) if v > 0},
    }


def generate_report(facts: dict) -> ESGReport:
    if not settings.GEMINI_API_KEY:
        raise ReportUnavailable("GEMINI_API_KEY is not set. Add it to backend/.env and restart the API.")
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents="Write the ESG briefing for this company.\n\nFACTS:\n" + json.dumps(facts, indent=2, default=str),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=ESGReport,
                temperature=0.2,
            ),
        )
    except genai_errors.APIError as exc:
        raise ReportUnavailable(f"Gemini API error: {exc}") from exc
    if response.parsed is None:
        raise ReportUnavailable("Gemini returned a response that did not match the report schema.")
    return response.parsed


def get_report(symbol: str, refresh: bool = False) -> dict | None:
    company = get_company(symbol)
    if company is None:
        return None
    sector = next((s for s in sector_summary() if s["sector"] == company["sector"]), None)
    facts = build_facts(company, sector)
    # Same facts + same model -> reuse the stored report; any data change produces a new hash and a fresh report.
    input_hash = hashlib.sha256(json.dumps(facts, sort_keys=True, default=str).encode()).hexdigest()
    model = settings.GEMINI_MODEL
    result = {"symbol": company["symbol"], "model": model, "facts": facts}

    if not refresh:
        with engine.connect() as conn:
            cached = conn.execute(
                select(ai_reports.c.report, ai_reports.c.created_at).where(
                    ai_reports.c.symbol == company["symbol"],
                    ai_reports.c.model == model,
                    ai_reports.c.input_hash == input_hash,
                )
            ).mappings().first()
        if cached:
            return {**result, "cached": True, "created_at": cached["created_at"], "report": cached["report"]}

    report = generate_report(facts).model_dump()
    stmt = pg_insert(ai_reports).values(symbol=company["symbol"], model=model, input_hash=input_hash, report=report)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "model", "input_hash"], set_={"report": stmt.excluded.report, "created_at": func.now()}
    )
    with engine.begin() as conn:
        conn.execute(stmt)
    return {**result, "cached": False, "created_at": datetime.now(timezone.utc), "report": report}
