from sqlalchemy import case, func, select

from db.database import companies, engine, nlp_features, performance, predictions

BENCHMARK = "SPY"
PRED_COLS = [
    predictions.c.pred_environment,
    predictions.c.pred_social,
    predictions.c.pred_governance,
    predictions.c.pred_total,
    predictions.c.pred_level,
    predictions.c.out_of_fold,
    predictions.c.model_version,
]
PERF_COLS = [
    performance.c.return_1y,
    performance.c.volatility_1y,
    performance.c.max_drawdown_1y,
    performance.c.sharpe_1y,
    performance.c.as_of,
]
LIST_COLS = [
    companies.c.symbol,
    companies.c.name,
    companies.c.sector,
    companies.c.industry,
    companies.c.employees,
    companies.c.esg_total,
    companies.c.esg_environment,
    companies.c.esg_social,
    companies.c.esg_governance,
    companies.c.risk_level,
    companies.c.controversy_level,
]
JOINED = companies.outerjoin(predictions, predictions.c.symbol == companies.c.symbol).outerjoin(
    performance, performance.c.symbol == companies.c.symbol
)


def _round(value, digits: int = 2):
    return None if value is None else round(float(value), digits)


def list_companies(sector: str | None = None, q: str | None = None) -> list[dict]:
    stmt = select(*LIST_COLS, *PRED_COLS, *PERF_COLS).select_from(JOINED).order_by(companies.c.symbol)
    if sector:
        stmt = stmt.where(companies.c.sector == sector)
    if q:
        pattern = f"%{q.strip()}%"
        stmt = stmt.where(companies.c.name.ilike(pattern) | companies.c.symbol.ilike(pattern))
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(stmt).mappings()]


def get_company(symbol: str) -> dict | None:
    stmt = (
        select(companies, *PRED_COLS, *PERF_COLS, nlp_features.c.themes, nlp_features.c.n_countries)
        .select_from(JOINED.outerjoin(nlp_features, nlp_features.c.symbol == companies.c.symbol))
        .where(companies.c.symbol == symbol.strip().upper())
    )
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
    return dict(row) if row else None


def sector_summary() -> list[dict]:
    stmt = (
        select(
            companies.c.sector,
            func.count().label("n_companies"),
            func.count(companies.c.esg_total).label("n_rated"),
            func.avg(companies.c.esg_total).label("avg_total"),
            func.avg(companies.c.esg_environment).label("avg_environment"),
            func.avg(companies.c.esg_social).label("avg_social"),
            func.avg(companies.c.esg_governance).label("avg_governance"),
            func.avg(performance.c.return_1y).label("avg_return_1y"),
        )
        .select_from(companies.outerjoin(performance, performance.c.symbol == companies.c.symbol))
        .where(companies.c.sector.is_not(None))
        .group_by(companies.c.sector)
        .order_by(func.avg(companies.c.esg_total).desc().nulls_last())
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [{k: _round(v, 4) if k.startswith("avg_") else v for k, v in row.items()} for row in rows]


def overview() -> dict:
    high_or_severe = case((companies.c.risk_level.in_(("High", "Severe")), 1.0), else_=0.0)
    with engine.connect() as conn:
        kpi = conn.execute(
            select(
                func.count().label("n_companies"),
                func.count(companies.c.esg_total).label("n_rated"),
                func.avg(companies.c.esg_total).label("avg_total"),
                func.avg(high_or_severe).filter(companies.c.risk_level.is_not(None)).label("share_high_severe"),
            ).select_from(companies)
        ).mappings().one()
        levels = conn.execute(
            select(companies.c.risk_level, func.count().label("n"))
            .where(companies.c.risk_level.is_not(None))
            .group_by(companies.c.risk_level)
        ).mappings().all()
        benchmark = conn.execute(select(*PERF_COLS).where(performance.c.symbol == BENCHMARK)).mappings().first()
    return {
        "n_companies": kpi["n_companies"],
        "n_rated": kpi["n_rated"],
        "avg_total": _round(kpi["avg_total"]),
        "share_high_severe": _round(kpi["share_high_severe"], 4),
        "risk_levels": {row["risk_level"]: row["n"] for row in levels},
        "benchmark": {"symbol": BENCHMARK, **dict(benchmark)} if benchmark else None,
    }
