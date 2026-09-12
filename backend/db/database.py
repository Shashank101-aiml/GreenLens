from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)

from core.config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
metadata = MetaData()

companies = Table(
    "companies",
    metadata,
    Column("symbol", String(10), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("sector", String(60)),
    Column("industry", String(100)),
    Column("employees", Integer),
    Column("description", Text),
    Column("esg_total", Float),
    Column("esg_environment", Float),
    Column("esg_social", Float),
    Column("esg_governance", Float),
    Column("controversy_level", String(20)),
    Column("controversy_score", Float),
    Column("risk_level", String(20)),
    Column("risk_percentile", Float),
)

nlp_features = Table(
    "nlp_features",
    metadata,
    Column("symbol", String(10), primary_key=True),
    Column("clean_text", Text),
    Column("themes", JSON),
    Column("n_countries", Integer),
)

performance = Table(
    "performance",
    metadata,
    Column("symbol", String(10), primary_key=True),
    Column("as_of", Date),
    Column("return_1y", Float),
    Column("volatility_1y", Float),
    Column("max_drawdown_1y", Float),
    Column("sharpe_1y", Float),
)

predictions = Table(
    "predictions",
    metadata,
    Column("symbol", String(10), primary_key=True),
    Column("model_version", String(40), nullable=False),
    Column("pred_environment", Float),
    Column("pred_social", Float),
    Column("pred_governance", Float),
    Column("pred_total", Float),
    Column("pred_level", String(20)),
    # True when the prediction came from cross-validation, i.e. the model never saw this company.
    Column("out_of_fold", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

ai_reports = Table(
    "ai_reports",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(10), nullable=False),
    Column("model", String(60), nullable=False),
    Column("input_hash", String(64), nullable=False),
    Column("report", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    UniqueConstraint("symbol", "model", "input_hash"),
)


def init_db() -> None:
    metadata.create_all(engine)
