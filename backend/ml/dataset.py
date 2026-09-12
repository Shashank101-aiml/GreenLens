import pandas as pd

from core.config import settings

COLUMNS = {
    "Symbol": "symbol",
    "Name": "name",
    "Sector": "sector",
    "Industry": "industry",
    "Full Time Employees": "employees",
    "Description": "description",
    "Total ESG Risk score": "esg_total",
    "Environment Risk Score": "esg_environment",
    "Social Risk Score": "esg_social",
    "Governance Risk Score": "esg_governance",
    "Controversy Level": "controversy_level",
    "Controversy Score": "controversy_score",
    "ESG Risk Level": "risk_level",
    "ESG Risk Percentile": "risk_percentile",
}
PILLARS = ["esg_environment", "esg_social", "esg_governance"]


def load_companies(path: str = settings.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=list(COLUMNS)).rename(columns=COLUMNS)
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["employees"] = pd.to_numeric(
        df["employees"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    ).astype("Int64")
    df["controversy_level"] = df["controversy_level"].str.replace("Controversy Level", "", regex=False).str.strip()
    df["risk_percentile"] = pd.to_numeric(
        df["risk_percentile"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce"
    )
    return df[list(COLUMNS.values())]
