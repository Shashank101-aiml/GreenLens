"""Load companies, NLP features, 1-year stock performance and model predictions into PostgreSQL.

Usage (from backend/, after `python -m ml.train`): python -m scripts.load_db
"""
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import Table, delete

from core.config import settings
from db.database import companies, engine, init_db, nlp_features, performance, predictions
from ml.dataset import load_companies
from nlp.text_features import analyze
from services.performance import compute_performance

log = logging.getLogger("load_db")


def records(df: pd.DataFrame, table: Table) -> list[dict]:
    frame = df[[c.name for c in table.columns if c.name in df.columns]]
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if not Path(settings.MODEL_PATH).exists():
        log.error("No model at %s. Run `python -m ml.train` first.", settings.MODEL_PATH)
        return 1

    init_db()
    df = load_companies()
    nlp_rows = analyze(df["description"].fillna("").tolist())
    nlp_df = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "clean_text": [r["clean_text"] for r in nlp_rows],
            "themes": [r["themes"] for r in nlp_rows],
            "n_countries": [r["n_countries"] for r in nlp_rows],
        }
    )
    if Path(settings.PRICES_DIR, "prices.csv").exists():
        perf = compute_performance()
    else:
        log.warning("No prices in %s; run `python -m scripts.fetch_market_data` for performance KPIs.", settings.PRICES_DIR)
        perf = pd.DataFrame()
    bundle = joblib.load(settings.MODEL_PATH)
    preds = bundle["predictions"].assign(model_version=bundle["version"])

    with engine.begin() as conn:
        for table, frame in ((companies, df), (nlp_features, nlp_df), (performance, perf), (predictions, preds)):
            conn.execute(delete(table))
            if not frame.empty:
                conn.execute(table.insert(), records(frame, table))
            log.info("%-13s %4d rows", table.name, len(frame))
    return 0


if __name__ == "__main__":
    sys.exit(main())
