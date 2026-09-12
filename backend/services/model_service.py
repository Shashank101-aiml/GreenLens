from functools import lru_cache

import joblib
import pandas as pd

from core.config import settings
from ml.features import build_features, risk_level
from nlp.text_features import analyze

PILLARS = {"environment": "esg_environment", "social": "esg_social", "governance": "esg_governance"}


class ModelNotTrained(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_bundle() -> dict:
    try:
        return joblib.load(settings.MODEL_PATH)
    except FileNotFoundError as exc:
        raise ModelNotTrained("No trained model found. Run `python -m ml.train` in backend/.") from exc


def model_summary() -> dict:
    bundle = get_bundle()
    oof = bundle["oof"]
    return {
        "version": bundle["version"],
        "trained_at": bundle["trained_at"],
        "n_train": bundle["n_train"],
        "cv": bundle["cv"],
        "model_names": bundle["model_names"],
        "metrics": bundle["metrics"],
        "total_metrics": bundle["total_metrics"],
        "top_terms": bundle["top_terms"],
        "lexicon": bundle["lexicon"],
        "oof": oof.astype(object).where(oof.notna(), None).to_dict("records"),
    }


def predict(description: str, sector: str, employees: int | None, controversy_level: str | None) -> dict:
    bundle = get_bundle()
    nlp_rows = analyze([description])
    raw = pd.DataFrame(
        [{"description": description, "sector": sector, "employees": employees, "controversy_level": controversy_level}]
    )
    X = build_features(raw, nlp_rows)
    pillars = {key: round(max(float(bundle["pipelines"][col].predict(X)[0]), 0.0), 2) for key, col in PILLARS.items()}
    total = round(sum(pillars.values()), 2)
    return {
        "pillars": pillars,
        "total": total,
        "risk_level": risk_level(total),
        "themes": nlp_rows[0]["themes"],
        "n_countries": nlp_rows[0]["n_countries"],
        "model_version": bundle["version"],
    }
