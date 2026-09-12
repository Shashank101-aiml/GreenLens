import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from nlp.text_features import THEMES, analyze

CONTROVERSY_ORDER = ["None", "Low", "Moderate", "Significant", "High", "Severe"]
RISK_ORDER = ["Negligible", "Low", "Medium", "High", "Severe"]
RISK_UPPER_BOUNDS = [10, 20, 30, 40]
TEXT_COL = "clean_text"
NUMERIC_COLS = [f"theme_{t}" for t in THEMES] + ["n_countries", "log_employees", "controversy_rank"]


def risk_level(total: float) -> str:
    # Sustainalytics bands. The dataset's ESG Risk Level is this bucketing of the total score (up to rounding).
    for upper, label in zip(RISK_UPPER_BOUNDS, RISK_ORDER):
        if total < upper:
            return label
    return RISK_ORDER[-1]


def build_features(df: pd.DataFrame, nlp_rows: list[dict] | None = None) -> pd.DataFrame:
    """Model inputs from raw company fields: description, sector, employees, controversy_level."""
    if nlp_rows is None:
        nlp_rows = analyze(df["description"].fillna("").tolist())
    feats = pd.DataFrame(index=df.index)
    feats[TEXT_COL] = [r["clean_text"] for r in nlp_rows]
    for theme in THEMES:
        feats[f"theme_{theme}"] = [r["themes"][theme] for r in nlp_rows]
    feats["n_countries"] = [r["n_countries"] for r in nlp_rows]
    feats["sector"] = df["sector"].fillna("Unknown").to_numpy()
    employees = pd.to_numeric(df["employees"], errors="coerce").astype(float)
    feats["log_employees"] = np.log1p(employees).to_numpy()
    ranks = {level: i for i, level in enumerate(CONTROVERSY_ORDER)}
    feats["controversy_rank"] = df["controversy_level"].map(ranks).astype(float).to_numpy()
    return feats


class SectorMeanRegressor(BaseEstimator, RegressorMixin):
    """Baseline: predict the training-set average score of the company's sector."""

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.means_ = y.groupby(X["sector"]).mean().to_dict()
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame):
        return X["sector"].map(self.means_).fillna(self.global_mean_).to_numpy(dtype=float)
