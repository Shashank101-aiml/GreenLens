"""Train one regressor per ESG pillar (Environment, Social, Governance) and evaluate with 5-fold CV.

Usage (from backend/): python -m ml.train
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from core.config import settings  # noqa: E402
from ml.dataset import PILLARS, load_companies  # noqa: E402
from ml.features import NUMERIC_COLS, RISK_ORDER, TEXT_COL, SectorMeanRegressor, build_features, risk_level  # noqa: E402
from nlp.text_features import analyze, get_lexicon  # noqa: E402

log = logging.getLogger("train")
CV = KFold(n_splits=5, shuffle=True, random_state=42)
BASELINE = "sector_mean_baseline"
PRED_COLS = {p: "pred_" + p.removeprefix("esg_") for p in PILLARS}


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True, max_features=3000), TEXT_COL),
            ("sector", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC_COLS),
        ]
    )


CANDIDATES = {
    BASELINE: lambda: SectorMeanRegressor(),
    "ridge": lambda: Pipeline([("prep", make_preprocessor()), ("model", RidgeCV(alphas=np.logspace(-2, 3, 11)))]),
    "random_forest": lambda: Pipeline(
        [
            ("prep", make_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.3, random_state=42, n_jobs=-1)),
        ]
    ),
    "gradient_boosting": lambda: Pipeline(
        [
            ("prep", make_preprocessor()),
            ("model", GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)),
        ]
    ),
}


def score(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }


def level_scores(pred_total: pd.Series, true_level: pd.Series) -> dict:
    rank = {level: i for i, level in enumerate(RISK_ORDER)}
    pred_level = pred_total.map(risk_level)
    return {
        "level_accuracy": round(float((pred_level == true_level).mean()), 3),
        "level_within_one": round(float((pred_level.map(rank) - true_level.map(rank)).abs().le(1).mean()), 3),
    }


def top_text_terms(ridge: Pipeline, n: int = 12) -> dict:
    names = ridge.named_steps["prep"].get_feature_names_out()
    coef = ridge.named_steps["model"].coef_
    text_idx = sorted((i for i, name in enumerate(names) if name.startswith("text__")), key=lambda i: coef[i])

    def fmt(idx):
        return [{"term": names[i].removeprefix("text__"), "weight": round(float(coef[i]), 3)} for i in idx]

    return {"raises_risk": fmt(text_idx[::-1][:n]), "lowers_risk": fmt(text_idx[:n])}


def save_diagnostics(actual: pd.DataFrame, oof: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, len(PILLARS), figsize=(15, 4.5))
    for ax, pillar in zip(axes, PILLARS):
        hi = float(max(actual[pillar].max(), oof[pillar].max())) * 1.05
        ax.scatter(actual[pillar], oof[pillar], s=12, alpha=0.6)
        ax.plot([0, hi], [0, hi], "k--", lw=1)
        ax.set(title=pillar.removeprefix("esg_").title(), xlabel="Rated risk score", ylabel="Out-of-fold prediction", xlim=(0, hi), ylim=(0, hi))
    fig.tight_layout()
    fig.savefig(out / "oof_predictions.png", dpi=120)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    out = Path(settings.ARTIFACTS_DIR)
    out.mkdir(parents=True, exist_ok=True)

    df = load_companies()
    log.info("Running spaCy over %d company descriptions", len(df))
    X_all = build_features(df, analyze(df["description"].fillna("").tolist()))
    rated = df[PILLARS].notna().all(axis=1)
    train_idx = df[rated].drop_duplicates("description").index
    X, Y = X_all.loc[train_idx], df.loc[train_idx, PILLARS]
    log.info("Training on %d rated companies", len(X))

    metrics, best, pipelines, top_terms = {}, {}, {}, {}
    oof = pd.DataFrame(index=X.index)
    baseline_oof = pd.DataFrame(index=X.index)
    for pillar in PILLARS:
        y = Y[pillar].to_numpy()
        cv_preds = {name: np.clip(cross_val_predict(factory(), X, y, cv=CV), 0, None) for name, factory in CANDIDATES.items()}
        metrics[pillar] = {name: score(y, pred) for name, pred in cv_preds.items()}
        for name, m in metrics[pillar].items():
            log.info("%-16s %-21s MAE %6.3f  RMSE %6.3f  R2 %6.3f", pillar, name, m["mae"], m["rmse"], m["r2"])
        pillar_metrics = metrics[pillar]
        best[pillar] = min(pillar_metrics, key=lambda name: pillar_metrics[name]["mae"])
        oof[pillar] = cv_preds[best[pillar]]
        baseline_oof[pillar] = cv_preds[BASELINE]
        pipelines[pillar] = CANDIDATES[best[pillar]]().fit(X, y)
        top_terms[pillar] = top_text_terms(CANDIDATES["ridge"]().fit(X, y))

    actual_total = df.loc[train_idx, "esg_total"]
    true_level = df.loc[train_idx, "risk_level"]
    total_metrics = {
        "model": {**score(actual_total, oof.sum(axis=1)), **level_scores(oof.sum(axis=1), true_level)},
        "baseline": {**score(actual_total, baseline_oof.sum(axis=1)), **level_scores(baseline_oof.sum(axis=1), true_level)},
    }

    predictions = pd.DataFrame({PRED_COLS[p]: np.clip(pipelines[p].predict(X_all), 0, None) for p in PILLARS}, index=df.index)
    # Rated companies get their out-of-fold prediction, so every score shown for them is from a model that never saw them.
    predictions.loc[oof.index, list(PRED_COLS.values())] = oof[PILLARS].to_numpy()
    predictions["out_of_fold"] = df.index.isin(oof.index)
    predictions["pred_total"] = predictions[list(PRED_COLS.values())].sum(axis=1)
    predictions["pred_level"] = predictions["pred_total"].map(risk_level)
    predictions.insert(0, "symbol", df["symbol"])
    predictions = predictions.round(2)

    oof_frame = pd.concat([df.loc[train_idx, ["symbol", "name", "sector"]], Y, oof.rename(columns=PRED_COLS).round(2)], axis=1)
    bundle = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": len(X),
        "cv": "5-fold KFold (shuffled, seed 42); out-of-fold predictions",
        "model_names": best,
        "metrics": metrics,
        "total_metrics": total_metrics,
        "top_terms": top_terms,
        "lexicon": get_lexicon(),
        "pipelines": pipelines,
        "predictions": predictions,
        "oof": oof_frame,
    }
    joblib.dump(bundle, settings.MODEL_PATH)
    summary = {k: v for k, v in bundle.items() if k not in ("pipelines", "predictions", "oof")}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_diagnostics(Y, oof, out)

    log.info("Best model per pillar: %s", best)
    log.info("Total ESG risk (sum of pillars): model %s | baseline %s", total_metrics["model"], total_metrics["baseline"])
    log.info("Saved %s, metrics.json and oof_predictions.png", settings.MODEL_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
