from pathlib import Path

import numpy as np
import pandas as pd

from core.config import settings

WINDOW = 252


def compute_performance(prices_dir: str = settings.PRICES_DIR) -> pd.DataFrame:
    """Trailing one-year return, volatility, max drawdown and Sharpe ratio per symbol."""
    prices = pd.read_csv(Path(prices_dir) / "prices.csv", index_col="date", parse_dates=True)
    rf_daily = pd.read_csv(Path(prices_dir) / "risk_free.csv", index_col="date", parse_dates=True)["daily"]

    window = prices.iloc[-(WINDOW + 1) :]
    returns = (window / window.shift(1) - 1).iloc[1:]
    volatility = returns.std() * np.sqrt(WINDOW)
    excess = returns.sub(rf_daily.reindex(returns.index), axis=0)

    perf = pd.DataFrame(
        {
            "return_1y": window.ffill().iloc[-1] / window.bfill().iloc[0] - 1,
            "volatility_1y": volatility,
            "max_drawdown_1y": (window / window.cummax() - 1).min(),
            "sharpe_1y": excess.mean() * WINDOW / volatility,
        }
    )
    perf.loc[returns.notna().sum() < WINDOW * 0.9] = np.nan
    perf["as_of"] = window.index[-1].date()
    perf.index.name = "symbol"
    return perf.reset_index()
