import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.fetch_market_data import (
    STALE_RUN_DAYS,
    TRADING_DAYS,
    align_risk_free,
    load_universe,
    quality_report,
    to_yahoo_symbol,
)


class TestFetchMarketData(unittest.TestCase):
    def test_to_yahoo_symbol(self):
        self.assertEqual(to_yahoo_symbol("BF.B"), "BF-B")
        self.assertEqual(to_yahoo_symbol(" aapl "), "AAPL")

    def test_load_universe_maps_yahoo_symbols_back_to_esg_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "universe.csv"
            pd.DataFrame({"Symbol": ["BF.B", "PARA", "AAPL", None]}).to_csv(path, index=False)
            mapping = load_universe(str(path))

        self.assertEqual(mapping, {"BF-B": "BF.B", "PSKY": "PARA", "AAPL": "AAPL", "SPY": "SPY"})

    def test_quality_report_flags(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        base = pd.Series(100 + np.arange(30) * 0.5, index=dates)
        prices = pd.DataFrame({name: base.copy() for name in ["OK", "SPLIT", "STALE", "LATE", "EARLY", "GAP"]})
        prices.iloc[15:, prices.columns.get_loc("SPLIT")] *= 0.4
        prices.iloc[10 : 10 + STALE_RUN_DAYS + 2, prices.columns.get_loc("STALE")] = 105.0
        prices.iloc[:10, prices.columns.get_loc("LATE")] = np.nan
        prices.iloc[-10:, prices.columns.get_loc("EARLY")] = np.nan
        prices.iloc[12:15, prices.columns.get_loc("GAP")] = np.nan

        flags = quality_report(prices, min_coverage=0.9)["issues"].str.split(";")

        self.assertEqual(flags["OK"], [""])
        self.assertEqual(flags["SPLIT"], ["extreme_move"])
        self.assertEqual(flags["STALE"], ["stale"])
        self.assertEqual(set(flags["LATE"]), {"low_coverage", "starts_late"})
        self.assertEqual(set(flags["EARLY"]), {"low_coverage", "ends_early"})
        self.assertEqual(flags["GAP"], ["internal_gaps"])

    def test_align_risk_free_carries_bond_holidays_forward(self):
        calendar = pd.DatetimeIndex(["2024-10-11", "2024-10-14", "2024-10-15"])  # 2024-10-14 is Columbus Day
        rf = pd.Series([4.60, 4.62], index=pd.DatetimeIndex(["2024-10-10", "2024-10-15"]))

        out = align_risk_free(rf, calendar)

        self.assertEqual(out["annual_pct"].tolist(), [4.60, 4.60, 4.62])
        self.assertAlmostEqual(out["daily"].iloc[0], 4.60 / 100 / TRADING_DAYS)


if __name__ == "__main__":
    unittest.main()
