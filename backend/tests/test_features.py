import unittest

import numpy as np
import pandas as pd

from ml.features import CONTROVERSY_ORDER, NUMERIC_COLS, TEXT_COL, SectorMeanRegressor, build_features, risk_level


class TestFeatures(unittest.TestCase):
    def test_risk_level_uses_sustainalytics_bands(self):
        cases = {0: "Negligible", 9.99: "Negligible", 10: "Low", 19.9: "Low", 20: "Medium", 35: "High", 40: "Severe", 55: "Severe"}
        for total, expected in cases.items():
            self.assertEqual(risk_level(total), expected, total)

    def test_sector_mean_baseline_falls_back_to_global_mean(self):
        model = SectorMeanRegressor().fit(pd.DataFrame({"sector": ["Energy", "Energy", "Technology"]}), [30.0, 40.0, 10.0])

        preds = model.predict(pd.DataFrame({"sector": ["Energy", "Technology", "Utilities"]}))

        np.testing.assert_allclose(preds, [35.0, 10.0, 80 / 3])

    def test_build_features_encodes_inputs(self):
        raw = pd.DataFrame(
            {
                "description": ["A regional bank offering loans and mortgages.", "A maker of solar panels and batteries."],
                "sector": ["Financial Services", None],
                "employees": [1000, None],
                "controversy_level": ["Moderate", None],
            }
        )

        feats = build_features(raw)

        self.assertTrue({TEXT_COL, "sector", *NUMERIC_COLS} <= set(feats.columns))
        self.assertEqual(feats.loc[0, "controversy_rank"], CONTROVERSY_ORDER.index("Moderate"))
        self.assertTrue(np.isnan(feats.loc[1, "controversy_rank"]))
        self.assertAlmostEqual(feats.loc[0, "log_employees"], np.log1p(1000))
        self.assertEqual(feats.loc[1, "sector"], "Unknown")
        self.assertGreater(feats.loc[0, "theme_finance"], 0)
        self.assertGreater(feats.loc[1, "theme_clean_energy"], 0)


if __name__ == "__main__":
    unittest.main()
