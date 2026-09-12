import os
import unittest

from fastapi.testclient import TestClient

from app.main import app
from core.config import settings

client = TestClient(app)
OIL = "The company explores for and produces crude oil and natural gas, and operates refineries and pipelines."
SOFTWARE = "The company develops cloud software and cybersecurity platforms for enterprise customers worldwide."


@unittest.skipUnless(os.path.exists(settings.MODEL_PATH), "model not trained; run python -m ml.train")
class TestModelAPI(unittest.TestCase):
    def predict(self, description: str, sector: str) -> dict:
        response = client.post("/model/predict", json={"description": description, "sector": sector, "employees": 20000})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_predict_returns_consistent_pillars(self):
        body = self.predict(OIL, "Energy")

        self.assertEqual(set(body["pillars"]), {"environment", "social", "governance"})
        self.assertAlmostEqual(body["total"], sum(body["pillars"].values()), places=1)
        self.assertIn(body["risk_level"], ["Negligible", "Low", "Medium", "High", "Severe"])
        self.assertGreater(body["themes"]["fossil_fuels"], 0)

    def test_oil_producer_has_higher_environmental_risk_than_software_firm(self):
        oil = self.predict(OIL, "Energy")
        software = self.predict(SOFTWARE, "Technology")

        self.assertGreater(oil["pillars"]["environment"], software["pillars"]["environment"])

    def test_predict_rejects_short_description(self):
        response = client.post("/model/predict", json={"description": "too short", "sector": "Energy"})

        self.assertEqual(response.status_code, 422)


class TestHealth(unittest.TestCase):
    def test_health_reports_components(self):
        body = client.get("/health").json()

        self.assertIn(body["database"], {"ok", "unavailable"})
        self.assertIn("model_version", body)


if __name__ == "__main__":
    unittest.main()
