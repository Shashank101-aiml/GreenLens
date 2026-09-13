import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Settings:
    PROJECT_NAME = "ESG Performance Analytics API"
    VERSION = "2.0.0"
    BASE_DIR = BASE_DIR

    DATA_PATH = os.path.join(BASE_DIR, "SP_500_ESG_Risk_Ratings.csv")
    PRICES_DIR = os.path.join(BASE_DIR, "data", "raw", "prices")
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml", "artifacts")
    MODEL_PATH = os.path.join(ARTIFACTS_DIR, "esg_models.joblib")

    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://esg:esg@localhost:5434/esg")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    API_URL = os.getenv("API_URL", "http://localhost:8001")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")


settings = Settings()
