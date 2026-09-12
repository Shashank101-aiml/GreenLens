from typing import Literal

from pydantic import BaseModel, Field

ControversyLevel = Literal["None", "Low", "Moderate", "Significant", "High", "Severe"]


class PredictRequest(BaseModel):
    description: str = Field(min_length=30, max_length=5000)
    sector: str = Field(min_length=2, max_length=60)
    employees: int | None = Field(default=None, ge=0, le=5_000_000)
    controversy_level: ControversyLevel | None = None
