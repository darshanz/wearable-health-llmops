from typing import Optional, Literal

from pydantic import BaseModel


class ReadinessPredictionRequest(BaseModel):
    steps_daily: Optional[float] = None
    sleep_minutes: Optional[float] = None
    time_in_bed: Optional[float] = None
    sleep_efficiency: Optional[float] = None
    hr_mean: Optional[float] = None
    hr_min: Optional[float] = None
    hr_max: Optional[float] = None
    hr_std: Optional[float] = None
    resting_heart_rate: Optional[float] = None
    calories_daily: Optional[float] = None
    mood: Optional[float] = None
    model_type: Literal["llm", "random_forest"] = "llm"


class ReadinessPredictionResponse(BaseModel):
    predicted_readiness: Optional[float] = None
    parse_success: bool
    prompt_version: str
    model_name: str
    error_message: Optional[str] = None
    raw_response: Optional[str] = None


class HealthResponse(BaseModel):
    status: str