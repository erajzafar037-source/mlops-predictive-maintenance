"""API request and response schemas."""
from pydantic import BaseModel,ConfigDict,Field
class PredictionRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    product_type: str=Field(pattern="^[LMH]$")
    air_temperature_k: float=Field(ge=250,le=350)
    process_temperature_k: float=Field(ge=250,le=370)
    rotational_speed_rpm: float=Field(gt=0,le=10000)
    torque_nm: float=Field(ge=0,le=200)
    tool_wear_min: float=Field(ge=0,le=1000)
class PredictionResponse(BaseModel):
    failure_probability: float
    predicted_failure: bool
    decision_threshold: float
    model_name: str
    disclaimer: str
