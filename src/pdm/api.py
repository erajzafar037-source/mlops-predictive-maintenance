"""FastAPI inference service for the trained model bundle."""
import os
from pathlib import Path
from typing import Any
import joblib,pandas as pd
from fastapi import FastAPI,HTTPException
from pdm.schemas import PredictionRequest,PredictionResponse
MODEL_PATH=Path(os.getenv("MODEL_PATH","artifacts/model.joblib")); _MODEL_BUNDLE:dict[str,Any]|None=None
app=FastAPI(title="Predictive Maintenance API",version="0.1.0",description="Research demonstration API for machine-failure risk scoring.")
def load_model_bundle():
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        if not MODEL_PATH.is_file(): raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
        _MODEL_BUNDLE=joblib.load(MODEL_PATH)
    return _MODEL_BUNDLE
@app.get("/health/live")
def live(): return {"status":"alive"}
@app.get("/health/ready")
def ready():
    try: load_model_bundle()
    except FileNotFoundError as exc: raise HTTPException(status_code=503,detail="Model artifact is not available") from exc
    return {"status":"ready"}
@app.get("/model")
def model_info():
    try: b=load_model_bundle()
    except FileNotFoundError as exc: raise HTTPException(status_code=503,detail="Model artifact is not available") from exc
    return {"model_name":str(b["model_name"]),"project_version":str(b["project_version"])}
@app.post("/predict",response_model=PredictionResponse)
def predict(request:PredictionRequest):
    try: b=load_model_bundle()
    except FileNotFoundError as exc: raise HTTPException(status_code=503,detail="Model artifact is not available") from exc
    row=pd.DataFrame([request.model_dump()])[b["features"]]; probability=float(b["model"].predict_proba(row)[0,1]); threshold=float(b["threshold"])
    return PredictionResponse(failure_probability=probability,predicted_failure=probability>=threshold,decision_threshold=threshold,model_name=str(b["model_name"]),disclaimer="Research demonstration only; not validated for real industrial safety decisions.")
