"""FastAPI inference service for the trained model bundle."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from pdm.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
_MODEL_BUNDLE: dict[str, Any] | None = None

PREDICTIONS = Counter(
    "pdm_predictions_total",
    "Number of prediction requests grouped by predicted outcome.",
    ["predicted_failure"],
)
INFERENCE_LATENCY = Histogram(
    "pdm_inference_duration_seconds",
    "Time spent loading the model and scoring a prediction.",
)

app = FastAPI(
    title="Predictive Maintenance API",
    version="0.2.0",
    description="Research demonstration API for machine-failure risk scoring.",
)


def load_model_bundle() -> dict[str, Any]:
    """Load and minimally validate the model bundle once per process."""
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
        try:
            candidate = joblib.load(MODEL_PATH)
        except Exception as exc:
            raise RuntimeError("Model artifact could not be loaded") from exc
        required = {"model", "model_name", "features", "threshold", "project_version"}
        if not isinstance(candidate, dict) or not required.issubset(candidate):
            raise RuntimeError("Model artifact does not match the expected bundle schema")
        if not 0 <= float(candidate["threshold"]) <= 1:
            raise RuntimeError("Model artifact contains an invalid decision threshold")
        _MODEL_BUNDLE = candidate
    return _MODEL_BUNDLE


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> dict[str, str]:
    try:
        load_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifact is not available") from exc
    except Exception as exc:
        logger.exception("Model readiness check failed")
        raise HTTPException(status_code=503, detail="Model artifact is not valid") from exc
    return {"status": "ready"}


@app.get("/model")
def model_info() -> dict[str, str]:
    try:
        bundle = load_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifact is not available") from exc
    except Exception as exc:
        logger.exception("Model metadata could not be loaded")
        raise HTTPException(status_code=503, detail="Model artifact is not valid") from exc
    return {
        "model_name": str(bundle["model_name"]),
        "project_version": str(bundle["project_version"]),
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    started = time.perf_counter()
    try:
        bundle = load_model_bundle()
        row = pd.DataFrame([request.model_dump()])[bundle["features"]]
        probability = float(bundle["model"].predict_proba(row)[0, 1])
        threshold = float(bundle["threshold"])
        if not 0 <= probability <= 1:
            raise ValueError("Model returned a probability outside [0, 1]")
        predicted_failure = probability >= threshold
        PREDICTIONS.labels(predicted_failure=str(predicted_failure).lower()).inc()
        return PredictionResponse(
            failure_probability=probability,
            predicted_failure=predicted_failure,
            decision_threshold=threshold,
            model_name=str(bundle["model_name"]),
            disclaimer=(
                "Research demonstration only; not validated for real industrial "
                "safety decisions."
            ),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifact is not available") from exc
    except Exception as exc:
        logger.exception("Prediction inference failed")
        raise HTTPException(status_code=500, detail="Prediction could not be completed") from exc
    finally:
        INFERENCE_LATENCY.observe(time.perf_counter() - started)
