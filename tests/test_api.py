import numpy as np
from fastapi.testclient import TestClient

import pdm.api as api


class FakeModel:
    def predict_proba(self, frame):
        assert len(frame) == 1
        return np.array([[0.15, 0.85]])


def test_health_live():
    response = TestClient(api.app).get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_prediction_uses_threshold(monkeypatch):
    bundle = {
        "model": FakeModel(),
        "model_name": "unit-test-model",
        "features": [
            "product_type",
            "air_temperature_k",
            "process_temperature_k",
            "rotational_speed_rpm",
            "torque_nm",
            "tool_wear_min",
        ],
        "threshold": 0.7,
        "project_version": "test",
    }
    monkeypatch.setattr(api, "load_model_bundle", lambda: bundle)
    response = TestClient(api.app).post(
        "/predict",
        json={
            "product_type": "L",
            "air_temperature_k": 300.0,
            "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500,
            "torque_nm": 40,
            "tool_wear_min": 20,
        },
    )
    assert response.status_code == 200
    assert response.json()["failure_probability"] == 0.85
    assert response.json()["predicted_failure"] is True
    assert response.json()["decision_threshold"] == 0.7


def test_readiness_is_unavailable_without_model_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_MODEL_BUNDLE", None)
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing-model.joblib")
    response = TestClient(api.app).get("/health/ready")
    assert response.status_code == 503


def test_metrics_endpoint_exposes_prometheus_format():
    response = TestClient(api.app).get("/metrics")
    assert response.status_code == 200
    assert "pdm_predictions_total" in response.text
    assert "pdm_inference_duration_seconds" in response.text


def test_prediction_rejects_out_of_range_sensor_values():
    response = TestClient(api.app).post(
        "/predict",
        json={
            "product_type": "L",
            "air_temperature_k": 300.0,
            "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500,
            "torque_nm": -1,
            "tool_wear_min": 20,
        },
    )
    assert response.status_code == 422


def test_prediction_rejects_unexpected_fields():
    response = TestClient(api.app).post(
        "/predict",
        json={
            "product_type": "L",
            "air_temperature_k": 300.0,
            "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500,
            "torque_nm": 40,
            "tool_wear_min": 20,
            "machine_failure": 1,
        },
    )
    assert response.status_code == 422
