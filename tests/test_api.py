import numpy as np
from fastapi.testclient import TestClient
import pdm.api as api
class FakeModel:
    def predict_proba(self,frame): assert len(frame)==1; return np.array([[0.15,0.85]])
def test_health_live():
    r=TestClient(api.app).get("/health/live"); assert r.status_code==200; assert r.json()["status"]=="alive"
def test_prediction_uses_threshold(monkeypatch):
    b={"model":FakeModel(),"model_name":"unit-test-model","features":["product_type","air_temperature_k","process_temperature_k","rotational_speed_rpm","torque_nm","tool_wear_min"],"threshold":0.7,"project_version":"test"}
    monkeypatch.setattr(api,"load_model_bundle",lambda:b)
    r=TestClient(api.app).post("/predict",json={"product_type":"L","air_temperature_k":300.0,"process_temperature_k":310.0,"rotational_speed_rpm":1500,"torque_nm":40,"tool_wear_min":20})
    assert r.status_code==200; assert r.json()["failure_probability"]==0.85; assert r.json()["predicted_failure"] is True; assert r.json()["decision_threshold"]==0.7
