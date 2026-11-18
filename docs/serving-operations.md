# Model Serving and Operations Runbook

## Release artifact

A successful training run writes three artifacts:

- model.joblib: fitted preprocessing/model pipeline, selected model name, feature contract, decision threshold and project version.
- metrics.json: split metadata, validation candidate scores, selected model, threshold policy and held-out test metrics.
- model_card.md: intended use, evaluation design, held-out metrics and limitations.

Keep the three files together as one versioned release bundle. Store production artifacts in an access-controlled artifact store; do not commit trained binary models or sensitive datasets to Git.

## Train and validate

1. Download the attributed UCI AI4I 2020 benchmark using scripts/download_data.py.
2. Run training with a fixed random seed and retain the generated metrics and model card.
3. Review validation average precision for model selection and F2 for threshold selection.
4. Review held-out test metrics only after selection. Do not tune from the test set.
5. Review false-negative and false-positive counts with a domain expert before selecting an operational threshold.
6. Optionally log the run to MLflow, then retain the model, evaluation report and dataset version as one release candidate.

## Serve locally

Start the API after a model bundle exists:

```bash
uvicorn pdm.api:app --host 0.0.0.0 --port 8000
```

The readiness endpoint returns 503 until a valid model artifact is available. The API validates sensor fields and value ranges, rejects unknown fields, and returns a probability plus the thresholded decision. It is a research demonstration; it must not control machinery or maintenance scheduling without external validation and safety review.

## Metrics

- GET /metrics exposes Prometheus counters and the inference-duration histogram.
- pdm_predictions_total is labelled by the predicted failure outcome.
- pdm_inference_duration_seconds supports p50/p95-style latency analysis using histogram buckets.

The example scrape configuration assumes the API is reachable from the Prometheus host as host.docker.internal:8000. Replace this target with your Kubernetes Service or production service-discovery target.

## Promotion and rollback

Promote a candidate only after reviewing model-card limitations, leakage controls, data provenance, test metrics, probability calibration and the cost of false negatives/positives. Publish a versioned artifact and update the serving deployment to that artifact. Keep the previous approved bundle available. Roll back by restoring the previous artifact and restarting/rolling the API deployment; verify /health/ready and a known test prediction.

A production pipeline should add approval gates, artifact signing/checksums, drift monitoring, access controls, deployment telemetry and a documented rollback rehearsal. Those controls are roadmap items, not claims about this repository's current deployment.

## Incident checklist

- **Readiness is 503:** confirm MODEL_PATH exists and the bundle can be loaded.
- **Prediction returns 422:** compare the request fields and sensor ranges with the published schema.
- **Latency rises:** inspect model load/reload behaviour, CPU/memory pressure and request volume.
- **Outcome mix shifts:** investigate the incoming data distribution and labelling process; do not automatically retrain or change thresholds.
- **Rollback:** restore the last approved model bundle and verify a known request before returning traffic.
