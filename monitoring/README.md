# Monitoring examples

The FastAPI service exposes Prometheus metrics at GET /metrics.

The prometheus.yml target assumes Prometheus runs in Docker Desktop and the API runs on the host at port 8000. Replace host.docker.internal:8000 with the appropriate address for your environment.

The alert-rules.yml file contains one illustrative p95 inference latency alert. It is not a production SLO. Establish alert thresholds from measured baseline traffic, load tests and service objectives before using them operationally.

Example query:

```promql
histogram_quantile(
  0.95,
  sum(rate(pdm_inference_duration_seconds_bucket[5m])) by (le)
)
```

Example outcome counter:

```promql
sum by (predicted_failure) (increase(pdm_predictions_total[1h]))
```
