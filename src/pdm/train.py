"""Train, compare, select and evaluate predictive-maintenance models."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, fbeta_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from pdm import __version__
from pdm.data import FEATURES, load_dataset
from pdm.modeling import build_candidates


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, beta: float = 2.0) -> float:
    """Choose a decision threshold on validation data using F-beta."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5
    beta_sq = beta ** 2
    denominator = beta_sq * precision[:-1] + recall[:-1]
    scores = np.divide(
        (1 + beta_sq) * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(np.clip(thresholds[int(np.nanargmax(scores))], 0.01, 0.99))


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train(data_path: str | Path, output_dir: str | Path, random_state: int = 42,
          track_mlflow: bool = False) -> dict:
    X, y = load_dataset(data_path)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val, y_train_val, test_size=0.25,
        random_state=random_state, stratify=y_train_val
    )

    fitted, validation_scores = {}, {}
    for name, estimator in build_candidates(random_state).items():
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_validation)[:, 1]
        validation_scores[name] = float(average_precision_score(y_validation, probabilities))
        fitted[name] = estimator

    selected_name = max(validation_scores, key=validation_scores.get)
    selected_model = fitted[selected_name]
    threshold = choose_threshold(
        y_validation.to_numpy(), selected_model.predict_proba(X_validation)[:, 1]
    )

    # Refit the selected pipeline on train + validation only; test remains untouched.
    selected_model.fit(X_train_val, y_train_val)
    test_metrics = _metrics(
        y_test.to_numpy(), selected_model.predict_proba(X_test)[:, 1], threshold
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": selected_model,
        "model_name": selected_name,
        "features": FEATURES,
        "threshold": threshold,
        "project_version": __version__,
    }, output / "model.joblib")

    report = {
        "project_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(data_path),
        "dataset_rows": int(len(X)),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "split_rows": {
            "train": int(len(X_train)),
            "validation": int(len(X_validation)),
            "test": int(len(X_test)),
        },
        "selection_metric": "validation_average_precision",
        "validation_average_precision": validation_scores,
        "selected_model": selected_name,
        "threshold_selection": "validation_F2",
        "test_metrics": test_metrics,
        "feature_columns": FEATURES,
        "excluded_columns": ["UDI", "UID", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
        "warning": "AI4I is synthetic. Results are a research baseline, not evidence of real-factory performance.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "model_card.md").write_text(_model_card(report), encoding="utf-8")
    if track_mlflow:
        _log_mlflow(selected_model, selected_name, threshold, validation_scores, test_metrics, output)
    return report


def _log_mlflow(model, model_name, threshold, validation_scores, test_metrics, output: Path) -> None:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError("Install requirements-mlops.txt to enable MLflow tracking") from exc

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("predictive-maintenance")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("selected_model", model_name)
        mlflow.log_param("decision_threshold", threshold)
        mlflow.log_params({f"validation_ap_{key}": value for key, value in validation_scores.items()})
        mlflow.log_metrics({
            f"test_{key}": value for key, value in test_metrics.items()
            if isinstance(value, (int, float))
        })
        mlflow.log_artifact(str(output / "metrics.json"))
        mlflow.sklearn.log_model(model, name="model")


def _model_card(report: dict) -> str:
    metrics = report["test_metrics"]
    return f"""# Model Card: Predictive Maintenance Classifier

## Intended use
Educational/research demonstration of binary machine-failure classification from process sensor readings.

## Data
AI4I 2020 Predictive Maintenance Dataset from UCI. It is synthetic and must not be treated as
representative of a specific factory without external validation.

## Training design
- Train/validation/test: 60% / 20% / 20%, stratified.
- Candidate selection: validation average precision (PR-AUC).
- Decision threshold: selected on validation data using F2 to emphasize recall.
- Test set: used once for final reporting after model selection.
- Identifier and failure-mode columns excluded to prevent leakage.

## Selected model
{report["selected_model"]}

## Held-out test metrics
- Average precision: {metrics["average_precision"]:.4f}
- ROC-AUC: {metrics["roc_auc"]:.4f}
- Precision: {metrics["precision"]:.4f}
- Recall: {metrics["recall"]:.4f}
- F2: {metrics["f2"]:.4f}
- Decision threshold: {metrics["threshold"]:.4f}
- Confusion matrix: `{metrics["confusion_matrix"]}`

## Limitations and risk
False negatives may correspond to missed failures; false positives may cause unnecessary inspection.
The operating threshold must be chosen with domain experts and real operational costs. This model is
not safety-certified and must not automatically control industrial equipment.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/ai4i2020.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-mlflow", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output, args.random_state, args.log_mlflow), indent=2))


if __name__ == "__main__":
    main()
