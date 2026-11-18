"""Dataset loading and leakage-safe feature preparation."""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "product_type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
TARGET = "machine_failure"
_ALIASES = {
    "type": "product_type",
    "product type": "product_type",
    "air temperature [k]": "air_temperature_k",
    "air temperature": "air_temperature_k",
    "process temperature [k]": "process_temperature_k",
    "process temperature": "process_temperature_k",
    "rotational speed [rpm]": "rotational_speed_rpm",
    "rotational speed": "rotational_speed_rpm",
    "torque [nm]": "torque_nm",
    "torque": "torque_nm",
    "tool wear [min]": "tool_wear_min",
    "tool wear": "tool_wear_min",
    "machine failure": TARGET,
}


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map source-specific column labels to the project's stable schema."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    columns = {
        column: _ALIASES.get(str(column).strip().lower(), str(column).strip().lower())
        for column in frame.columns
    }
    normalized = frame.rename(columns=columns).copy()
    if normalized.columns.duplicated().any():
        duplicates = normalized.columns[normalized.columns.duplicated()].tolist()
        raise ValueError(f"Dataset contains duplicate columns after normalization: {duplicates}")
    return normalized


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select operational signals and validate the binary target without leakage."""
    if frame.empty:
        raise ValueError("Dataset must contain at least one row")

    clean = canonicalize_columns(frame)
    missing = sorted(set(FEATURES + [TARGET]).difference(clean.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = clean[FEATURES].copy()
    y = pd.to_numeric(clean[TARGET], errors="coerce")
    if y.isna().any():
        raise ValueError("Target contains missing or non-numeric values")
    if not y.isin([0, 1]).all():
        found = sorted(pd.unique(y).tolist())
        raise ValueError(f"Target must be binary 0/1; found: {found}")

    product_type = X["product_type"].astype("string").str.strip().str.upper()
    invalid_types = sorted(set(product_type.dropna().tolist()).difference({"L", "M", "H"}))
    if invalid_types:
        raise ValueError(f"Product type must be one of L, M or H; found: {invalid_types}")
    X["product_type"] = product_type.replace("", pd.NA)

    for column in FEATURES[1:]:
        numeric = pd.to_numeric(X[column], errors="coerce")
        X[column] = numeric.replace([np.inf, -np.inf], np.nan)
        if X[column].isna().all():
            raise ValueError(f"Numeric feature {column!r} contains no usable values")

    return X, y.astype("int64").rename(TARGET)


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV dataset and apply the same validation used by the training pipeline."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {source}. Run python scripts/download_data.py first."
        )
    return prepare_features(pd.read_csv(source))
