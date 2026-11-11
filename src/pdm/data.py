"""Dataset loading and leakage-safe feature preparation."""
from pathlib import Path
import pandas as pd
FEATURES = ["product_type", "air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
TARGET = "machine_failure"
_ALIASES = {"type":"product_type","product type":"product_type","air temperature [k]":"air_temperature_k","air temperature":"air_temperature_k","process temperature [k]":"process_temperature_k","process temperature":"process_temperature_k","rotational speed [rpm]":"rotational_speed_rpm","rotational speed":"rotational_speed_rpm","torque [nm]":"torque_nm","torque":"torque_nm","tool wear [min]":"tool_wear_min","tool wear":"tool_wear_min","machine failure":TARGET}
def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={c:_ALIASES.get(str(c).strip().lower(),str(c).strip().lower()) for c in frame.columns}).copy()
def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    clean=canonicalize_columns(frame); missing=sorted(set(FEATURES+[TARGET]).difference(clean.columns))
    if missing: raise ValueError(f"Dataset is missing required columns: {missing}")
    X=clean[FEATURES].copy(); y=pd.to_numeric(clean[TARGET],errors="coerce")
    if y.isna().any(): raise ValueError("Target contains missing or non-numeric values")
    observed=set(y.astype(int).unique())
    if not observed.issubset({0,1}): raise ValueError(f"Target must be binary 0/1; found: {sorted(observed)}")
    X["product_type"]=X["product_type"].astype("string")
    for col in FEATURES[1:]: X[col]=pd.to_numeric(X[col],errors="coerce")
    if X[FEATURES[1:]].isna().all(axis=None): raise ValueError("All numeric feature values are missing")
    return X,y.astype("int64").rename(TARGET)
def load_dataset(path: str|Path) -> tuple[pd.DataFrame,pd.Series]:
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(f"Dataset not found at {p}. Run python scripts/download_data.py first.")
    return prepare_features(pd.read_csv(p))
