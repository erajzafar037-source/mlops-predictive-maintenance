import pandas as pd
import pytest

from pdm.data import FEATURES, prepare_features


def sample_frame():
    return pd.DataFrame({
        "UDI": [1, 2],
        "Product ID": ["L1", "M1"],
        "Type": ["L", "M"],
        "Air temperature [K]": [300.1, 301.0],
        "Process temperature [K]": [310.1, 311.0],
        "Rotational speed [rpm]": [1500, 1600],
        "Torque [Nm]": [40.0, 42.0],
        "Tool wear [min]": [10, 20],
        "Machine failure": [0, 1],
        "TWF": [0, 0],
        "HDF": [0, 0],
        "PWF": [0, 1],
        "OSF": [0, 0],
        "RNF": [0, 0],
    })


def test_prepare_features_excludes_ids_and_failure_sublabels():
    X, y = prepare_features(sample_frame())
    assert X.columns.tolist() == FEATURES
    assert "UDI" not in X.columns
    assert "product id" not in X.columns
    assert "twf" not in X.columns
    assert y.tolist() == [0, 1]


def test_prepare_features_rejects_non_binary_target():
    frame = sample_frame()
    frame.loc[0, "Machine failure"] = 2
    with pytest.raises(ValueError, match="binary"):
        prepare_features(frame)


def test_prepare_features_rejects_fractional_target_instead_of_truncating():
    frame = sample_frame()
    frame.loc[0, "Machine failure"] = 0.5
    with pytest.raises(ValueError, match="binary"):
        prepare_features(frame)


def test_prepare_features_rejects_missing_required_column():
    with pytest.raises(ValueError, match="missing required"):
        prepare_features(sample_frame().drop(columns=["Torque [Nm]"]))


def test_prepare_features_rejects_empty_dataset():
    with pytest.raises(ValueError, match="at least one row"):
        prepare_features(sample_frame().iloc[0:0])


def test_prepare_features_rejects_all_missing_numeric_feature():
    frame = sample_frame()
    frame["Torque [Nm]"] = None
    with pytest.raises(ValueError, match="no usable values"):
        prepare_features(frame)


def test_prepare_features_rejects_unknown_product_type():
    frame = sample_frame()
    frame.loc[0, "Type"] = "X"
    with pytest.raises(ValueError, match="Product type"):
        prepare_features(frame)
