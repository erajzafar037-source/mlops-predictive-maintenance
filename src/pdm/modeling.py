"""Preprocessing and model candidates."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from pdm.data import FEATURES
NUMERIC_FEATURES=FEATURES[1:]; CATEGORICAL_FEATURES=FEATURES[:1]
def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    steps=[("imputer",SimpleImputer(strategy="median"))]
    if scale_numeric: steps.append(("scaler",StandardScaler()))
    numeric=Pipeline(steps); categorical=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric",numeric,NUMERIC_FEATURES),("categorical",categorical,CATEGORICAL_FEATURES)],remainder="drop")
def build_candidates(random_state: int=42) -> dict[str,Pipeline]:
    return {
      "logistic_regression":Pipeline([("preprocess",_preprocessor(True)),("classifier",LogisticRegression(class_weight="balanced",max_iter=2000,random_state=random_state))]),
      "random_forest":Pipeline([("preprocess",_preprocessor(False)),("classifier",RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced_subsample",random_state=random_state,n_jobs=-1))]),
      "hist_gradient_boosting":Pipeline([("preprocess",_preprocessor(False)),("classifier",HistGradientBoostingClassifier(max_iter=150,learning_rate=0.08,l2_regularization=1.0,random_state=random_state))]),
    }
