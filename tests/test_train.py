import numpy as np
import pandas as pd
from pdm.train import train
def test_training_writes_auditable_artifacts(tmp_path):
    rng=np.random.default_rng(17); n=500; torque=rng.normal(40,8,n); wear=rng.integers(0,250,n); z=-7.0+0.035*wear+0.035*torque; probs=1/(1+np.exp(-z)); y=rng.binomial(1,probs); y[:10]=1; y[10:30]=0
    f=pd.DataFrame({"UDI":np.arange(n),"Product ID":[f"L{i}" for i in range(n)],"Type":rng.choice(["L","M","H"],n),"Air temperature [K]":rng.normal(300,2,n),"Process temperature [K]":rng.normal(310,1,n),"Rotational speed [rpm]":rng.normal(1500,100,n),"Torque [Nm]":torque,"Tool wear [min]":wear,"Machine failure":y,"TWF":0,"HDF":0,"PWF":0,"OSF":0,"RNF":0})
    data=tmp_path/"sample.csv"; f.to_csv(data,index=False); out=tmp_path/"artifacts"; report=train(data,out)
    assert all((out/name).is_file() for name in ["model.joblib","metrics.json","model_card.md"]); assert report["split_rows"]=={"train":300,"validation":100,"test":100}; assert report["selected_model"] in {"logistic_regression","random_forest","hist_gradient_boosting"}; assert 0<=report["test_metrics"]["average_precision"]<=1
