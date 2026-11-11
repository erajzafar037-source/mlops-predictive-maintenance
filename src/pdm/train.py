"""Train, compare, select and evaluate predictive-maintenance models."""
import argparse,json,os
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import average_precision_score,confusion_matrix,fbeta_score,precision_recall_curve,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split
from pdm import __version__
from pdm.data import FEATURES,load_dataset
from pdm.modeling import build_candidates
def choose_threshold(y_true: np.ndarray,probabilities: np.ndarray,beta: float=2.0) -> float:
    precision,recall,thresholds=precision_recall_curve(y_true,probabilities)
    if thresholds.size==0: return 0.5
    b2=beta**2; den=b2*precision[:-1]+recall[:-1]
    scores=np.divide((1+b2)*precision[:-1]*recall[:-1],den,out=np.zeros_like(den),where=den>0)
    return float(np.clip(thresholds[int(np.nanargmax(scores))],0.01,0.99))
def _metrics(y_true: np.ndarray,probabilities: np.ndarray,threshold: float) -> dict:
    pred=(probabilities>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y_true,pred,labels=[0,1]).ravel()
    return {"average_precision":float(average_precision_score(y_true,probabilities)),"roc_auc":float(roc_auc_score(y_true,probabilities)),"precision":float(precision_score(y_true,pred,zero_division=0)),"recall":float(recall_score(y_true,pred,zero_division=0)),"f2":float(fbeta_score(y_true,pred,beta=2,zero_division=0)),"threshold":float(threshold),"confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}}
def train(data_path: str|Path,output_dir: str|Path,random_state: int=42,track_mlflow: bool=False) -> dict:
    X,y=load_dataset(data_path)
    Xtv,Xtest,ytv,ytest=train_test_split(X,y,test_size=0.20,random_state=random_state,stratify=y)
    Xtr,Xv,ytr,yv=train_test_split(Xtv,ytv,test_size=0.25,random_state=random_state,stratify=ytv)
    fitted={}; scores={}
    for name,est in build_candidates(random_state).items():
        est.fit(Xtr,ytr); scores[name]=float(average_precision_score(yv,est.predict_proba(Xv)[:,1])); fitted[name]=est
    name=max(scores,key=scores.get); model=fitted[name]
    threshold=choose_threshold(yv.to_numpy(),model.predict_proba(Xv)[:,1])
    model.fit(Xtv,ytv); test_metrics=_metrics(ytest.to_numpy(),model.predict_proba(Xtest)[:,1],threshold)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":model,"model_name":name,"features":FEATURES,"threshold":threshold,"project_version":__version__},out/"model.joblib")
    report={"project_version":__version__,"created_at_utc":datetime.now(timezone.utc).isoformat(),"dataset":str(data_path),"dataset_rows":int(len(X)),"class_distribution":{str(k):int(v) for k,v in y.value_counts().sort_index().items()},"split_rows":{"train":int(len(Xtr)),"validation":int(len(Xv)),"test":int(len(Xtest))},"selection_metric":"validation_average_precision","validation_average_precision":scores,"selected_model":name,"threshold_selection":"validation_F2","test_metrics":test_metrics,"feature_columns":FEATURES,"excluded_columns":["UDI","UID","Product ID","TWF","HDF","PWF","OSF","RNF"],"warning":"AI4I is synthetic. Results are a research baseline, not evidence of real-factory performance."}
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); (out/"model_card.md").write_text(_model_card(report),encoding="utf-8")
    if track_mlflow: _log_mlflow(model,name,threshold,scores,test_metrics,out)
    return report
def _log_mlflow(model,name,threshold,scores,metrics,out):
    try:
        import mlflow,mlflow.sklearn
    except ImportError as exc: raise RuntimeError("Install requirements-mlops.txt to enable MLflow tracking") from exc
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")); mlflow.set_experiment("predictive-maintenance")
    with mlflow.start_run(run_name=name):
        mlflow.log_param("selected_model",name); mlflow.log_param("decision_threshold",threshold)
        mlflow.log_params({f"validation_ap_{k}":v for k,v in scores.items()})
        mlflow.log_metrics({f"test_{k}":v for k,v in metrics.items() if isinstance(v,(int,float))})
        mlflow.log_artifact(str(out/"metrics.json")); mlflow.sklearn.log_model(model,name="model")
def _model_card(report: dict) -> str:
    m=report["test_metrics"]
    return f"""# Model Card: Predictive Maintenance Classifier\n\n## Intended use\nEducational/research demonstration of binary machine-failure classification from process sensor readings.\n\n## Data\nAI4I 2020 from UCI. The dataset is synthetic and must not be treated as representative of a specific factory without external validation.\n\n## Training design\n- Stratified train/validation/test: 60% / 20% / 20%.\n- Candidate selection: validation average precision (PR-AUC).\n- Decision threshold: selected on validation data using F2 to emphasize recall.\n- Test set: used once for final reporting after model selection.\n- Identifier and failure-mode columns excluded to prevent leakage.\n\n## Selected model\n{name}\n\n## Held-out test metrics\n- Average precision: {m['average_precision']:.4f}\n- ROC-AUC: {m['roc_auc']:.4f}\n- Precision: {m['precision']:.4f}\n- Recall: {m['recall']:.4f}\n- F2: {m['f2']:.4f}\n- Threshold: {m['threshold']:.4f}\n- Confusion matrix: {m['confusion_matrix']}\n\n## Limitations\nFalse negatives may correspond to missed failures; false positives may cause unnecessary inspection. Thresholds require operational cost analysis. This model is not safety-certified and must not control industrial equipment.\n""".replace("{name}",report["selected_model"])
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data",default="data/ai4i2020.csv"); p.add_argument("--output",default="artifacts"); p.add_argument("--random-state",type=int,default=42); p.add_argument("--log-mlflow",action="store_true"); a=p.parse_args()
    print(json.dumps(train(a.data,a.output,a.random_state,a.log_mlflow),indent=2))
if __name__=="__main__": main()
