"""Download the official UCI AI4I 2020 dataset."""
import argparse
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output",default="data/ai4i2020.csv"); a=p.parse_args()
    ds=fetch_ucirepo(id=601); X,y=ds.data.features,ds.data.targets
    if X is None or y is None: raise RuntimeError("UCI did not return both feature and target tables")
    df=pd.concat([X.reset_index(drop=True),y.reset_index(drop=True)],axis=1); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
    print(f"Saved {len(df):,} rows and {len(df.columns)} columns to {out}"); print("Source: UCI Machine Learning Repository, DOI 10.24432/C5HS5C (CC BY 4.0)")
if __name__=="__main__": main()
