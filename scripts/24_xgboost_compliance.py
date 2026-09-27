"""Train the Apache-2.0 XGBoost replacement on the frozen v2 feature pool.

The grouped train/calibration/dev split and 16-feature missing-address schema are
reused exactly. Confirmation is optional so it cannot be touched accidentally
during parameter iteration.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluate import evaluate
from src.predict import predict,select_threshold

spec=importlib.util.spec_from_file_location('missing_address_helpers',ROOT/'scripts/22_missing_address_ablation.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--confirmation',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    original=pd.read_csv(a.run/'candidate_pairs_internal.tsv',sep='\t',dtype=str)
    allowed=set(zip(original.source1_entity_id,original.target_entity_id))
    pairs,x,y,truth=helpers.score_pairs(a.candidates/'pairs.jsonl',a.candidates/'sample_s1.jsonl',allowed)
    if len(pairs)!=len(original) or set(zip(pairs.source1_entity_id,pairs.target_entity_id))!=allowed:
        raise RuntimeError('Feature rows do not align with the frozen capped10 K20 pool')
    split=json.loads((a.run/'split.json').read_text());mask=pairs.source1_entity_id.isin(split['train']).to_numpy()
    model=xgb.XGBClassifier(objective='binary:logistic',eval_metric='logloss',n_estimators=200,max_depth=4,learning_rate=.08,reg_lambda=1,tree_method='hist',n_jobs=min(8,os.cpu_count() or 1),random_state=2026)
    model.fit(x[mask],y[mask],verbose=False);pairs['score']=model.predict_proba(x)[:,1]
    threshold,sweep=select_threshold(pairs,truth,split['calibration']);dev=evaluate(truth,predict(pairs,threshold),split['validation'])
    confirmation=None
    if a.confirmation:
        cpairs,cx,_,ctruth=helpers.score_pairs(a.confirmation/'pairs.jsonl',a.confirmation/'sample_s1.jsonl')
        cpairs['score']=model.predict_proba(cx)[:,1];confirmation=evaluate(ctruth,predict(cpairs,threshold))
    dist=importlib.metadata.distribution('xgboost')
    report={'model':'xgboost.XGBClassifier','xgboost_version':xgb.__version__,'xgboost_license':dist.metadata.get('License'),'parameters':model.get_params(),'training_split_entities':len(split['train']),'training_pairs':int(mask.sum()),'calibration_entities':len(split['calibration']),'dev_entities':len(split['validation']),'dev_pairs':len(pairs),'feature_schema':'ordered 16 missing-address-v2 features; unavailable address comparisons are NaN; address_missing retained','threshold_source':'macro F0.5 sweep on frozen calibration S1 IDs only','threshold':float(threshold),'dev':dev,'validation':dev,'confirmation':confirmation,'confirmation_threshold_tuned':False,'cheap_only':True,'uses_fuzzy':True,'neutral_missing_address':True,'run_seconds':time.monotonic()-start,'python':platform.python_version(),'platform':platform.platform(),'git_commit_at_run':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    joblib.dump({'model':model,'threshold':float(threshold),'uses_fuzzy':True,'neutral_missing_address':True,'feature_count':16,'model_license':'Apache-2.0'},a.output/'model.joblib')
    (a.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'threshold_sweep.json').write_text(json.dumps(sweep,indent=2)+'\n')
    print(json.dumps({'threshold':threshold,'dev':dev,'confirmation':confirmation,'run_seconds':report['run_seconds']},indent=2))


if __name__=='__main__':main()
