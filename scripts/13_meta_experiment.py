"""One bounded shallow boosted-tree experiment plus country transfer sanity checks."""
import argparse,json,time,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluate import evaluate
from src.predict import select_threshold,predict

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--sample',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cheap-only',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
sample={r['entity_id']:r for r in map(json.loads,a.sample.read_text().splitlines())};truth={i:set(r['matches']) for i,r in sample.items()}
split=json.loads((a.run/'split.json').read_text());base=json.loads((a.run/'validation.json').read_text())
arrays=np.load(a.run/'features.npz');x,y=arrays['features'],arrays['labels'];x=x[:,3:13] if a.cheap_only else x;pairs=pd.read_csv(a.run/'candidate_pairs_internal.tsv',sep='\t',dtype=str)
mask=pairs.source1_entity_id.isin(split['train']).to_numpy()
model=HistGradientBoostingClassifier(max_iter=200,max_depth=4,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1,early_stopping=False,random_state=2026)
model.fit(x[mask],y[mask]);pairs['score']=model.predict_proba(x)[:,1]
t,sweep=select_threshold(pairs,truth,split['calibration']);prediction=predict(pairs,t)
cal=evaluate(truth,prediction,split['calibration']);val=evaluate(truth,prediction,split['validation'])
slices={country:evaluate(truth,prediction,[i for i in split['validation'] if sample[i]['country']==country]) for country in ['India','US']}
for source in ['S2-','S3-']:
 slices[source]=evaluate({i:{v for v in ts if v.startswith(source)} for i,ts in truth.items()},{i:{v for v in ts if v.startswith(source)} for i,ts in prediction.items()},split['validation'])
previous=base['results']['logistic_features']['calibration']
keep=cal['macro_F0.5']>previous['macro_F0.5'] and cal['singleton_F0.5']>=previous['singleton_F0.5']-.02
report={'model':'HistGradientBoostingClassifier, shallow 200 iterations, depth 4','base_run':str(a.run),'threshold':t,'calibration':cal,'validation':val,'slices':slices,'threshold_sweep':sweep,'retained_by_calibration':keep,'runtime_seconds':time.monotonic()-start,'cheap_only':a.cheap_only,'feature_indices':list(range(3,13)) if a.cheap_only else list(range(14)),'notes':'No internal random pair holdout: early_stopping=False. S1 train/calibration/validation reused unchanged. Tree-only comparison; no neural training.'}
joblib.dump({'model':model,'threshold':t,'feature_names':base['feature_names'][3:13] if a.cheap_only else base['feature_names'],'source_run':str(a.run)},a.output/'model.joblib')
pairs.to_csv(a.output/'scores.tsv',sep='\t',index=False)
# Train country-specific logistic classifiers, calibrate on same origin country, evaluate other-country holdout ONCE.
transfer={}
for origin,destination in ([] if a.cheap_only else [('India','US'),('US','India')]):
 train=[i for i in split['train'] if sample[i]['country']==origin];calids=[i for i in split['calibration'] if sample[i]['country']==origin];valids=[i for i in split['validation'] if sample[i]['country']==destination]
 m=pairs.source1_entity_id.isin(train).to_numpy();clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,C=1,random_state=2026));clf.fit(x[m],y[m])
 scored=pairs.copy();scored['score']=clf.predict_proba(x)[:,1];threshold,_=select_threshold(scored,truth,calids)
 transfer[f'{origin}_to_{destination}']={'threshold':threshold,'train_s1':len(train),'calibration_s1':len(calids),'validation_s1':len(valids),'metrics':evaluate(truth,predict(scored,threshold),valids)}
report['country_transfer_logistic']=transfer
(a.output/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
